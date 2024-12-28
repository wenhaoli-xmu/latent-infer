import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope
from peft import LoraConfig, TaskType, get_peft_model

from flash_attn import flash_attn_func
import json, logging
from torch.utils.checkpoint import checkpoint
from functools import partial


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



def model_forward(self, input_ids, input_embeds, kv_cache):
    assert input_ids is None or input_embeds is None, f"cannot assign `input_ids` and `input_embeds` simutanously"

    hidden_states, kv_cache = self.model(input_ids, input_embeds, kv_cache)
    hidden_states = hidden_states[:, -1:, :]

    logits = self.lm_head(hidden_states)

    return hidden_states, logits, kv_cache



def model_model_forward(self, input_ids, input_embeds, kv_cache):
    if input_embeds is None:
        input_embeds = self.embed_tokens(input_ids)

    if kv_cache is None:
        kv_cache = [[None, None] for _ in range(len(self.layers))]

    hidden_states = input_embeds

    for layer in self.layers:
        hidden_states, kv_cache = layer(
            hidden_states, 
            kv_cache)

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache


def layer_forward(self, hidden_states, kv_cache):    
    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, kv_cache = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache



def self_attn_forward(self, hidden_states, kv_cache):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    def do_projection(proj, states, num_heads, head_dim):
        return proj(states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim)

    if kv_cache[self.layer_idx][0] is not None:
        keys = torch.cat([kv_cache[self.layer_idx][0], keys], dim=-2)
        vals = torch.cat([kv_cache[self.layer_idx][1], vals], dim=-2)

    kv_cache[self.layer_idx][0] = keys
    kv_cache[self.layer_idx][1] = vals

    keys_expand = repeat_kv(keys, num_kv_group)
    vals_expand = repeat_kv(vals, num_kv_group)

    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys_expand.shape[-2])
    cos, sin = self.rotary_emb(keys_expand, seq_len=max(len1, len2))

    ques, keys_expand = check_and_apply_qk_rope(ques, keys_expand, cos, sin)


    ques = ques.transpose(1,2)
    keys_expand = keys_expand.transpose(1,2)
    vals_expand = vals_expand.transpose(1,2)


    attn_output = flash_attn_func(
        q=ques, 
        k=keys_expand, 
        v=vals_expand,
        causal=True)

    attn_output= attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output, kv_cache


class ModelForTraining:

    def _get_conf(self, config):
        if config is not None:
            with open(config, 'r') as f:
                conf = json.load(f)
            print("=" * 40 + " Config " + "=" * 40)
            print(json.dumps(conf, indent=4
                ).replace("\n    ", "\n"
                ).replace("{", ""
                ).replace("}", ""
                ).strip().replace('"', ''))
            print("=" * 88)
            self.conf = conf
        else:
            self.conf = None


    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.save_ckp = save_ckp
        self.load_ckp = load_ckp
        self.model = model
        self.device = next(iter(model.parameters())).device

        self._get_conf(config)
        self._replace_foward_functions()
        self._maybe_enable_lora()

        self.model.train()



    def _replace_foward_functions(self):
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        
        # build latent query parameter
        latent_query = torch.empty(
            size=(1,1,self.model.config.hidden_size), 
            dtype=torch.bfloat16, 
            device=self.device)
        self.model.latent_query = torch.nn.Parameter(latent_query, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.model.latent_query.data)


        # build reinforcement learning head
        self.model.flag_head = torch.nn.Linear(
            in_features=self.model.lm_head.in_features, 
            out_features=1,
            dtype=torch.bfloat16,
            device=self.device)
        torch.nn.init.xavier_uniform_(self.model.flag_head.weight)
        torch.nn.init.constant_(self.model.flag_head.bias, val=-1.0)


        for layer in self.model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
    


    def _maybe_enable_lora(self):
        if self.conf['lora']['enable'] is True:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.conf['lora']['r'],
                lora_alpha=self.conf['lora']['alpha'],
                lora_dropout=self.conf['lora']['dropout'],
                target_modules=['q_proj', 'v_proj']
            )
            self.model = get_peft_model(self.model, peft_config)
        

    def _get_layers(self):
        if self.conf['lora']['enable'] is True:
            return self.model.base_model.model.layers
        return self.model.model.layers
    

    def _rl_params(self):
        return list(self.model.flag_head.parameters())


    def _lm_params(self):
        params = [self.model.latent_query]

        if self.conf['lora']['enable'] is True:
            for layer in self._get_layers():
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.k_proj.lora_A.default.weight,
                    layer.self_attn.k_proj.lora_B.default.weight]

        return params
    

    def _zero_grad(self):
        for param in self.ft_params():
            if param.grad is not None:
                param.grad.data.zero_()


    def _compute_nll_and_flag(self, hidden_states):
        logits = self.model.flag_head(hidden_states.detach()).ravel().float()
        nll = -torch.nn.functional.logsigmoid(logits)
        probs = logits.detach().sigmoid()
        flag = (probs > torch.rand(1, dtype=probs.dtype, device=probs.device)).item()
        return nll if flag else logits + nll, flag
    

    def _greedy_decode(self, logits):
        return logits.argmax(dim=-1).reshape(1,1)
    

    def _is_eos_token(self, next_token, eos_token_id):
        return next_token.ravel().item() in eos_token_id
    

    def _split_input_ids(self, input_ids):
        return input_ids[:, -1:], input_ids[:, :-1]
    

    def _slice(self, x, start, end):
        return torch.tensor(x[start:end], dtype=torch.int64).unsqueeze(0).to(self.device)
    

    def _compute_lm_loss(self, logits, label):
        label = torch.tensor(label, dtype=torch.int64, device=self.device)
        lm_loss = torch.nn.functional.cross_entropy(logits.ravel(), label)
        return lm_loss.unsqueeze(0)
    

    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)


    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save([maybe_zero_3(param) for param in self.ft_params()], ckp)


    def enable_fsdp(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        class_type = type(self._get_layers()[0])

        my_auto_wrap_policy = partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls=set([class_type]))

        self.model = FSDP(
            module=self.model, 
            auto_wrap_policy=my_auto_wrap_policy)
        
        torch.cuda.empty_cache()
        

    def no_sync(self):
        return self.model.no_sync()
        


    def ft_params(self):
        params = list(self.model.flag_head.parameters())
        params.append(self.model.latent_query)

        if self.conf['lora']['enable'] is True:
            for layer in self._get_layers():
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.k_proj.lora_A.default.weight,
                    layer.self_attn.k_proj.lora_B.default.weight]

        return params


    def sample(self, input_ids: list, labels: list):

        """
        假设input-ids和labels是已经错开的了
        """

        assert isinstance(input_ids, list)
        assert isinstance(labels, list)
        assert labels[0] == -100


        self._zero_grad()
        self.device = next(iter(self.model.parameters())).device

        start = 0
        kv_cache = None
        rl_losses, lm_losses = [], []
        flags = []


        while start < len(input_ids):
            if labels[start] == -100:
                # parallel computation without gradient
                end = start + 1
                while labels[end] == -100:
                    end += 1

                with torch.no_grad():
                    # no need to compute gradient in pre-filling phase
                    _, _, kv_cache = self.model(
                        input_ids=self._slice(input_ids, start, end - 1),
                        input_embeds=None,
                        kv_cache=kv_cache)
                
                hidden_states, logits, kv_cache = self.model(
                    input_ids=self._slice(input_ids, end - 1, end),
                    input_embeds=None,
                    kv_cache=kv_cache)
                
                start = end

            else:
                # decoding phase
                if flag is True:
                    rl_losses.append(nll)

                    hidden_states, logits, kv_cache = self.model(
                        input_ids=None,
                        input_embeds=self.model.latent_query,
                        kv_cache=kv_cache)
                else:
                    lm_loss = self._compute_lm_loss(logits, labels[start])
                    lm_losses.append(lm_loss)

                    hidden_states, logits, kv_cache = self.model(
                        input_ids=self._slice(input_ids, start, start + 1),
                        input_embeds=None,
                        kv_cache=kv_cache)
                
                    start += 1
                    
            nll, flag = self._compute_nll_and_flag(hidden_states)
            flags.append(flag)


        ratio = sum(flags) / len(flags)


        rl_loss = torch.cat(rl_losses).mean() if len(rl_losses) > 1 else rl_losses[0].mean()
        lm_loss = torch.cat(lm_losses).mean() if len(lm_losses) > 1 else lm_losses[0].mean()
            

        (rl_loss + lm_loss).backward()


        reward = lm_loss.item() + self.conf['reward']['alpha'] * ratio
        reward = torch.tensor(reward, dtype=torch.bfloat16, device=self.device)

        # 收集
        # 1. rl的gradient
        # 2. lm的gradient
        # 3. latent output占比

        rl_grads = []
        lm_grads = []

        for param in self._rl_params():
            cond = param.grad is not None
            cond = cond and param.grad.data.count_nonzero().sum() > 0
            
            if cond:
                rl_grads.append(param.grad.data.ravel())
            else:
                rl_grads.append(torch.zeros_like(param.data).ravel())

        
        for param in self._lm_params():
            cond = param.grad is not None
            cond = cond and param.grad.data.count_nonzero().sum() > 0

            if cond:
                lm_grads.append(param.grad.data.ravel())
            else:
                lm_grads.append(torch.zeros_like(param.data).ravel())


        return torch.cat(rl_grads, dim=0), torch.cat(lm_grads), reward
