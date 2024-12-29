import torch
import types
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from .utils import check_and_apply_qk_rope, maybe_zero_3
from peft import LoraConfig, TaskType, get_peft_model

from flash_attn import flash_attn_func
import json


def model_forward(self, input_ids, kv_cache, reduce_logits=True, **kwargs):

    hidden_states, kv_cache = self.model(input_ids, kv_cache)

    if reduce_logits:
        hidden_states = hidden_states[:, -1:, :]

    logits = self.lm_head(hidden_states)

    return hidden_states, logits, kv_cache



def model_model_forward(self, input_ids, kv_cache):
    if input_ids is None:
        input_embeds = self.latent_query
    else:
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


class ModelForTraining(torch.nn.Module):

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

        super().__init__()

        self.save_ckp = save_ckp
        self.load_ckp = load_ckp
        self.model = model

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
            device='cuda')
        self.model.model.latent_query = torch.nn.Parameter(latent_query, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.model.model.latent_query.data)


        # build reinforcement learning head
        self.model.flag_head = torch.nn.Linear(
            in_features=self.model.lm_head.in_features, 
            out_features=1,
            dtype=torch.bfloat16,
            device='cuda')
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
            return self.model.model.model.layers
        return self.model.model.layers
    

    def rl_params(self):
        if self.conf['lora']['enable'] is True:
            return list(self.model.base_model.flag_head.parameters())
        else:
            return list(self.model.flag_head.parameters())


    def lm_params(self):
        if self.conf['lora']['enable'] is True:
            params = [self.model.model.model.latent_query]
        else:
            params = [self.model.model.latent_query]

        if self.conf['lora']['enable'] is True:
            for layer in self._get_layers():

                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight]

        return params


    def _compute_nll_and_flag(self, hidden_states):
        logits = self.model.flag_head(hidden_states.detach()).ravel().float()
        nll = -torch.nn.functional.logsigmoid(logits)
        probs = logits.detach().sigmoid()
        flag = (probs > torch.rand(1, dtype=probs.dtype, device=probs.device)).item()
        return nll if flag else logits + nll, flag

    
    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)


    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save([maybe_zero_3(param) for param in self.ft_params()], ckp)


    def forward(self, input_ids=None, label=None, kv_cache=None, **kwargs):
        hidden_states, logits, kv_cache = self.model(
            input_ids=input_ids, 
            kv_cache=kv_cache)
        
        nll, flag = self._compute_nll_and_flag(hidden_states)

        loss = None
        if label is not None:
            label = torch.tensor(label, dtype=torch.int64, device='cuda')
            loss = torch.nn.functional.cross_entropy(logits.ravel(), label)
                

        return dict(
            kv_cache=kv_cache,
            loss=loss,
            nll=nll,
            flag=flag)


    def ft_params(self):
        return self._rl_params() + self._lm_params()
