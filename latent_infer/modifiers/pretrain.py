import torch
import types
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from .utils import check_and_apply_qk_rope, maybe_zero_3
from peft import LoraConfig, TaskType, get_peft_model

from flash_attn import flash_attn_func
import json



def model_forward(self, input_ids, position_ids, attention_mask, **kwargs):
    hidden_states = self.model(input_ids, position_ids, attention_mask)
    logits = self.lm_head(hidden_states)
    return logits


def model_model_forward(self, input_ids, position_ids, attention_mask):
    batch_size, num_tokens = input_ids.shape
    
    # embedding
    hidden_states = torch.empty(
        (batch_size, num_tokens, self.embed_tokens.weight.shape[-1]),
        dtype=torch.bfloat16,
        device='cuda')

    for i, data in enumerate(input_ids):
        latent_mask = data == -100
        ordinal_input_ids = data[~latent_mask]
        ordinal_embed = self.embed_tokens(ordinal_input_ids.unsqueeze(0))
        hidden_states[i, ~latent_mask, :] = ordinal_embed
        hidden_states[i, latent_mask, :] = self.latent_embed.squeeze(0).expand_as(hidden_states[i, latent_mask, :])


    for layer in self.layers:
        hidden_states = layer(hidden_states, position_ids, attention_mask)

    hidden_states = self.norm(hidden_states)

    return hidden_states


def layer_forward(self, hidden_states, position_ids, attention_mask):    
    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def do_projection(proj, states, num_heads, head_dim, disable_lora):
    if disable_lora and hasattr(proj, 'base_layer'):
        return proj.base_layer(states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    else:
        return proj(states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)


def maybe_latent_proj(self, hidden_states, position_ids):
    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, True)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, True)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, True)    

    if -100 in position_ids:

        latent_mask = position_ids == -100
        
        ques_latent = do_projection(self.q_proj, hidden_states, num_heads, head_dim, False)
        keys_latent = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, False)
        vals_latent = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, False)

        ques = ques.transpose(1,2)
        keys = keys.transpose(1,2)
        vals = vals.transpose(1,2)
    
        ques_latent = ques_latent.transpose(1,2)
        keys_latent = keys_latent.transpose(1,2)
        vals_latent = vals_latent.transpose(1,2)

        ques[latent_mask, :, :] = ques_latent[latent_mask, :, :]
        keys[latent_mask, :, :] = keys_latent[latent_mask, :, :]
        vals[latent_mask, :, :] = vals_latent[latent_mask, :, :]

        ques = ques.transpose(1,2)
        keys = keys.transpose(1,2)
        vals = vals.transpose(1,2)

    return ques, keys, vals


def self_attn_forward(self, hidden_states, position_ids, attention_mask):
    num_kv_group = self.config.num_attention_heads // self.config.num_key_value_heads

    ques, keys, vals = self.maybe_latent_proj(hidden_states, position_ids)

    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys.shape[-2])
    cos, sin = self.rotary_emb(keys, seq_len=max(len1, len2))
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin, position_ids)

    keys_expand = repeat_kv(keys, num_kv_group)
    vals_expand = repeat_kv(vals, num_kv_group)


    if attention_mask is None:
        ques = ques.transpose(1,2)
        keys_expand = keys_expand.transpose(1,2)
        vals_expand = vals_expand.transpose(1,2)

        attn_output = flash_attn_func(
            q=ques, 
            k=keys_expand, 
            v=vals_expand,
            causal=True)
        
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=ques,
            key=keys_expand,
            value=vals_expand,
            attn_mask=attention_mask)
        attn_output = attn_output.transpose(1,2)

    attn_output= attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output




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


    def _replace_foward_functions(self):
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        
        latent_embed = torch.empty(
            size=(1,1,self.model.config.hidden_size), 
            dtype=torch.bfloat16, 
            device='cuda')
        self.model.model.latent_embed = torch.nn.Parameter(latent_embed, requires_grad=True)

        for layer in self.model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.self_attn.maybe_latent_proj = types.MethodType(maybe_latent_proj, layer.self_attn)

    
    def get_model(self):
        if self.conf['lora']['enable'] is True:
            return self.model.model
        return self.model
        

    def init_latent_embed(self, token_id):
        embed = self.get_model().model.embed_tokens.weight[token_id]
        self.get_model().model.latent_embed.data = embed.reshape(1,1,-1)
        self.get_model().model.latent_embed.requires_grad_(True)

    
    def load_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.load_ckp
        checkpoint = torch.load(ckp, map_location="cpu")
        for param1, param2 in zip(self.ft_params(), checkpoint):
            param1.data = param2.data.to(device=param1.data.device, dtype=param1.data.dtype)


    def save_checkpoint(self, ckp: str = None):
        ckp = ckp if ckp is not None else self.save_ckp
        torch.save([maybe_zero_3(param) for param in self.ft_params()], ckp)


    def forward(self, input_ids, labels, position_ids, attention_mask=None):
        input_ids = input_ids.cuda()
        labels = labels.cuda()

        logits = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

        logits = logits.flatten(0,1)
        labels = labels.flatten(0,1)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        with torch.no_grad():
            detailed_loss = torch.nn.functional.cross_entropy(logits, labels, reduce=False)

        return dict(loss=loss, detailed_loss=detailed_loss)


    def ft_params(self):
        params = [self.get_model().model.latent_embed]

        if self.conf['lora']['enable'] is True:
            for layer in self.get_model().model.layers:
                params += [
                    layer.self_attn.q_proj.lora_A.default.weight,
                    layer.self_attn.q_proj.lora_B.default.weight,
                    layer.self_attn.v_proj.lora_A.default.weight,
                    layer.self_attn.v_proj.lora_B.default.weight]
                
        return params
