import torch
import types
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from ..modifier import Modifier
from .utils import check_and_apply_qk_rope
from flash_attn import flash_attn_func


def model_forward(self, input_ids, input_embeds, kv_cache):
    hidden_states, kv_cache = self.model(input_ids, input_embeds, kv_cache)

    logits = self.lm_head(hidden_states)
    latent_states = self.latent_head(hidden_states[..., -1:, :])

    return dict(
        logits=logits,
        latent_states=latent_states,
        kv_cache=kv_cache)



def model_model_forward(self, input_ids, input_embeds, kv_cache):

    assert input_ids is None or input_embeds is None

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


class ModelForTraining(Modifier):


    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):

        self.get_conf(config)
        model = self._replace_foward_functions(model)
        model.train()
        super().__init__(model, save_ckp, load_ckp)


    def _replace_foward_functions(self, model):
        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)

        # build latent head
        model.latent_head = torch.nn.Linear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.in_features,
            bias=False,
            device='cuda',
            dtype=torch.bfloat16)
        
        torch.nn.init.normal_(model.latent_head.weight.data, std=0.01)


        for layer in model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

        return model


    def ft_params(self):
        params = list(self.model.latent_head.parameters())
        return params
