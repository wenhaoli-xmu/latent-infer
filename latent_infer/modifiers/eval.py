import torch
import types
import torch.distributed
from transformers.models.llama.modeling_llama import repeat_kv, CrossEntropyLoss
from ..modifier import Modifier
from .utils import Conv, check_and_apply_qk_rope, slerp



def prune_labels(labels, masks):
    valid_masks = list(filter(lambda x: x is not None, masks))

    num_masked_out = sum([mask.sum().item() for mask in valid_masks])
    num_labels = labels.shape[-1]

    first_label, other_labels = labels[..., :1], labels[..., 1:]

    for mask in valid_masks:
        other_labels = other_labels[~mask].unsqueeze(0)
    
    labels = torch.cat([first_label, other_labels], dim=-1)

    assert labels.shape[-1] == num_labels - num_masked_out, f"element count mismatch when conducting pruning to labels."


    return labels



def model_forward(
        self, 
        input_ids, 
        labels, 
        return_logits, 
        sample_pruning, 
        deterministic_pruning, 
        trainable_tokens,
        merge_method):

    # model forward function
    hidden_states, logits, masks = self.model(
        input_ids=input_ids,
        return_logits=return_logits,
        sample_pruning=sample_pruning,
        deterministic_pruning=deterministic_pruning,
        merge_method=merge_method)

    output = self.lm_head(hidden_states).float()

    if len(sample_pruning) + len(deterministic_pruning) > 0 and labels is not None:
        labels = prune_labels(labels, masks)

    if trainable_tokens is not None:
        output = output[..., -trainable_tokens:, :]
        labels = labels[..., -trainable_tokens:]

    loss = None
    shift_output = output[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss()
    shift_output = shift_output.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_output.device)
    loss = loss_fct(shift_output, shift_labels)

    return loss, logits, masks



def model_model_forward(
        self, 
        input_ids, 
        return_logits, 
        sample_pruning, 
        deterministic_pruning,
        merge_method):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    logits = [None] * len(self.layers)
    masks = [None] * len(self.layers)

    for layer_idx, layer in enumerate(self.layers):

        hidden_states, logit, mask = layer(
            hidden_states,
            layer_idx in return_logits,
            layer_idx in sample_pruning, 
            layer_idx in deterministic_pruning,
            merge_method,)

        logits[layer_idx] = logit
        masks[layer_idx] = mask
        
    hidden_states = self.norm(hidden_states)

    return hidden_states, logits, masks



def prune(hidden_states, logits, sampling, merge_method):

    if sampling:
        mask = logits.sigmoid() > torch.rand_like(logits)
    else:
        mask = logits > 0

    mask_expand = mask.unsqueeze(-1).expand_as(hidden_states[:, 1:, :])

    before = hidden_states.clone()


    if merge_method == 'add':
        hidden_states[:, :-1, :][mask_expand] += hidden_states[:, 1:, :][mask_expand]

    elif merge_method == 'avg':
        hidden_states[:, :-1, :][mask_expand] = hidden_states[:, :-1, :][mask_expand] * 0.5 + hidden_states[:, 1:, :][mask_expand] * 0.5

    elif merge_method == 'slerp':
        batch_size, _, hidden_size = hidden_states.shape
        assert batch_size == 0, f"only support 1 batch size currently."

        v0 = hidden_states[:, :-1, :][mask_expand].reshape(-1, hidden_size)
        v1 = hidden_states[:, 1:, :][mask_expand].reshape(-1, hidden_size)

        merged = slerp(0.5, v0, v1)
        hidden_states = merged.reshape(1, -1, hidden_size)

    else:
        raise NotImplementedError


    if torch.allclose(before, hidden_states) and mask.int().sum().item() > 0:
        from pygments.console import colorize
        print(colorize("yellow", "Warning: ") + "merge operation dose not change the value of hidden states, which is abnormal.", flush=True)
    

    hidden_size = hidden_states.shape[-1]
    first_hidden_states, other_hidden_states = hidden_states[:, :1, :], hidden_states[:, 1:, :]
    other_hidden_states = other_hidden_states[~mask_expand].reshape(1, -1, hidden_size)
    hidden_states = torch.cat([first_hidden_states, other_hidden_states], dim=1)

    return hidden_states, mask



def layer_forward(
        self, 
        hidden_states: torch.Tensor, 
        return_logit: bool, 
        enable_sampling_pruning: bool, 
        enable_deterministic_pruning: bool,
        merge_method: str
):    
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)


    assert not (enable_sampling_pruning and enable_deterministic_pruning), f"sampling and deterministic can not be conducted simutanously"


    # maybe prune
    logit = self.conv(hidden_states)


    if enable_sampling_pruning or enable_deterministic_pruning:
        hidden_states, mask = prune(
            hidden_states, 
            logit, 
            sampling=enable_sampling_pruning, 
            merge_method=merge_method)


    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states


    return (
        hidden_states,
        logit if return_logit else None,
        mask if enable_sampling_pruning or enable_deterministic_pruning else None)



def self_attn_forward(self, hidden_states: torch.Tensor):

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

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)

    len1 = self.config.max_position_embeddings if hasattr(self.config, "max_position_embeddings") else 0
    len2 = max(ques.shape[-2], keys.shape[-2])
    cos, sin = self.rotary_emb(keys, seq_len=max(len1, len2))

    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=ques,
        key=keys,
        value=vals,
        is_causal=True)

    attn_output = attn_output.transpose(1,2).flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class Decoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.decoder.forward = types.MethodType(model_forward, self.decoder)
        self.decoder.model.forward = types.MethodType(model_model_forward, self.decoder.model)


        for layer in self.decoder.model.layers:
            info = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype}

            # modify the forward function
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.conv = Conv(hidden_size=4096, kernel_size=2)
            layer.conv.to(**info)


    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder


    def forward(
            self, 
            input_ids, 
            labels=None, 
            return_logits=[], 
            sample_pruning=[], 
            deterministic_pruning=[], 
            trainable_tokens=None,
            merge_method='avg'):

        assert input_ids.ndim == 2, f"input_ids must be 2 dimentional, but got {input_ids.ndim}"

        device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.decoder(
            input_ids, 
            labels=labels,
            return_logits=return_logits,
            sample_pruning=sample_pruning,
            deterministic_pruning=deterministic_pruning,
            trainable_tokens=trainable_tokens,
            merge_method=merge_method)

        return outputs


class ModelForEvaluation(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        self.get_conf(config)
        decoder = Decoder(model)
        decoder = Model(decoder)
        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        params = []
        for layer in self.model.decoder.decoder.model.layers:
            params += layer.conv.get_params()
        return params
    

    def layer_ft_params(self, layer_idx: int):
        layer = self.model.decoder.decoder.model.layers[layer_idx]
        return layer.conv.get_params()
