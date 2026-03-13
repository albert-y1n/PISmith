"""
This module incorporates code from the AT2 codebase.
"""

import math
from typing import Any, Optional
import torch
import transformers.models


def infer_model_type(model):
    model_type_to_keyword = {
        "llama": "llama",
        "gpt": "gpt_oss",
        "glm": "glm",
        "phi": "phi3",
        "qwen2": "qwen2",
        "qwen3": "qwen3",
        "gemma": "gemma3",
    }
    for keyword, model_type in model_type_to_keyword.items():
        if keyword in model.name_or_path.lower():
            return model_type
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}. Specify `model_type`.")


def get_helpers(model_type):
    #for model_name in dir(transformers.models):
    #     if not model_name.startswith('__') and ("gemma" in model_name or "chatglm" in model_name):
    #         print(model_name)
    if not hasattr(transformers.models, model_type):
        raise ValueError(f"Unknown model: {model_type}")
    model_module = getattr(transformers.models, model_type)
    modeling_module = getattr(model_module, f"modeling_{model_type}")
    return modeling_module.apply_rotary_pos_emb, modeling_module.repeat_kv


def get_position_ids_and_attention_mask(model, hidden_states):
    input_embeds = hidden_states[0]
    _, seq_len, _ = input_embeds.shape
    position_ids = torch.arange(0, seq_len, device=model.device).unsqueeze(0)
    attention_mask = torch.ones(
        seq_len, seq_len + 1, device=model.device, dtype=model.dtype
    )
    attention_mask = torch.triu(attention_mask, diagonal=1)
    attention_mask *= torch.finfo(model.dtype).min
    attention_mask = attention_mask[None, None]
    return position_ids, attention_mask


def get_attentions_shape(model):
    try:
        num_layers = len(model.model.layers)
        num_heads = model.model.config.num_attention_heads
    except:
        num_layers = len(model.model.language_model.layers)
        num_heads = model.model.language_model.config.num_attention_heads
    return num_layers, num_heads


def get_layer_attention_weights(
    model,
    hidden_states,
    layer_index,
    position_ids,
    attention_mask,
    attribution_start=None,
    attribution_end=None,
    model_type=None,
):
    model_type = model_type or infer_model_type(model)
    if model_type in ("gemma3"):
        language_model = model.model.language_model
    else:
        language_model = model.model
    num_layers = len(language_model.layers)
    assert layer_index >= 0 and layer_index < num_layers
    layer = language_model.layers[layer_index]
    # layer = model.model.layers[layer_index]
    self_attn = layer.self_attn
    hidden_states = hidden_states[layer_index]
    #print("hidden_states_shape: ", hidden_states.shape)
    hidden_states = layer.input_layernorm(hidden_states)
    bsz, q_len, _ = hidden_states.size()

    num_attention_heads = language_model.config.num_attention_heads
    num_key_value_heads = language_model.config.num_key_value_heads
    head_dim = self_attn.head_dim

    if model_type in ("llama", "qwen2", "qwen1.5", "gemma3", "glm"):
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
    elif model_type in ("phi3",):
        qkv = self_attn.qkv_proj(hidden_states)
        query_pos = num_attention_heads * head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}")

    query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
    key_states = key_states.transpose(1, 2)

    # Normalize query and key states
    if model_type in ["gemma3", "qwen3"]:
        query_states = self_attn.q_norm(query_states)
        key_states = self_attn.k_norm(key_states)

    # Apply RoPE
    if model_type in ["gemma3"]:
        if self_attn.is_sliding:
            position_embeddings = language_model.rotary_emb_local(
                hidden_states, position_ids
            )
        else:
            position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    else:
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

    cos, sin = position_embeddings

    apply_rotary_pos_emb, repeat_kv = get_helpers(model_type)
    query_states = query_states.to("cuda:0")
    key_states = key_states.to("cuda:0")
    cos = cos.to("cuda:0")
    sin = sin.to("cuda:0")

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attribution_start = attribution_start if attribution_start is not None else 1
    attribution_end = attribution_end if attribution_end is not None else q_len + 1
    causal_mask = causal_mask[:, :, attribution_start - 1 : attribution_end - 1]
    query_states = query_states[:, :, attribution_start - 1 : attribution_end - 1]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        head_dim
    )
    attn_weights = attn_weights + causal_mask
    dtype = attn_weights.dtype
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

    return attn_weights


def get_attention_weights_one_layer(
    model: Any,
    hidden_states: Any,
    layer_index: int,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Any:

    with torch.no_grad():
        position_ids, attention_mask = get_position_ids_and_attention_mask(
            model, hidden_states
        )
        num_layers, num_heads = get_attentions_shape(model)
        num_tokens = hidden_states[0].shape[1] + 1
        attribution_start = attribution_start if attribution_start is not None else 1
        attribution_end = attribution_end if attribution_end is not None else num_tokens
        num_target_tokens = attribution_end - attribution_start
        weights = torch.zeros(
            num_layers,
            num_heads,
            num_target_tokens,
            num_tokens - 1,
            device=model.device,
            dtype=model.dtype,
        )

        weights = get_layer_attention_weights(
            model,
            hidden_states,
            layer_index,
            position_ids,
            attention_mask,
            attribution_start=attribution_start,
            attribution_end=attribution_end,
            model_type=model_type,
        )

    return weights


def get_hidden_states_one_layer(
    model: Any,
    hidden_states: Any,
    layer_index: int,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Any:
    def get_hidden_states(
        model,
        hidden_states,
        layer_index,
        position_ids,
        attention_mask,
        attribution_start=None,
        attribution_end=None,
        model_type=None,
        ):
        model_type = model_type or infer_model_type(model)

        if model_type in ("gemma3"):
            language_model = model.model.language_model
        else:
            language_model = model.model
        num_layers = len(language_model.layers)
        assert layer_index >= 0 and layer_index < num_layers
        
        layer = language_model.layers[layer_index]

        self_attn = layer.self_attn
        hidden_states = hidden_states[layer_index]
        hidden_states = layer.input_layernorm(hidden_states)
        bsz, q_len, _ = hidden_states.size()

        num_attention_heads = language_model.config.num_attention_heads
        num_key_value_heads = language_model.config.num_key_value_heads
        head_dim = self_attn.head_dim

        if model_type in ("llama", "qwen2", "qwen1.5","gemma3","glm"):
            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
        elif model_type in ("phi3",):
            qkv = self_attn.qkv_proj(hidden_states)
            query_pos = num_attention_heads * head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]
        else:
            raise ValueError(f"Unknown model: {model.name_or_path}")

        query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).mean(dim=(0, 2))
        return key_states

    with torch.no_grad():
        position_ids, attention_mask = get_position_ids_and_attention_mask(
            model, hidden_states
        )
        num_layers, num_heads = get_attentions_shape(model)
        num_tokens = hidden_states[0].shape[1] + 1
        attribution_start = attribution_start if attribution_start is not None else 1
        attribution_end = attribution_end if attribution_end is not None else num_tokens
        num_target_tokens = attribution_end - attribution_start
        weights = torch.zeros(
            num_layers,
            num_heads,
            num_target_tokens,
            num_tokens - 1,
            device=model.device,
            dtype=model.dtype,
        )

        hidden_states = get_hidden_states(
            model,
            hidden_states,
            layer_index,
            position_ids,
            attention_mask,
            attribution_start=attribution_start,
            attribution_end=attribution_end,
            model_type=model_type,
        )
 

    return hidden_states