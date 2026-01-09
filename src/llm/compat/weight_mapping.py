"""
Weight Mapping for HuggingFace Model Conversion.

Provides mappings from HuggingFace weight names to this project's naming convention.
Supports Llama, Mistral, and Qwen architectures.
"""

from typing import Any

# Weight name mappings: HF name pattern -> our name pattern
# Uses {layer} as placeholder for layer index

LLAMA_MAPPING = {
    # Embeddings
    "model.embed_tokens.weight": "embedding_layer.token_embeddings.weight",
    # Final norm
    "model.norm.weight": "final_norm.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
    # Per-layer mappings (use {layer} placeholder)
    "model.layers.{layer}.self_attn.q_proj.weight": "transformer_blocks.{layer}.attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight": "transformer_blocks.{layer}.attn.k_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.weight": "transformer_blocks.{layer}.attn.v_proj.weight",
    "model.layers.{layer}.self_attn.o_proj.weight": "transformer_blocks.{layer}.attn.out_proj.weight",
    "model.layers.{layer}.mlp.gate_proj.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.up_proj.weight": "transformer_blocks.{layer}.mlp.up_proj.weight",
    "model.layers.{layer}.mlp.down_proj.weight": "transformer_blocks.{layer}.mlp.down_proj.weight",
    "model.layers.{layer}.input_layernorm.weight": "transformer_blocks.{layer}.norm1.weight",
    "model.layers.{layer}.post_attention_layernorm.weight": "transformer_blocks.{layer}.norm2.weight",
}

MISTRAL_MAPPING = {
    # Same as Llama for most parts
    **LLAMA_MAPPING,
    # Mistral uses same naming as Llama
}

QWEN_MAPPING = {
    # Embeddings
    "transformer.wte.weight": "embedding_layer.token_embeddings.weight",
    # Final norm
    "transformer.ln_f.weight": "final_norm.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
    # Per-layer mappings
    "transformer.h.{layer}.attn.c_attn.weight": "transformer_blocks.{layer}.attn.qkv_proj.weight",
    "transformer.h.{layer}.attn.c_proj.weight": "transformer_blocks.{layer}.attn.out_proj.weight",
    "transformer.h.{layer}.mlp.w1.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "transformer.h.{layer}.mlp.w2.weight": "transformer_blocks.{layer}.mlp.up_proj.weight",
    "transformer.h.{layer}.mlp.c_proj.weight": "transformer_blocks.{layer}.mlp.down_proj.weight",
    "transformer.h.{layer}.ln_1.weight": "transformer_blocks.{layer}.norm1.weight",
    "transformer.h.{layer}.ln_2.weight": "transformer_blocks.{layer}.norm2.weight",
}

# Qwen2 uses Llama-style naming
QWEN2_MAPPING = {
    **LLAMA_MAPPING,
}

# Architecture type to mapping
ARCHITECTURE_MAPPINGS = {
    "llama": LLAMA_MAPPING,
    "mistral": MISTRAL_MAPPING,
    "qwen": QWEN_MAPPING,
    "qwen2": QWEN2_MAPPING,
}


def detect_architecture(config: dict[str, Any]) -> str:
    """
    Detect model architecture from HuggingFace config.

    Args:
        config: HuggingFace model config dict.

    Returns:
        Architecture name (llama, mistral, qwen, qwen2).
    """
    model_type = config.get("model_type", "").lower()

    if "llama" in model_type:
        return "llama"
    elif "mistral" in model_type or "mixtral" in model_type:
        return "mistral"
    elif "qwen2" in model_type:
        return "qwen2"
    elif "qwen" in model_type:
        return "qwen"
    else:
        # Default to llama-style for unknown
        return "llama"


def get_weight_mapping(architecture: str) -> dict[str, str]:
    """
    Get weight name mapping for an architecture.

    Args:
        architecture: Architecture name.

    Returns:
        Dictionary mapping HF names to our names.
    """
    return ARCHITECTURE_MAPPINGS.get(architecture, LLAMA_MAPPING)


def expand_layer_mapping(mapping: dict[str, str], num_layers: int) -> dict[str, str]:
    """
    Expand layer-indexed mappings for all layers.

    Args:
        mapping: Base mapping with {layer} placeholders.
        num_layers: Number of transformer layers.

    Returns:
        Expanded mapping with concrete layer indices.
    """
    expanded = {}

    for hf_pattern, our_pattern in mapping.items():
        if "{layer}" in hf_pattern:
            for layer_idx in range(num_layers):
                hf_name = hf_pattern.format(layer=layer_idx)
                our_name = our_pattern.format(layer=layer_idx)
                expanded[hf_name] = our_name
        else:
            expanded[hf_pattern] = our_pattern

    return expanded


def convert_hf_weights(
    hf_state_dict: dict[str, Any],
    architecture: str,
    num_layers: int,
) -> dict[str, Any]:
    """
    Convert HuggingFace state dict to our naming convention.

    Args:
        hf_state_dict: HuggingFace model state dict.
        architecture: Model architecture.
        num_layers: Number of transformer layers.

    Returns:
        Converted state dict with our naming.
    """
    mapping = get_weight_mapping(architecture)
    expanded_mapping = expand_layer_mapping(mapping, num_layers)

    converted = {}
    unmapped = []

    for hf_name, tensor in hf_state_dict.items():
        if hf_name in expanded_mapping:
            our_name = expanded_mapping[hf_name]
            converted[our_name] = tensor
        else:
            # Try partial match for bias terms etc.
            matched = False
            for hf_pattern, our_pattern in expanded_mapping.items():
                if hf_name.replace(".bias", ".weight") == hf_pattern:
                    our_name = our_pattern.replace(".weight", ".bias")
                    converted[our_name] = tensor
                    matched = True
                    break

            if not matched:
                unmapped.append(hf_name)

    if unmapped:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Unmapped weights: {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")

    return converted


def get_config_mapping(hf_config: dict[str, Any]) -> dict[str, Any]:
    """
    Map HuggingFace config to our config format.

    Args:
        hf_config: HuggingFace config dict.

    Returns:
        Our config dict.
    """
    return {
        "vocab_size": hf_config.get("vocab_size", 32000),
        "hidden_size": hf_config.get("hidden_size", 4096),
        "num_layers": hf_config.get("num_hidden_layers", 32),
        "num_heads": hf_config.get("num_attention_heads", 32),
        "num_kv_heads": hf_config.get("num_key_value_heads"),
        "intermediate_size": hf_config.get("intermediate_size"),
        "max_seq_len": hf_config.get("max_position_embeddings", 4096),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-5),
        "rope_theta": hf_config.get("rope_theta", 10000.0),
    }
