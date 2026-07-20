"""
Weight Mapping for HuggingFace Model Conversion.

Provides mappings from HuggingFace weight names to this project's naming convention.
Supports Llama, Mistral, and Qwen architectures.
"""

from typing import Any

import torch

# Weight name mappings: HF name pattern -> our name pattern
# Uses {layer} as placeholder for layer index.
#
# IMPORTANT: the ``self_attn.q_proj`` / ``self_attn.k_proj`` /
# ``self_attn.v_proj`` keys are the **HF-side target names** (output
# of ``convert_our_weights`` and input to ``convert_hf_weights`` for
# the *intermediate* rename before split/concat). Our MHA stores Q,
# K, V in a single ``qkv_proj`` Linear; the conversion code splits
# the combined projection when going ``ours -> HF`` and concatenates
# when going ``HF -> ours``. The HF targets here exist for the
# forward (HF -> ours) mapping so users who already have split
# q/k/v state dicts can be loaded without the concat helper.

LLAMA_MAPPING = {
    # Embeddings
    "model.embed_tokens.weight": "embedding_layer.token_embeddings.weight",
    # Final norm
    "model.norm.weight": "final_norm.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
    # Per-layer mappings (use {layer} placeholder).
    # Note: the model uses ``self_attn`` (not ``attn``) and ``out_proj``.
    # q/k/v are mapped to separate intermediate names so the
    # ``convert_hf_to_combined_qkv`` helper can concat them into
    # ``qkv_proj`` (the MHA impl uses a single combined projection).
    "model.layers.{layer}.self_attn.q_proj.weight": "transformer_blocks.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.self_attn.q_proj.bias": "transformer_blocks.{layer}.self_attn.q_proj.bias",
    "model.layers.{layer}.self_attn.k_proj.weight": "transformer_blocks.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.bias": "transformer_blocks.{layer}.self_attn.k_proj.bias",
    "model.layers.{layer}.self_attn.v_proj.weight": "transformer_blocks.{layer}.self_attn.v_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.bias": "transformer_blocks.{layer}.self_attn.v_proj.bias",
    "model.layers.{layer}.self_attn.o_proj.weight": "transformer_blocks.{layer}.self_attn.out_proj.weight",
    "model.layers.{layer}.self_attn.o_proj.bias": "transformer_blocks.{layer}.self_attn.out_proj.bias",
    # MLP projections. Our MLP exposes:
    #   * ``fc1`` — the "up" projection (always present)
    #   * ``gate_proj`` — the gating path (only when ``use_glu=True``)
    #   * ``fc2`` — the "down" projection
    # HF Llama's names are ``gate_proj``, ``up_proj``, ``down_proj``.
    # The mapping below translates ``up_proj`` <-> ``fc1`` and
    # ``down_proj`` <-> ``fc2`` so a published model is loadable by
    # HF's transformers library AND roundtrips through our
    # ``from_pretrained``.
    "model.layers.{layer}.mlp.gate_proj.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.gate_proj.bias": "transformer_blocks.{layer}.mlp.gate_proj.bias",
    "model.layers.{layer}.mlp.up_proj.weight": "transformer_blocks.{layer}.mlp.fc1.weight",
    "model.layers.{layer}.mlp.up_proj.bias": "transformer_blocks.{layer}.mlp.fc1.bias",
    "model.layers.{layer}.mlp.down_proj.weight": "transformer_blocks.{layer}.mlp.fc2.weight",
    "model.layers.{layer}.mlp.down_proj.bias": "transformer_blocks.{layer}.mlp.fc2.bias",
    # Layer norms
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
    # Per-layer mappings (model attribute is ``self_attn``).
    "transformer.h.{layer}.attn.c_attn.weight": "transformer_blocks.{layer}.self_attn.qkv_proj.weight",
    "transformer.h.{layer}.attn.c_proj.weight": "transformer_blocks.{layer}.self_attn.out_proj.weight",
    "transformer.h.{layer}.mlp.w1.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "transformer.h.{layer}.mlp.w2.weight": "transformer_blocks.{layer}.mlp.fc1.weight",
    "transformer.h.{layer}.mlp.c_proj.weight": "transformer_blocks.{layer}.mlp.fc2.weight",
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


def convert_our_weights(
    our_state_dict: dict[str, Any],
    architecture: str,
    num_layers: int,
    *,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> dict[str, Any]:
    """
    Convert our naming convention to HuggingFace state dict.

    Inverse of :func:`convert_hf_weights` for the supported weight
    names. Used by ``save_pretrained`` to publish models to
    HuggingFace in a format the existing ``from_pretrained`` can
    roundtrip-load.

    Splits our **combined** ``qkv_proj`` projection into HF's separate
    ``q_proj`` / ``k_proj`` / ``v_proj`` weights so the published
    artifact is loadable by both our ``from_pretrained`` (which uses
    the reverse concat) and HF's transformers library.

    Args:
        our_state_dict: Our model state dict (e.g. ``model.state_dict()``).
        architecture: Target HF architecture (must match the model).
        num_layers: Number of transformer layers in the model.
        num_heads: Total attention heads. Required when the model has
            a combined ``qkv_proj`` so we can split Q vs. K/V.
        num_kv_heads: Number of KV heads (for GQA/MQA). Defaults to
            ``num_heads`` (standard MHA).
        head_dim: Per-head dimension. Defaults to ``hidden_size // num_heads``.

    Returns:
        Converted state dict with HuggingFace naming.
    """
    mapping = get_weight_mapping(architecture)
    expanded_mapping = expand_layer_mapping(mapping, num_layers)

    # Build the reverse map: our_name -> hf_name. The forward map is
    # hf_name -> our_name, so we invert it.
    reverse_mapping = {our_name: hf_name for hf_name, our_name in expanded_mapping.items()}

    converted = {}
    unmapped = []

    # First pass: split combined qkv_proj into q_proj/k_proj/v_proj if
    # the model uses a combined projection (the current MHA impl does).
    qkv_keys = [
        k for k in our_state_dict if k.endswith(".self_attn.qkv_proj.weight") or k.endswith(".self_attn.qkv_proj.bias")
    ]
    if qkv_keys and num_heads is not None:
        n_q = num_heads
        n_kv = num_kv_heads if num_kv_heads is not None else num_heads
        # head_dim must be supplied; defaulting here would silently
        # mis-split the projection.
        if head_dim is None:
            raise ValueError(
                "head_dim is required when splitting combined qkv_proj. Pass it explicitly from the model's MHA block."
            )
        q_size = n_q * head_dim
        kv_size = n_kv * head_dim

        for qkv_key in qkv_keys:
            tensor = our_state_dict[qkv_key]
            # Strip the trailing ".qkv_proj" suffix so we get just the
            # layer prefix, e.g. "transformer_blocks.0.self_attn".
            our_prefix = qkv_key.rsplit(".", 1)[0].rsplit(".", 1)[0]
            # Translate to HF naming: replace ``transformer_blocks.``
            # with ``model.layers.`` — the rest stays the same.
            hf_prefix = our_prefix.replace("transformer_blocks.", "model.layers.") + "."

            if "weight" in qkv_key:
                # Linear weight: shape (out_features, in_features)
                q_w, k_w, v_w = tensor.split([q_size, kv_size, kv_size], dim=0)
                converted[hf_prefix + "q_proj.weight"] = q_w.contiguous()
                converted[hf_prefix + "k_proj.weight"] = k_w.contiguous()
                converted[hf_prefix + "v_proj.weight"] = v_w.contiguous()
            else:
                q_b, k_b, v_b = tensor.split([q_size, kv_size, kv_size], dim=0)
                converted[hf_prefix + "q_proj.bias"] = q_b.contiguous()
                converted[hf_prefix + "k_proj.bias"] = k_b.contiguous()
                converted[hf_prefix + "v_proj.bias"] = v_b.contiguous()

    # Second pass: rename everything else via the reverse mapping.
    for our_name, tensor in our_state_dict.items():
        if our_name in qkv_keys:
            continue  # already handled above
        if our_name in reverse_mapping:
            converted[reverse_mapping[our_name]] = tensor
        else:
            # Try partial match for bias terms (mirrors the HF -> ours path).
            matched = False
            for our_pattern, hf_pattern in reverse_mapping.items():
                if our_name.replace(".bias", ".weight") == our_pattern:
                    converted[hf_pattern.replace(".weight", ".bias")] = tensor
                    matched = True
                    break

            if not matched:
                unmapped.append(our_name)

    if unmapped:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Unmapped weights (ours -> HF): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")

    return converted


def convert_hf_to_combined_qkv(
    our_state_dict: dict[str, Any],
    num_layers: int,
    *,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> dict[str, Any]:
    """
    Concatenate our separate ``q_proj`` / ``k_proj`` / ``v_proj``
    projections into the combined ``qkv_proj``.

    Used by :func:`llm.compat.hf_loader.from_pretrained` after
    :func:`convert_hf_weights` has renamed HF Llama's separate
    q/k/v projections to our naming. Our MHA stores Q/K/V in a
    single ``qkv_proj`` Linear — this helper fuses the three
    projections back together so ``load_state_dict`` finds the
    expected key.

    Args:
        our_state_dict: Our-renamed state dict (output of
            ``convert_hf_weights``). Keys must use our naming
            (``transformer_blocks.{layer}.self_attn.q_proj.weight``
            etc.).
        num_layers: Number of transformer layers.
        num_heads: Total attention heads.
        num_kv_heads: Number of KV heads (for GQA/MQA). Defaults to
            ``num_heads`` (standard MHA).
        head_dim: Per-head dimension.

    Returns:
        State dict with combined ``qkv_proj`` projections.
    """
    if num_heads is None or head_dim is None:
        raise ValueError(
            "num_heads and head_dim are required to concatenate q/k/v projections. "
            "Pass them from the loaded model's MHA block."
        )
    n_kv = num_kv_heads if num_kv_heads is not None else num_heads
    q_size = num_heads * head_dim
    kv_size = n_kv * head_dim

    out = dict(our_state_dict)
    for layer_idx in range(num_layers):
        prefix = f"transformer_blocks.{layer_idx}.self_attn."
        q_w = out.pop(f"{prefix}q_proj.weight", None)
        k_w = out.pop(f"{prefix}k_proj.weight", None)
        v_w = out.pop(f"{prefix}v_proj.weight", None)
        q_b = out.pop(f"{prefix}q_proj.bias", None)
        k_b = out.pop(f"{prefix}k_proj.bias", None)
        v_b = out.pop(f"{prefix}v_proj.bias", None)

        if q_w is not None and k_w is not None and v_w is not None:
            combined_w = torch.cat([q_w, k_w, v_w], dim=0)
            out[f"{prefix}qkv_proj.weight"] = combined_w.contiguous()
        if q_b is not None and k_b is not None and v_b is not None:
            combined_b = torch.cat([q_b, k_b, v_b], dim=0)
            out[f"{prefix}qkv_proj.bias"] = combined_b.contiguous()

    return out


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
