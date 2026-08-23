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
    # Learned positional encoding (only present on pos_encoding_learned
    # models; inverse maps back so the trained PE survives save->load, RIL
    # ISS-063). Non-standard key — ignored by HF transformers, honored by
    # our own from_pretrained.
    "model.positional_encoding.pos_embedding.weight": "embedding_layer.positional_encoding.pos_embedding.weight",
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
    # MLA extra parameters (our pseudo-MLA): identity through the Llama
    # naming scheme so a trained MLA model round-trips save_pretrained ->
    # from_pretrained with its latents / latent_*_proj / input_kv_proj
    # intact instead of being silently dropped at random init (RIL ISS-169).
    # External DeepSeek-style MLA checkpoints use different names and are out
    # of scope for the llama loader.
    "model.layers.{layer}.self_attn.latents": "transformer_blocks.{layer}.self_attn.latents",
    "model.layers.{layer}.self_attn.latent_q_proj.weight": "transformer_blocks.{layer}.self_attn.latent_q_proj.weight",
    "model.layers.{layer}.self_attn.latent_q_proj.bias": "transformer_blocks.{layer}.self_attn.latent_q_proj.bias",
    "model.layers.{layer}.self_attn.latent_v_proj.weight": "transformer_blocks.{layer}.self_attn.latent_v_proj.weight",
    "model.layers.{layer}.self_attn.latent_v_proj.bias": "transformer_blocks.{layer}.self_attn.latent_v_proj.bias",
    "model.layers.{layer}.self_attn.latent_output_proj.weight": "transformer_blocks.{layer}.self_attn.latent_output_proj.weight",
    "model.layers.{layer}.self_attn.latent_output_proj.bias": "transformer_blocks.{layer}.self_attn.latent_output_proj.bias",
    "model.layers.{layer}.self_attn.input_kv_proj.weight": "transformer_blocks.{layer}.self_attn.input_kv_proj.weight",
    "model.layers.{layer}.self_attn.input_kv_proj.bias": "transformer_blocks.{layer}.self_attn.input_kv_proj.bias",
    # MLP projections. Our MLP exposes:
    #   * ``fc1`` — activated path (``silu(fc1(x))``, the *gate* role)
    #   * ``gate_proj`` — the raw multiplying path (the *up* role)
    #   * ``fc2`` — the "down" projection
    # HF Llama's names are ``gate_proj`` (activated), ``up_proj`` (raw
    # multiplier), ``down_proj``.  Our forward computes
    # ``fc2(silu(fc1(x)) * gate_proj(x))`` while HF computes
    # ``down(silu(gate_proj(x)) * up_proj(x))`` — so the *roles* line up as
    # ``fc1`` <-> ``gate_proj`` and ``gate_proj`` <-> ``up_proj``.  An earlier
    # version mapped by name instead of by role and swapped the two, so
    # ``from_pretrained(real_llama)`` computed ``silu(up_proj(x)) * gate_proj(x)``
    # — a different function whenever the two tensors differ (always).
    "model.layers.{layer}.mlp.gate_proj.weight": "transformer_blocks.{layer}.mlp.fc1.weight",
    "model.layers.{layer}.mlp.gate_proj.bias": "transformer_blocks.{layer}.mlp.fc1.bias",
    "model.layers.{layer}.mlp.up_proj.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "model.layers.{layer}.mlp.up_proj.bias": "transformer_blocks.{layer}.mlp.gate_proj.bias",
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
    # Qwen GPT-style MLP: ``c_proj(act(w1(x)) * w2(x))`` — ``w1`` is the
    # activated gate, ``w2`` the raw multiplier (same role split as Llama).
    "transformer.h.{layer}.mlp.w1.weight": "transformer_blocks.{layer}.mlp.fc1.weight",
    "transformer.h.{layer}.mlp.w2.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "transformer.h.{layer}.mlp.c_proj.weight": "transformer_blocks.{layer}.mlp.fc2.weight",
    "transformer.h.{layer}.ln_1.weight": "transformer_blocks.{layer}.norm1.weight",
    "transformer.h.{layer}.ln_2.weight": "transformer_blocks.{layer}.norm2.weight",
}

# Qwen2 uses Llama-style naming
QWEN2_MAPPING = {
    **LLAMA_MAPPING,
}

# llama.cpp GGUF tensor names -> our names. llama.cpp emits these exact
# names for every dense Llama-style architecture (llama/mistral/qwen2/...):
# ``token_embd`` / ``output_norm`` / ``output`` for embedding, final norm and
# head, and per-layer ``blk.{i}.attn_(q|k|v|output)`` /
# ``blk.{i}.ffn_(gate|up|down)`` / ``blk.{i}.attn_norm`` /
# ``blk.{i}.ffn_norm``. The GLU role split mirrors the HF mapping — GGUF
# ``ffn_gate`` is the SILU-activated gate (our ``fc1``), ``ffn_up`` the raw
# multiplier (our ``gate_proj``), ``ffn_down`` the output projection (our
# ``fc2``). Q/K/V stay split here exactly like the HF loader's intermediate
# names so ``convert_hf_to_combined_qkv`` can fuse them back into our combined
# ``qkv_proj``.
GGUF_MAPPING = {
    "token_embd.weight": "embedding_layer.token_embeddings.weight",
    "output_norm.weight": "final_norm.weight",
    "output.weight": "lm_head.weight",
    "blk.{layer}.attn_norm.weight": "transformer_blocks.{layer}.norm1.weight",
    "blk.{layer}.attn_norm.bias": "transformer_blocks.{layer}.norm1.bias",
    "blk.{layer}.ffn_norm.weight": "transformer_blocks.{layer}.norm2.weight",
    "blk.{layer}.ffn_norm.bias": "transformer_blocks.{layer}.norm2.bias",
    "blk.{layer}.attn_q.weight": "transformer_blocks.{layer}.self_attn.q_proj.weight",
    "blk.{layer}.attn_q.bias": "transformer_blocks.{layer}.self_attn.q_proj.bias",
    "blk.{layer}.attn_k.weight": "transformer_blocks.{layer}.self_attn.k_proj.weight",
    "blk.{layer}.attn_k.bias": "transformer_blocks.{layer}.self_attn.k_proj.bias",
    "blk.{layer}.attn_v.weight": "transformer_blocks.{layer}.self_attn.v_proj.weight",
    "blk.{layer}.attn_v.bias": "transformer_blocks.{layer}.self_attn.v_proj.bias",
    "blk.{layer}.attn_output.weight": "transformer_blocks.{layer}.self_attn.out_proj.weight",
    "blk.{layer}.attn_output.bias": "transformer_blocks.{layer}.self_attn.out_proj.bias",
    "blk.{layer}.ffn_gate.weight": "transformer_blocks.{layer}.mlp.fc1.weight",
    "blk.{layer}.ffn_gate.bias": "transformer_blocks.{layer}.mlp.fc1.bias",
    "blk.{layer}.ffn_up.weight": "transformer_blocks.{layer}.mlp.gate_proj.weight",
    "blk.{layer}.ffn_up.bias": "transformer_blocks.{layer}.mlp.gate_proj.bias",
    "blk.{layer}.ffn_down.weight": "transformer_blocks.{layer}.mlp.fc2.weight",
    "blk.{layer}.ffn_down.bias": "transformer_blocks.{layer}.mlp.fc2.bias",
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
        Architecture name (llama, mistral, qwen, qwen2, mixtral) or
        ``"unknown"`` for an unsupported ``model_type``.
    """
    model_type = config.get("model_type", "").lower()

    if "llama" in model_type:
        return "llama"
    elif "mixtral" in model_type:
        # Mixtral is MoE (sparse experts + router). Our mapping is dense-only
        # and from_pretrained would build a DENSE model, silently dropping
        # every experts.*/gate tensor (RIL ISS-144). Keep it distinguishable
        # so the loader can REJECT it with a clear error instead of shipping
        # a model whose routers/experts are all at random init.
        return "mixtral"
    elif "mistral" in model_type:
        return "mistral"
    elif model_type in {"qwen2_moe", "qwen2moe", "qwen3", "qwen3_moe", "qwen3moe"}:
        # Qwen2MoE / Qwen3(MoE) are NOT supported. The substring rules below
        # would map ``qwen2_moe`` onto the DENSE Qwen2 rules (dropping every
        # expert/router tensor) and ``qwen3`` onto the Qwen1 GPT-style rules
        # (dropping most weights) — both then run from RANDOM init with
        # warnings only (the ISS-144 / round-71 anti-garbage-load philosophy).
        # Route them to "unknown" so the loader refuses loudly.
        return "unknown"
    elif "qwen2" in model_type:
        return "qwen2"
    elif "qwen" in model_type:
        return "qwen"
    else:
        # Unknown model_type (gpt2, gemma, baichuan, ...). Previously this
        # defaulted to the llama mapping and from_pretrained loaded with
        # strict=False — every unmapped weight stayed at random init and the
        # model generated garbage with only warning logs (round-71 compat
        # fix). Return a distinguishable token so the loader REFUSES instead.
        return "unknown"


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


def convert_gguf_weights(
    gguf_state_dict: dict[str, Any],
    num_layers: int,
) -> tuple[dict[str, Any], list[str]]:
    """Translate llama.cpp GGUF tensor names into our naming convention.

    Args:
        gguf_state_dict: GGUF tensors keyed by llama.cpp names (``token_embd``,
            ``blk.N.attn_q``, ``blk.N.ffn_gate``, ...).
        num_layers: Number of transformer layers (to expand the ``{layer}``
            placeholders).

    Returns:
        ``(converted, unmapped)`` — converted tensors keyed by our naming
        (q/k/v still *split* as ``.*_proj``, ready for
        :func:`convert_hf_to_combined_qkv`), and the list of GGUF tensor names
        that had no mapping. A non-empty ``unmapped`` list means the file is
        not a pure dense Llama-style GGUF (or carries extra tensors); the
        caller should refuse rather than silently drop them (RIL ISS-220
        philosophy).
    """
    mapping = expand_layer_mapping(GGUF_MAPPING, num_layers)
    converted: dict[str, Any] = {}
    unmapped: list[str] = []
    for gguf_name, tensor in gguf_state_dict.items():
        our_name = mapping.get(gguf_name)
        if our_name is not None:
            converted[our_name] = tensor
        else:
            unmapped.append(gguf_name)
    return converted, unmapped


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
    num_kv_heads: int | None = None,  # noqa: ARG001 - reserved for GQA-aware paths; not used by simple concat
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
        # Mistral's sliding-window attention: absent on Llama/Qwen externals
        # and defaulted to None (full-context) — wiring it prevents the silent
        # full-context attention past a 4096 window (RIL ISS-242).
        "window_size": hf_config.get("sliding_window"),
        # Sparse/streaming attention scheme (RIL TASK-244): our own publisher
        # persists ``attn_sparse`` (kind + params) so a sparse model roundtrips
        # with its scheme instead of silently rebuilding as dense on load.
        # External checkpoints carry no such key and default to None (dense).
        "attn_sparse": hf_config.get("attn_sparse"),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-5),
        "rope_theta": hf_config.get("rope_theta", 10000.0),
        # HF Llama/GPT-style configs carry the MLP activation as
        # ``hidden_act``; real Llama/Mistral use ``silu`` (SwiGLU).  The
        # loader maps this onto our ``mlp_activation`` so a published model
        # (or HF checkpoint) round-trips with the *same* MLP function rather
        # than silently defaulting to gelu.
        # Attention family so an MLA model round-trips as MLA instead of
        # silently rebuilding as MHA with every MLA tensor dropped at random
        # init (RIL ISS-169). Our own publisher persists it; external
        # Llama/Mistral checkpoints carry no such key and default to MHA.
        "attn_impl": hf_config.get("attn_impl", "mha"),
        "mlp_activation": hf_config.get("hidden_act", "silu"),
        # Whether the MLP is gated (SwiGLU). Real Llama/Mistral default to
        # True, but our own ``save_pretrained`` persists the actual
        # ``use_glu`` so a DEFAULT (non-GLU) model round-trips with its
        # fc1/fc2 MLP instead of being rebuilt as GLU with random gate
        # weights (RIL ISS-056). Absent (an external HF checkpoint) -> True.
        "use_glu": hf_config.get("use_glu", True),
        # Learned positional encoding flag so a learned-PE model round-trips
        # with its trained pos_embedding weights instead of silently falling
        # back to sinusoidal (RIL ISS-063). Default False (matching the
        # DecoderModel default).
        "pos_encoding_learned": hf_config.get("pos_encoding_learned", False),
        # Normalization implementation so an RMSNorm-trained model round-trips
        # with RMSNorm instead of silently rebuilding as LayerNorm (RIL
        # ISS-062). Our own publisher persists it; external checkpoints
        # default to layer_norm (the DecoderModel default).
        "norm_impl": hf_config.get("norm_impl", "layer_norm"),
        # Pre-LN vs post-LN block ordering. Our own publisher persists the
        # explicit flag so a post-LN model roundtrips as post-LN instead of
        # silently becoming the pre-LN default (RIL ISS-072).
        "norm_first": hf_config.get("norm_first", True),
        # RoPE: real Llama/Mistral/Qwen always use rotary position embedding
        # (their HF configs carry ``rope_theta`` and no ``use_rope`` key), so
        # an external checkpoint defaults to RoPE-on. Our own publisher
        # persists the explicit flag, keeping save->load self-consistent
        # (RIL ISS-062 — core.rope had zero callers before this wiring).
        "use_rope": hf_config.get("use_rope", True),
        # Bias flags: real Llama/Mistral checkpoints are bias-free (no
        # qkv/mlp/lm_head biases — ``attention_bias=False``). An external
        # checkpoint therefore defaults to bias-free; our own publisher
        # persists the actual flags so a biased model roundtrips with its
        # biases (RIL ISS-062).
        #
        # External checkpoints declare attention bias under HF's CANONICAL
        # ``attention_bias`` key (Qwen-style), not our repo-custom names —
        # falling back to those keys silently dropped every attention/MLP
        # bias for ``attention_bias: true`` checkpoints (RIL ISS-145). Prefer
        # our own persisted flags, then HF's canonical key for externals.
        "qkv_bias": hf_config.get("qkv_bias", hf_config.get("attention_bias", False)),
        "mlp_bias": hf_config.get("mlp_bias", hf_config.get("attention_bias", False)),
        "lm_head_bias": hf_config.get("lm_head_bias", hf_config.get("attention_bias", False)),
    }
