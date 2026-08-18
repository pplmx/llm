"""HuggingFace publish helpers — reverse of :mod:`llm.compat.hf_loader`.

Provides :func:`save_pretrained` and :func:`push_to_hub` so models
trained with this project can be shared on HuggingFace Hub in a
format that the existing :func:`llm.compat.hf_loader.from_pretrained`
can roundtrip-load (i.e. produces ``config.json`` +
``model.safetensors`` in Llama-style naming).

Both helpers are **soft-dependency-friendly**:

* ``safetensors`` is required for save — install via
  ``pip install 'llm[compat]'``.
* ``huggingface_hub`` is required for push — same install command.

Each raises a clear ``ImportError`` with the install hint when the
corresponding dependency is missing, mirroring the convention used
elsewhere in the project (``flash_attn``, ``huggingface_hub`` in the
loader, etc.).

The reverse weight mapping lives in :mod:`llm.compat.weight_mapping`
(see :func:`llm.compat.weight_mapping.convert_our_weights`).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from llm.compat.weight_mapping import convert_our_weights
from llm.models.decoder import DecoderModel

logger = logging.getLogger(__name__)


# --- Soft-dependency probes -------------------------------------------------


_safetensors_spec = importlib.util.find_spec("safetensors")
SAFETENSORS_AVAILABLE: bool = _safetensors_spec is not None

_hf_hub_spec = importlib.util.find_spec("huggingface_hub")
HF_HUB_AVAILABLE: bool = _hf_hub_spec is not None


# --- Llama-shaped config builder -------------------------------------------


def _model_weight_dtype(model: DecoderModel) -> torch.dtype:
    """Best-effort weight dtype for the ``torch_dtype`` config field.

    On a dense model this is ``lm_head.weight.dtype`` (the same signal the
    old inline sniff used). On a quantized model the ``lm_head`` has been
    replaced by a quantized layer that stores *integer* weights (int8
    ``weight_quantized`` / packed buffers) with float scales — reading
    ``lm_head.weight`` then raises ``AttributeError`` (RIL ISS-198). The
    embedding layer is never quantized (it is an ``nn.Embedding``), so its
    weight dtype is the representative compute dtype of the model.
    """
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    embedding_layer = getattr(model, "embedding_layer", None)
    token_embeddings = getattr(embedding_layer, "token_embeddings", None)
    weight = getattr(token_embeddings, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.dtype
    # Last-resort fallback: any floating-point parameter (a quantized model
    # keeps float scales/bias even when raw weights are int). Fails into
    # float32 so the config stays JSON-serializable rather than raising.
    for param in model.parameters():
        if param.dtype.is_floating_point:
            return param.dtype
    return torch.float32


def _build_hf_config(model: DecoderModel, architecture: str = "llama") -> dict[str, Any]:
    """Build a HuggingFace-style ``config.json`` payload from a DecoderModel.

    Only fields that ``from_pretrained`` consults are populated; the
    remainder can be added by the user via a follow-up PR if they
    publish models that need RoPE / sliding-window / etc. metadata.
    """
    if architecture != "llama":
        # Only Llama-style is wired through the reverse mapping right
        # now (Qwen / Mistral differ only in naming, but the entry
        # point must still agree with ``from_pretrained``).
        raise ValueError(
            f"Architecture {architecture!r} is not yet supported by save_pretrained. "
            "Use architecture='llama' or open a follow-up issue."
        )

    num_kv_heads = getattr(model.transformer_blocks[0].self_attn, "num_kv_heads", None)
    # HF Llama's ``intermediate_size`` is the gate_proj / up_proj output
    # dim. Our MLP exposes either ``fc1`` (the "up" projection when
    # use_glu is False) or both ``fc1`` and ``gate_proj`` (when
    # use_glu is True — SwiGLU). Both share the same output dim.
    mlp0 = model.transformer_blocks[0].mlp
    fc1 = getattr(mlp0, "fc1", None)
    up_proj = getattr(mlp0, "up_proj", None)
    if isinstance(fc1, nn.Linear):
        intermediate_size = fc1.out_features
    elif isinstance(up_proj, nn.Linear):
        intermediate_size = up_proj.out_features
    else:
        intermediate_size = 4 * model.hidden_size

    dtype_str = "float16"
    model_dtype = _model_weight_dtype(model)
    if model_dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    elif model_dtype == torch.float32:
        dtype_str = "float32"

    # Persist the MLP activation so ``from_pretrained`` rebuilds the model
    # with the *same* MLP function. HF Llama/GPT-style configs call this
    # ``hidden_act`` (silu for real Llama). Our MLP exposes ``activation_name``
    # ("silu"/"gelu"/"relu") — map it to the HF key; without this the loader
    # would default to silu and a gelu-trained model would silently change
    # function across a save -> load roundtrip.
    hidden_act = getattr(mlp0, "activation_name", "silu")
    # Persist whether the MLP is gated (SwiGLU) so ``from_pretrained``
    # rebuilds the same architecture. The DEFAULT ``use_glu=False`` model has
    # only ``fc1``/``fc2``; hardcoding GLU on load left every ``gate_proj`` at
    # random init (RIL ISS-056).
    use_glu = bool(getattr(mlp0, "use_glu", True))
    # Learned positional encoding is a model-defining flag; without persisting
    # it (and mapping the trained pos_embedding weight) a learned-PE model
    # silently loses its PE across save->load (RIL ISS-063).
    pos_encoding_learned = bool(getattr(model.embedding_layer.positional_encoding, "learned", False))
    norm_impl = str(getattr(model, "norm_impl", "layer_norm"))
    # Pre-LN vs post-LN block ordering is model-defining: a post-LN trained
    # model silently rebuilds as the pre-LN default without this flag being
    # persisted (RIL ISS-072). Persist the actual value the same way the
    # other architecture flags (norm_impl, RoPE, biases) are handled.
    norm_first = bool(getattr(model, "norm_first", True))
    # RoPE is model-defining too (real Llama/Mistral inject position in
    # attention, not additive embeddings). Persist the actual flag + base so a
    # RoPE model roundtrips as RoPE and a non-RoPE model does NOT silently
    # become RoPE on load (the loader defaults missing ``use_rope`` to True
    # for external checkpoints — RIL ISS-062).
    use_rope = bool(getattr(model, "use_rope", False))
    rope_theta = float(getattr(model, "rope_theta", 10000.0))
    # Bias flags are model-defining (real Llama/Mistral are bias-free; our
    # grown-from-scratch default is biased). Persist the actual values so a
    # biased model roundtrips with its biases and an unbiased one does NOT
    # silently gain random biases on load (RIL ISS-062).
    qkv_bias = bool(getattr(model, "qkv_bias", True))
    mlp_bias = bool(getattr(model, "mlp_bias", True))
    lm_head_bias = bool(getattr(model, "lm_head_bias", True))
    # Attention family is model-defining (RIL ISS-169): without persisting it,
    # an MLA model roundtrips through save_pretrained/from_pretrained as a
    # plain MHA and every MLA tensor (latents / latent_*_proj / input_kv_proj)
    # is silently dropped by ``load_state_dict(strict=False)`` — garbage
    # output with only a warning. Persist the same way hidden_act / use_glu /
    # norm_first are handled.
    attn_impl = "mla" if type(model.transformer_blocks[0].self_attn).__name__ == "MultiLatentAttention" else "mha"

    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": model.embedding_layer.token_embeddings.num_embeddings,
        "hidden_size": model.hidden_size,
        "num_hidden_layers": len(model.transformer_blocks),
        "num_attention_heads": model.num_heads,
        "num_key_value_heads": num_kv_heads if num_kv_heads is not None else model.num_heads,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": model.max_seq_len,
        # Persist the model's ACTUAL norm epsilon. The default was a hardcoded
        # 1e-5, so a model trained with any other eps (e.g. 1e-6, Qwen2's
        # convention) rebuilt with a silently different normalization function
        # across save_pretrained -> from_pretrained (the loader honors
        # whatever is written — we were writing the wrong value). Both
        # LayerNorm and RMSNorm carry ``eps``.
        "rms_norm_eps": float(getattr(model.transformer_blocks[0].norm1, "eps", 1e-5)),
        "rope_theta": rope_theta,
        "torch_dtype": dtype_str,
        "hidden_act": hidden_act,
        "use_glu": use_glu,
        "attn_impl": attn_impl,
        "pos_encoding_learned": pos_encoding_learned,
        "norm_impl": norm_impl,
        "norm_first": norm_first,
        "use_rope": use_rope,
        "use_alibi": bool(getattr(model, "use_alibi", False)),
        "qkv_bias": qkv_bias,
        "mlp_bias": mlp_bias,
        "lm_head_bias": lm_head_bias,
    }


# --- save_pretrained -------------------------------------------------------


@runtime_checkable
class _SizedAttention(Protocol):
    """Structural type for attention backends exposing the HF config dims."""

    num_heads: int
    num_kv_heads: int
    head_dim: int


def save_pretrained(model: DecoderModel, save_directory: str | Path) -> Path:
    """Save a :class:`DecoderModel` in HuggingFace-compatible format.

    Writes:

    * ``config.json`` — Llama-style config; loadable by ``LlamaConfig``.
    * ``model.safetensors`` — state dict in Llama-style naming. Roundtrip-
      loadable by :func:`llm.compat.hf_loader.from_pretrained`.

    Args:
        model: A trained :class:`DecoderModel` instance.
        save_directory: Local directory to write into. Created if it
            doesn't exist.

    Returns:
        The resolved ``save_directory`` path.

    Raises:
        ImportError: If ``safetensors`` is not installed.
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "save_pretrained requires the 'safetensors' package. Install with `pip install 'llm[compat]'`."
        )

    from safetensors.torch import save_file

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # 1. Write config.json (Llama-style).
    hf_config = _build_hf_config(model)
    config_path = save_directory / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    logger.info(f"Wrote HF config to {config_path}")

    # 2. Convert + write state_dict.
    num_layers = len(model.transformer_blocks)
    attn0 = model.transformer_blocks[0].self_attn
    if not isinstance(attn0, _SizedAttention):
        raise TypeError(f"attention backend {type(attn0).__name__} must expose num_heads/num_kv_heads/head_dim")
    converted = convert_our_weights(
        model.state_dict(),
        architecture="llama",
        num_layers=num_layers,
        num_heads=attn0.num_heads,
        num_kv_heads=attn0.num_kv_heads,
        head_dim=attn0.head_dim,
    )

    # Clone + contiguous for safetensors (it rejects views).
    contiguous = {k: v.detach().contiguous().clone() for k, v in converted.items()}
    weights_path = save_directory / "model.safetensors"
    save_file(contiguous, str(weights_path))
    logger.info(f"Wrote {len(contiguous)} tensors to {weights_path}")

    return save_directory


# --- push_to_hub -----------------------------------------------------------


def push_to_hub(
    model: DecoderModel,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload model",
    exist_ok: bool = True,
    save_directory: str | Path | None = None,
) -> str:
    """Save the model locally and push to a HuggingFace Hub repo.

    Calls :func:`save_pretrained` to a staging directory (or to
    ``save_directory`` if provided) and uploads via
    ``huggingface_hub.upload_folder``. The repo is created on first
    push unless ``exist_ok=False``.

    Args:
        model: A trained :class:`DecoderModel`.
        repo_id: HF Hub repo ID (e.g. ``"alice/my-llm"``).
        token: HF auth token. Falls back to ``HF_TOKEN`` env var if
            ``None``. Use ``huggingface-cli login`` to persist.
        private: Whether to create the repo as private.
        commit_message: Git commit message on the Hub side.
        exist_ok: Don't raise if the repo already exists.
        save_directory: Optional staging dir; defaults to a temporary
            directory under the system temp path.

    Returns:
        The HF Hub URL of the pushed repo (e.g.
        ``"https://huggingface.co/alice/my-llm"``).

    Raises:
        ImportError: If ``huggingface_hub`` (or ``safetensors``) is
            not installed.
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "push_to_hub requires the 'huggingface_hub' package. Install with `pip install 'llm[compat]'`."
        )

    import tempfile

    from huggingface_hub import HfApi

    if save_directory is None:
        save_directory = Path(tempfile.mkdtemp(prefix="llm-push-"))
    save_directory = Path(save_directory)

    save_pretrained(model, save_directory)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=exist_ok)
    api.upload_folder(
        folder_path=str(save_directory),
        repo_id=repo_id,
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Pushed model to {url}")
    return url
