"""Model architecture configuration.

``ModelConfig`` describes the model itself — its shape (``hidden_size``,
``num_layers``, ``num_heads``), which attention/MLP/norm implementation to
build, and the capability constraints those choices impose. It deliberately
lives in ``models/`` rather than ``training/core/config.py``, where it used to
be defined: it is a property of the model, not of a training run, and keeping
it under ``training`` forced ``runtime.model_factory``, ``serving.loader``,
``export`` and ``multimodal`` — none of which train anything — to import the
training stack just to name the type.

``llm.training.core.config`` re-exports it for backward compatibility, so
existing imports keep working.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model configuration"""

    hidden_size: int = Field(512, gt=0)
    num_heads: int = Field(8, gt=0)
    num_kv_heads: int | None = Field(None, description="If None, defaults to num_heads")
    intermediate_size: int | None = Field(None, description="If None, defaults to 4 * hidden_size")
    num_layers: int = Field(2, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    use_glu: bool = False
    vocab_size: int = Field(50257, gt=0)
    max_seq_len: int = Field(512, gt=0)
    num_experts: int = Field(0, description="Number of experts when mlp_impl='moe'")
    top_k: int = Field(0, description="Top-k experts when mlp_impl='moe'")

    # Registry keys (attn_impl='mla' does not support KV cache / generation)
    attn_impl: str = "mha"
    mlp_impl: str = "mlp"
    norm_impl: str = "layer_norm"  # Resolved via NORM_REGISTRY in ModelFactory

    # Downstream hint for the continuous batching engine and serving API.
    # When True, the model is expected to write into a KV cache during
    # autoregressive decoding; ``check_consistency`` will reject configurations
    # where ``attn_impl`` does not support that.
    use_kv_cache: bool = False

    # Model-defining flags (defaults mirror the DecoderModel constructor so
    # a default config is byte-identical to before these fields existed).
    # These persist into the checkpoint's ``model_config`` sidecar and are
    # mapped by ``decoder_kwargs_from_config`` — without them a model built
    # with non-default flags (real Llama/Mistral: RoPE + bias-free) was
    # silently rebuilt with the defaults when served from a checkpoint,
    # leaving architectural params at random init.
    pos_encoding_learned: bool = False
    mlp_activation: str = "gelu"
    norm_first: bool = True
    qkv_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    use_rope: bool = False
    rope_theta: float = Field(
        10000.0,
        description=(
            "RoPE base frequency. Must be finite and > 0: a NaN value makes every "
            "positional encoding NaN, so the model returns an all-NaN tensor with no "
            "error at construction time (measured); inf is accepted by RoPE but is "
            "nonsense as a base frequency."
        ),
    )

    @field_validator("rope_theta")
    @classmethod
    def _validate_rope_theta(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"rope_theta must be a finite value > 0, got {value!r}")
        return value

    use_alibi: bool = False  # ALiBi linear-bias PE (BLOOM-style, mha backend only)

    # Optional sparse / streaming attention scheme (ROADMAP 15.1/15.2).
    # ``kind`` selects one of the dispatched builders (block_sparse / streaming /
    # longformer / bigbird); remaining keys are that builder's params. When set a
    # caller can build the mask via
    # ``llm.core.attn.sparse.build_config_attention_mask``.
    attn_sparse: dict[str, Any] | None = Field(None, description="Optional sparse attention scheme config")

    @model_validator(mode="after")
    def check_consistency(self) -> ModelConfig:
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        if self.intermediate_size <= 0:
            raise ValueError("Intermediate size must be positive")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        # ``num_kv_heads`` is the GQA knob: MHA/flash_attn repeat each KV head
        # ``num_heads // num_kv_heads`` times, so the value must divide
        # ``num_heads``. Without this check the backends raised on their own
        # and DIVERGED: ``num_kv_heads=0`` reached ``num_heads % 0`` and died
        # with a bare ``ZeroDivisionError``, ``num_kv_heads=-2`` built a
        # negative-width projection without complaint, and the MLA backend
        # (which has no GQA and ignores the knob entirely) accepted any value
        # — so the same config was rejected by `mha` and silently mis-built by
        # `mla`. Validate here so every backend agrees.
        if self.num_kv_heads is not None:
            if self.num_kv_heads <= 0:
                raise ValueError(f"num_kv_heads must be positive when set, got {self.num_kv_heads}")
            if self.num_kv_heads > self.num_heads:
                raise ValueError(
                    f"num_kv_heads ({self.num_kv_heads}) must not exceed num_heads ({self.num_heads}); "
                    "grouped-query attention uses FEWER KV heads than query heads (num_heads == num_kv_heads is plain MHA)."
                )
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
                )

        if self.mlp_impl == "moe":
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive when mlp_impl='moe'.")
            if self.top_k <= 0 or self.top_k > self.num_experts:
                raise ValueError("top_k must be positive and <= num_experts when mlp_impl='moe'.")

        # Validate attn_impl is a known registry entry.
        # We import lazily to avoid a circular import (config.py is imported
        # by the attention modules' own registration paths).
        from llm.core.registry import ATTENTION_KV_CACHE_CAPABILITY

        if self.attn_impl not in ATTENTION_KV_CACHE_CAPABILITY:
            available = ", ".join(sorted(ATTENTION_KV_CACHE_CAPABILITY))
            raise ValueError(
                f"Unknown attn_impl '{self.attn_impl}'. "
                f"Available: {available}. Register a new attention impl via "
                f"@register_attention and declare its KV-cache capability via "
                f"set_attention_kv_cache_capability."
            )

        # ``norm_impl`` / ``mlp_impl`` are registry keys too, but they were only
        # checked when the model was BUILT. A typo therefore survived config
        # load, CLI parsing and the whole startup path, then failed deep in
        # ``DecoderModel`` construction — while ``attn_impl`` already failed
        # here. Validate all three at the same point.
        from llm.core.registry import MLP_REGISTRY, NORM_REGISTRY, ensure_norms_registered

        # NORM_REGISTRY populates lazily (see ``_resolve_norm_factory``), so
        # trigger the registration before comparing against its names.
        ensure_norms_registered()
        if self.norm_impl not in NORM_REGISTRY.names():
            raise ValueError(f"Unknown norm_impl '{self.norm_impl}'. Available: {', '.join(NORM_REGISTRY.names())}.")
        if self.mlp_impl not in MLP_REGISTRY.names():
            raise ValueError(f"Unknown mlp_impl '{self.mlp_impl}'. Available: {', '.join(MLP_REGISTRY.names())}.")

        if self.use_kv_cache and not ATTENTION_KV_CACHE_CAPABILITY[self.attn_impl]:
            # Derive the suggestion from the registry instead of hard-coding a
            # list. The message used to end "(currently: mha)" while
            # flash_attn and mla both support KV cache too, so a user on a
            # non-supporting impl was steered to the least attractive of three
            # options — and the list silently re-staled each time an impl gained
            # support.
            kv_capable = sorted(name for name, supports in ATTENTION_KV_CACHE_CAPABILITY.items() if supports)
            raise ValueError(
                f"attn_impl='{self.attn_impl}' does not support KV cache "
                f"(capability=False). Set model.use_kv_cache=False or switch "
                f"to an attention impl that supports KV cache "
                f"(currently: {', '.join(kv_capable)})."
            )

        # RoPE is only wired into the MHA and flash_attn backends (the block
        # threads the kwargs; the kernels rotate Q/K by head position). MLA's
        # static-latent-query scheme performs no positional encoding, and its
        # ``**`` catch-all used to SILENTLY swallow ``use_rope=True`` — the
        # config validated, the model built, and the position encoding was
        # simply absent. Refuse loudly instead of running a
        # silently wrong model.
        if self.use_rope and self.attn_impl not in ("mha", "flash_attn"):
            raise ValueError(
                f"attn_impl='{self.attn_impl}' does not support use_rope=True. "
                f"RoPE is wired for 'mha' and 'flash_attn' only; switch "
                f"attn_impl or set model.use_rope=False."
            )
        # ALiBi is mutually exclusive with RoPE and (in this milestone) only
        # wired for the mha backend (ALiBi milestone).
        if self.use_alibi and self.use_rope:
            raise ValueError("use_alibi=True and use_rope=True are mutually exclusive position encodings")
        if self.use_alibi and self.attn_impl != "mha":
            raise ValueError(
                f"attn_impl='{self.attn_impl}' does not support use_alibi=True. "
                f"ALiBi is wired for 'mha' only in this milestone; switch "
                f"attn_impl or set model.use_alibi=False."
            )
        if self.attn_sparse is not None:
            if not isinstance(self.attn_sparse, dict) or "kind" not in self.attn_sparse:
                raise ValueError("attn_sparse must be a dict with a 'kind' key")
            # Lazy import to avoid a config<->attention import cycle.
            from llm.core.attn.sparse import SUPPORTED_KINDS

            if self.attn_sparse["kind"] not in SUPPORTED_KINDS:
                raise ValueError(
                    f"unknown attn_sparse kind '{self.attn_sparse['kind']}'; expected one of {SUPPORTED_KINDS}"
                )
        return self
