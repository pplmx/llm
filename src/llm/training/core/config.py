from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # leaving architectural params at random init (RIL ISS-129).
    pos_encoding_learned: bool = False
    mlp_activation: str = "gelu"
    norm_first: bool = True
    qkv_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    use_rope: bool = False
    rope_theta: float = 10000.0
    use_alibi: bool = False  # ALiBi linear-bias PE (BLOOM-style, mha backend only)

    # Optional sparse / streaming attention scheme (ROADMAP 15.1/15.2, TASK-243).
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

        if self.use_kv_cache and not ATTENTION_KV_CACHE_CAPABILITY[self.attn_impl]:
            raise ValueError(
                f"attn_impl='{self.attn_impl}' does not support KV cache "
                f"(capability=False). Set model.use_kv_cache=False or switch "
                f"to an attention impl that supports KV cache (currently: mha)."
            )

        # RoPE is only wired into the MHA and flash_attn backends (the block
        # threads the kwargs; the kernels rotate Q/K by head position). MLA's
        # static-latent-query scheme performs no positional encoding, and its
        # ``**`` catch-all used to SILENTLY swallow ``use_rope=True`` — the
        # config validated, the model built, and the position encoding was
        # simply absent (RIL ISS-140). Refuse loudly instead of running a
        # silently wrong model.
        if self.use_rope and self.attn_impl not in ("mha", "flash_attn"):
            raise ValueError(
                f"attn_impl='{self.attn_impl}' does not support use_rope=True. "
                f"RoPE is wired for 'mha' and 'flash_attn' only; switch "
                f"attn_impl or set model.use_rope=False."
            )
        # ALiBi is mutually exclusive with RoPE and (in this milestone) only
        # wired for the mha backend (RIL — ALiBi milestone).
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


class TrainingConfig(BaseModel):
    """Training configuration"""

    batch_size: int = Field(128, gt=0)
    epochs: int = Field(10, gt=0)
    lr: float = Field(
        1e-3,
        gt=0,
        validation_alias=AliasChoices("lr", "learning_rate"),
        description=(
            "Learning rate. The YAML key is ``training.lr``; "
            "``training.learning_rate`` is accepted as an alias for "
            "backwards compatibility with older preset configs."
        ),
    )
    weight_decay: float = 0.01
    num_samples: int = 20000
    scheduler_type: str = Field("cosine", pattern="^(cosine|step|plateau)$")
    warmup_epochs: int = 1
    gradient_clip_val: float = 1.0
    run_validation: bool = True
    max_steps: int = Field(
        0,
        ge=0,
        description=(
            "Hard cap on total optimizer steps. 0 = no cap (driven by "
            "epochs * steps_per_epoch). Useful for smoke configs that "
            "shouldn't run forever even if num_samples is large."
        ),
    )

    # DPO beta (KL constraint strength) — opt-in via the DPO task.
    # ``DPOTask.build_model`` reads this via ``getattr`` with a default
    # of 0.1, so it's safe for non-DPO training runs to leave it unset.
    # The default matches the DPO paper (Rafailov et al. 2023).
    dpo_beta: float = Field(
        0.1,
        gt=0,
        description=(
            "DPO temperature / KL constraint strength. Higher beta → "
            "stronger KL penalty against the reference model, more "
            "conservative updates. Lower beta → faster convergence but "
            "risk of reward hacking. Standard literature value: 0.1."
        ),
    )

    # SimPO (Meng et al. 2024) knobs — opt-in via the ``simpo`` task. The
    # reference-free implicit reward is ``beta * mean_logp``; ``gamma`` is the
    # target preferred-vs-rejected reward margin; ``simpo_lambda`` weights the
    # chosen-response SFT regularizer. Safe to leave unset for non-SimPO runs.
    simpo_beta: float = Field(
        2.0,
        gt=0,
        description=(
            "SimPO reward scale on the length-normalized mean log-prob "
            "(larger = stronger gradient on the preference margin). Standard "
            "literature value: 2.0."
        ),
    )
    simpo_gamma: float = Field(
        0.0,
        description=(
            "SimPO target reward margin gamma between the preferred and "
            "rejected responses. Tune upward to enforce a minimum separation."
        ),
    )
    simpo_lambda: float = Field(
        1.0,
        ge=0,
        description=(
            "SimPO weight on the chosen-response SFT regularizer "
            "(-lambda * beta * mean_logp chosen). 0 disables the SFT term."
        ),
    )

    # AdaLoRA (T3 #40-#42). Defaults preserve current behavior — the
    # callback is only registered when ``use_adalora=True``. Mirrors
    # the layer-side AdaLoRALinear defaults (init_rank=12, alpha=32,
    # target_rank=6) so an opt-in config requires no other knobs.
    use_adalora: bool = Field(
        False,
        description=(
            "Master switch for AdaLoRA adaptive-budget pruning. When "
            "True, LanguageModelingTask applies AdaLoRA to the model "
            "and registers AdaLoRAPruningCallback on the engine."
        ),
    )
    adalora_init_rank: int = Field(12, gt=0)
    adalora_target_rank: int = Field(6, gt=0)
    adalora_alpha: float = Field(32.0, gt=0)
    adalora_orth_reg_weight: float = Field(0.5, ge=0)
    adalora_ema_alpha: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="EMA smoothing factor for the gradient tracker.",
    )
    adalora_tinit: int = Field(
        0,
        ge=0,
        description="First optimizer step eligible for pruning.",
    )
    adalora_tfinal: int | None = Field(
        None,
        ge=0,
        description=(
            "Optimizer step at which the rank budget reaches adalora_target_rank. None → epochs * steps_per_epoch // 2."
        ),
    )
    adalora_prune_every: int = Field(
        50,
        ge=1,
        description="Optimizer-step cadence for the prune call.",
    )
    adalora_target_modules: list[str] | None = Field(
        None,
        description=(
            "Optional list of module-name substring patterns forwarded "
            "to apply_adalora. None → every nn.Linear is wrapped."
        ),
    )

    @model_validator(mode="after")
    def _validate_adalora(self) -> TrainingConfig:
        # Cross-field checks that the Field constraints alone can't express.
        if self.adalora_target_rank > self.adalora_init_rank:
            raise ValueError(
                f"adalora_target_rank ({self.adalora_target_rank}) must be "
                f"≤ adalora_init_rank ({self.adalora_init_rank})"
            )
        if self.adalora_tfinal is not None and self.adalora_tfinal <= self.adalora_tinit:
            raise ValueError(
                f"adalora_tfinal ({self.adalora_tfinal}) must be strictly "
                f"greater than adalora_tinit ({self.adalora_tinit})"
            )
        return self

    # Prefix Tuning (T2 PEFT). Defaults preserve current behavior —
    # the wrapper is only applied when ``use_prefix_tuning=True``.
    # Mirrors the layer-side ``PrefixTuningAttention`` defaults so an
    # opt-in config requires no other knobs. Unlike AdaLoRA, Prefix
    # Tuning has no scheduler / tracker — ``apply_prefix_tuning`` is
    # a one-shot wrap at ``build_model`` time and the user calls
    # ``fold_reparameterization`` at inference time (matching the
    # LoRA apply / merge pattern).
    use_prefix_tuning: bool = Field(
        False,
        description=(
            "Master switch for Prefix Tuning. When True, "
            "LanguageModelingTask wraps every MultiHeadAttention with "
            "PrefixTuningAttention and freezes the base MHA so only "
            "the prefix path is trainable."
        ),
    )
    prefix_tuning_len: int = Field(
        10,
        gt=0,
        description=(
            "Number of prefix tokens prepended to each layer's K and V. "
            "Li & Liang 2021 used 10; larger values increase trainable "
            "parameters linearly."
        ),
    )
    prefix_reparam_hidden: int | None = Field(
        None,
        gt=0,
        description=(
            "Hidden dim of the reparameterization MLPs. None → defaults "
            "to ``kv_dim`` at the wrapper layer (full-rank projection). "
            "Smaller values reduce trainable parameters."
        ),
    )
    prefix_target_modules: list[str] | None = Field(
        None,
        description=(
            "Optional list of module-name substring patterns forwarded "
            "to apply_prefix_tuning. None → every MultiHeadAttention "
            "is wrapped."
        ),
    )

    # IA³ (T2 PEFT). Defaults preserve current behavior — the wrapper
    # is only applied when ``use_ia3=True``. Mirrors the layer-side
    # ``IA3Linear`` defaults so an opt-in config requires no other
    # knobs. Like Prefix Tuning, IA³ has no scheduler / tracker —
    # ``apply_ia3`` is a one-shot wrap at ``build_model`` time and the
    # user calls ``merge_ia3`` at inference time (matching the LoRA
    # apply / merge pattern).
    use_ia3: bool = Field(
        False,
        description=(
            "Master switch for IA³ (T-Few). When True, "
            "LanguageModelingTask wraps every nn.Linear with IA3Linear "
            "and freezes the base weight so only the ia3_l scale is "
            "trainable. Per-layer cost is out_features trainable params."
        ),
    )
    ia3_init_scale: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Initial value of the IA³ multiplicative scale. Defaults "
            "to 1.0 so the wrapper is the identity transform at step 1 — "
            "no chicken-and-egg stall. Setting to a different value (e.g. "
            "0.5) starts the model at a uniformly-downweighted version "
            "of the base — useful only if you have a reason to start "
            "off-distribution."
        ),
    )
    ia3_target_modules: list[str] | None = Field(
        None,
        description=(
            "Optional list of module-name substring patterns forwarded to apply_ia3. None → every nn.Linear is wrapped."
        ),
    )

    # BitFit (T2 PEFT). Defaults preserve current behavior — biases
    # are only enabled when ``use_bitfit=True``. Mirrors the
    # ``apply_bitfit`` defaults so an opt-in config requires no
    # other knobs. Like Prefix Tuning and IA³, BitFit has no
    # scheduler / tracker — ``apply_bitfit`` is a one-shot
    # ``requires_grad`` toggle at ``build_model`` time.
    use_bitfit: bool = Field(
        False,
        description=(
            "Master switch for BitFit (bias-only fine-tuning). When "
            "True, LanguageModelingTask calls apply_bitfit(model) to "
            "freeze every parameter and enable gradients on every "
            "bias. Per-model cost is exactly the sum of bias sizes."
        ),
    )
    bitfit_target_modules: list[str] | None = Field(
        None,
        description=(
            "Optional list of module-name substring patterns forwarded "
            "to apply_bitfit. A bias is trainable only if its qualified "
            "name contains at least one of the patterns. None → every "
            "bias is trainable."
        ),
    )

    # Adapter Layers (T2 PEFT). Defaults preserve current behavior —
    # the wrapper is only applied when ``use_adapter=True``.
    # Mirrors the layer-side ``AdapterLinear`` defaults so an opt-in
    # config requires no other knobs. Like Prefix Tuning / IA³ /
    # BitFit, Adapter has no scheduler / tracker — ``apply_adapter``
    # is a one-shot wrap at ``build_model`` time. There is no
    # inference-time merge for adapters — the up projection is zero
    # so the wrapper contributes zero unless trained.
    use_adapter: bool = Field(
        False,
        description=(
            "Master switch for Adapter Layers (Houlsby 2019). When "
            "True, LanguageModelingTask wraps every nn.Linear with "
            "AdapterLinear (down → activation → up bottleneck residual) "
            "and freezes the base weight so only the adapter is "
            "trainable."
        ),
    )
    adapter_bottleneck_dim: int = Field(
        64,
        gt=0,
        description=(
            "Width of the adapter's bottleneck dim. Defaults to 64 "
            "(the Houlsby 2019 paper convention). Larger values "
            "increase adapter capacity linearly; smaller values "
            "reduce trainable parameters."
        ),
    )
    adapter_target_modules: list[str] | None = Field(
        None,
        description=(
            "Optional list of module-name substring patterns forwarded "
            "to apply_adapter. None → every nn.Linear is wrapped."
        ),
    )

    # Unified PEFT dispatch (T2 PEFT #44). When ``peft_method`` is set,
    # ``LanguageModelingTask.build_model`` resolves the method through
    # ``llm.core.peft.apply_peft`` (and forwards ``peft_kwargs``) instead
    # of the per-method ``use_*`` flags below. The legacy flags remain
    # the source of truth when ``peft_method`` is ``None`` — existing
    # configs are unaffected.
    #
    # Validated against :data:`llm.core.peft.PEFT_REGISTRY` at config-load
    # time so unknown methods raise before ``build_model`` is ever called
    # (matches the ``attn_impl`` validation pattern in
    # ``ModelConfig.check_consistency``).
    peft_method: str | None = Field(
        None,
        description=(
            "Optional PEFT method name resolved through PEFT_REGISTRY "
            "(e.g. 'lora', 'ia3', 'adalora', 'prefix_tuning', 'bitfit', "
            "'adapter', 'qlora'). When set, LanguageModelingTask "
            "dispatches via apply_peft(...) instead of the per-method "
            "use_* flags. None → legacy per-method flag path."
        ),
    )
    peft_kwargs: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional kwargs forwarded verbatim to apply_peft (and "
            "thence to the per-method apply_* function). Method-"
            "specific — see each PEFT method's docstring. Only "
            "consulted when peft_method is set."
        ),
    )
    peft_save_path: str | None = Field(
        None,
        description=(
            "Optional path for the adapter-only sidecar file written "
            "by PEFTAdapterCheckpointCallback at on_train_end "
            "(T2 PEFT #48). When None, the callback derives a default "
            "of {checkpoint_dir}/peft_adapter_{method}.bin. The sidecar "
            "is the same format as save_peft(...) and round-trips "
            "via load_peft(...) into a fresh model. Independent of "
            "the main CheckpointManager flow — set this when you want "
            "to share just the adapter (cross-base-model transfer, "
            "adapter-only inference)."
        ),
    )

    # Quantization-Aware Training (QAT, ROADMAP 13.2 / RIL DEC-054). Optional.
    # When enabled, ``LanguageModelingTask.build_model`` wraps matching
    # ``nn.Linear`` layers in ``FakeQuantLinear`` so the forward fake-quantizes
    # (dynamic scale, straight-through estimator) while the full-precision
    # weights stay trainable — improving post-quant accuracy vs PTQ.
    use_qat: bool = Field(False, description="Enable QAT: fake-quantize linear weights during training.")
    qat_bits: int = Field(
        8,
        ge=0,
        description="QAT uniform symmetric bit width (4 or 8) for the fake quantizer.",
    )
    qat_quant_activation: bool = Field(
        False,
        description="QAT: also fake-quantize activations (per-tensor) in the quantized linears.",
    )
    qat_target_modules: list[str] | None = Field(
        None,
        description=(
            "QAT: name suffixes of the nn.Linear layers to fake-quantize "
            "(e.g. ['fc1', 'fc2', 'qkv_proj']); None/empty -> every nn.Linear."
        ),
    )

    @field_validator("qat_bits")
    @classmethod
    def _validate_qat_bits(cls, value: int) -> int:
        if value not in (4, 8):
            raise ValueError(f"qat_bits must be 4 or 8, got {value}")
        return value

    # Knowledge Distillation (ROADMAP 13.4 / RIL DEC-055). Consumed by the
    # ``distill`` task; controls the Hinton-style KD loss.
    distill_temperature: float = Field(
        4.0,
        gt=0,
        description="KD softening temperature T (softmax(student/T) vs softmax(teacher/T)).",
    )
    distill_alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description="KD hard-label CE weight in [0,1]; the KL term contributes (1-alpha).",
    )
    distill_teacher_path: str | None = Field(
        None,
        description=(
            "Path to the frozen-teacher checkpoint the ``distill`` task loads "
            "(a model saved by CheckpointManager with the same architecture as "
            "``config.model``). When unset, the distill task builds a "
            "freshly-seeded teacher (dev/test convenience)."
        ),
    )

    # GRPO (Group Relative Policy Optimization, ROADMAP 阶段十一 / RIL TASK-229).
    grpo_clip_eps: float = Field(
        0.2,
        ge=0,
        lt=1,
        description="GRPO policy-ratio clipping range (1-eps, 1+eps).",
    )
    grpo_kl_beta: float = Field(
        0.0,
        ge=0,
        description="GRPO K3 KL-to-reference penalty weight (0 disables).",
    )
    grpo_group_size: int = Field(
        4,
        gt=1,
        description="Number of GRPO responses sampled per group/prompt.",
    )

    @field_validator("peft_method", mode="after")
    @classmethod
    def _validate_peft_method(cls, value: str | None) -> str | None:
        """Reject unknown PEFT method names at config-load time.

        Lazy import: ``llm.core.peft`` is not imported at module top so
        this validator does not pull every PEFT module (lora / qlora /
        adalora / prefix_tuning / ia3 / bitfit / adapter) into every
        process that just constructs a ``TrainingConfig`` — only
        configurations that actually set ``peft_method`` pay the import
        cost.
        """
        if value is None:
            return value
        from llm.core.peft import PEFT_REGISTRY, ensure_methods_registered

        # Idempotent — first call populates the registry.
        ensure_methods_registered()
        if value not in PEFT_REGISTRY:
            available = ", ".join(PEFT_REGISTRY.names())
            raise ValueError(
                f"peft_method '{value}' not found in PEFTMethod registry. "
                f"Available: {available}. Register a new method via the "
                f"'llm.peft_methods' setuptools entry-point group."
            )
        return value


class DistributedConfig(BaseSettings):
    """Distributed configuration (aware of environment variables)"""

    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int | None = None  # Lazy init
    node_rank: int = 0
    backend: str = "nccl"
    parallel_strategy: str = Field(
        "ddp",
        pattern="^(ddp|fsdp|tp|pp|3d|zero)$",
        description=(
            "Parallel strategy: 'ddp' (default), 'fsdp', 'tp' (tensor parallelism, "
            "with optional tp_size < world_size for TP+data-parallel 2D), or 'pp' "
            "(pipeline parallelism, RIL DEC-049/TASK-210, with optional "
            "pp_size < world_size for PP+data-parallel 2D, RIL TASK-211). PP lays "
            "the world out as pipeline stages (one stage per rank) and only "
            "supports the standard-language-modeling loop: it refuses "
            "non-standard-loop tasks, AMP, torch.compile and TP+FSDP composition "
            "with a clear error. '3d' (RIL DEC-052/TASK-216) composes pipeline + "
            "tensor parallelism (dp_size data-parallel size, dp=1 for pure PP+TP); "
            "it needs explicit dp_size*pp_size*tp_size == world_size and the "
            "standard-loop LM contract. 'zero' (RIL TASK-269/DEC-097) runs native "
            "ZeRO Stage-1: each rank holds full parameters/gradients (averaged "
            "over the world group each step) but partitions optimizer state "
            "~1/world_size, then all-gathers the updated weights so all ranks "
            "stay in lockstep. Requires the standard loop; fp16 AMP and "
            "non-standard-loop tasks are refused in v1."
        ),
    )
    tp_size: int = Field(
        0,
        ge=0,
        description=(
            "Tensor-parallel size for parallel_strategy='tp'. A value of 0 "
            "means 'use world_size' (pure TP, one TP group = the whole world). "
            "A value less than world_size enables TP + data-parallel 2D "
            "(TASK-202): ranks are laid out row-major as [DP][TP] - each TP "
            "group is a contiguous world_size/tp_size-rank range that "
            "partitions the model in parallel over its own data-parallel "
            "shard, and the world_size/tp_size DP groups average gradients "
            "across data shards at each step. world_size must divide evenly "
            "by tp_size. Every partitioned axis (num_heads / num_kv_heads / "
            "vocabulary / intermediate width) must divide evenly by it."
        ),
    )
    pp_size: int = Field(
        0,
        ge=0,
        description=(
            "Pipeline size for parallel_strategy='pp'. A value of 0 "
            "means 'use world_size' (pure PP, one pipeline group = the whole "
            "world). A value less than world_size enables PP + data-parallel "
            "2D (TASK-211): ranks are laid out row-major as [DP][PP] - each "
            "pipeline group is a contiguous world_size/pp_size-rank range "
            "whose stage-to-stage P2P stays intranode-friendly, and the "
            "world_size/pp_size DP groups (strided columns holding the same "
            "stage) average gradients across data shards at each step. "
            "world_size must divide evenly by pp_size."
        ),
    )
    dp_size: int = Field(
        1,
        ge=0,
        description=(
            "Data-parallel size for parallel_strategy='3d' (RIL DEC-052/TASK-216). "
            "1 (default) means pure pipeline+tensor parallel (no data-parallel "
            "dimension); > 1 enables the full DP+PP+TP grid (TASK-217). "
            "world_size must equal dp_size * pp_size * tp_size."
        ),
    )
    pp_n_microbatches: int = Field(
        1,
        ge=1,
        description=(
            "Number of microbatches (chunks) the pipeline schedule splits "
            "each training batch into (parallel_strategy='pp', TASK-213). "
            "n_microbatches > 1 lets the ScheduleGPipe overlap stages "
            "(reducing bubbles) and shrinks activation memory per stage; the "
            "schedule normalizes the gradient by the microbatch count, so the "
            "per-optimizer-step gradient is identical to n_microbatches=1. "
            "Only the reported loss changes shape: the engine reports the "
            "MEAN of the per-microbatch losses."
        ),
    )
    collective_timeout_seconds: int = Field(
        1800,
        gt=0,
        description=(
            "Bounded timeout for the distributed process group. A rank that "
            "crashes mid-epoch otherwise leaves every other rank blocking "
            "forever in a collective (all_reduce / barrier / FSDP ops) with "
            "no error — the job hangs instead of failing loudly. Lower it "
            "for fast fail-fast in tests; raises "
            "torch.distributed.TimeoutError on lapse (RIL TASK-195/ISS-232)."
        ),
    )

    # FSDP-specific knobs. Only consulted when ``parallel_strategy="fsdp"``.
    # The defaults are conservative: BF16 mixed precision matches what
    # ``DistributedConfig`` already picks for AMP, the auto-wrap threshold
    # is large enough to keep tiny submodules unwrapped (matching common
    # practice for transformer block FSDP), and CPU offload is opt-in
    # because it slows training significantly.
    fsdp_mixed_precision: str = Field(
        "bf16",
        pattern="^(fp32|bf16|fp16)$",
        description=(
            "FSDP parameter / gradient / buffer dtype. 'bf16' is the "
            "recommended default for modern GPUs; 'fp16' requires a "
            "loss scaler and is rarely worth the complexity."
        ),
    )
    fsdp_auto_wrap_min_params: int = Field(
        10_000_000,
        ge=0,
        description=(
            "FSDP size-based auto-wrap threshold. Modules with at "
            "least this many parameters are wrapped as their own FSDP "
            "unit; smaller submodules stay inside the parent unit. "
            "Set to 0 to wrap every leaf module (rarely useful)."
        ),
    )
    fsdp_cpu_offload: bool = Field(
        False,
        description=(
            "Offload FSDP parameters to CPU when not in use. Trades "
            "training throughput for GPU memory; useful only when "
            "the model is too big to fit even after BF16 sharding."
        ),
    )

    @field_validator("gpus_per_node", mode="before")
    @classmethod
    def set_gpus_per_node(cls, v: int | None) -> int:
        # pydantic-settings feeds the raw env string into a ``mode="before"``
        # validator, so ``v`` is a ``str`` when set via ``GPUS_PER_NODE=2`` /
        # ``LLM_DISTRIBUTED__GPUS_PER_NODE=2`` (round-76 deep-dive D3 — the
        # env-scaled path crashed with ``'>' not supported between str and
        # int`` before training); coerce before comparing.
        try:
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"gpus_per_node must be an integer, got {v!r}")
            requested = int(v) if v is not None else None
        except TypeError, ValueError:
            raise ValueError(f"gpus_per_node must be an integer, got {v!r}") from None
        available = torch.cuda.device_count()
        if requested is None:
            return available
        if requested > available:
            raise ValueError(f"Requested {requested} GPUs but only {available} available")
        return requested

    @model_validator(mode="after")
    def validate_3d_grid(self) -> DistributedConfig:
        if self.parallel_strategy != "3d":
            return self
        if self.dp_size < 1 or self.pp_size < 1 or self.tp_size < 1:
            raise ValueError(
                "parallel_strategy='3d' requires explicit, positive dp_size / pp_size / "
                f"tp_size (got dp_size={self.dp_size}, pp_size={self.pp_size}, tp_size={self.tp_size}); "
                "the 3D grid must tile the world exactly."
            )
        return self


class OptimizationConfig(BaseModel):
    """Performance optimization configuration"""

    use_compile: bool = True
    # ``torch.compile`` mode. See torch docs for full semantics.
    #   - ``default``: best general-purpose starting point
    #   - ``reduce-overhead``: uses CUDA graphs — incompatible with
    #     variable-length sequences and KV-cache eviction; only safe for
    #     pure fixed-shape training (no inference with use_cache=True)
    #   - ``max-autotune``: long warmup, picks best kernel per shape
    #   - ``max-autotune-no-cudagraphs``: like max-autotune without graphs
    compile_mode: str = Field(
        "default",
        pattern="^(default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs)$",
        description="torch.compile mode. 'default' is recommended for variable-length training.",
    )
    compile_dynamic: bool | None = Field(
        default=None,
        description=(
            "Mark dynamic dimensions for torch.compile (e.g., the sequence length). "
            "If None, PyTorch's default heuristic is used."
        ),
    )
    use_amp: bool = True
    amp_dtype: str = Field("auto", pattern="^(auto|float16|bfloat16)$")
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    gradient_accumulation_steps: int = Field(1, ge=1)


class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""

    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    save_interval: int = Field(1, gt=0)
    keep_last_n: int = Field(5, gt=0)
    save_best: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration"""

    log_interval: int = 10
    log_level: str = "INFO"
    log_dir: str = "logs"
    save_logs: bool = True


class DataConfig(BaseModel):
    """Data configuration"""

    data_source: str = Field("local", pattern="^(local|hf|dedup_local|dedup_hf)$")
    tokenizer_type: str = Field("simple", pattern="^(simple|hf)$")
    tokenizer_path: str | None = None  # Path to file (simple) or repo_id/path (hf)
    dataset_path: str | None = None
    val_dataset_path: str | None = None  # Optional explicit validation file
    dataset_name: str | None = None  # HuggingFace dataset id when data_source='hf'
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    max_seq_len: int = Field(
        512,
        gt=0,
        description="Context length each sample is truncated/padded to; must be positive (RIL ISS-199 — a non-positive value silently truncates ids and misaligns attention_mask).",
    )
    steps_per_epoch: int | None = Field(
        None,
        gt=0,
        description="Fixed optimizer steps per epoch for streaming DataModules",
    )
    skip_undecodable_rows: bool = Field(
        True,
        description=(
            "Skip text rows the tokenizer cannot encode (characters outside "
            "its vocabulary) with a logged warning instead of aborting the "
            "run. The default character tokenizer is ASCII-only, so a real "
            "corpus almost always contains un-encodable rows; set False to "
            "fail loud on the first such row (round-76 TASK-189)."
        ),
    )

    # Dedup wrapper knobs (only consulted when data_source starts with
    # 'dedup_'). The defaults are no-ops so existing configs are
    # unaffected; users opt in by either switching data_source to
    # 'dedup_local' / 'dedup_hf' or by passing seen_hashes_path.
    seen_hashes_path: str | None = Field(
        None,
        description=(
            "Path to a file holding previously seen content hashes "
            "(one per line, hex-encoded). Consulted by the dedup "
            "wrapper when data_source is a dedup_* variant."
        ),
    )
    write_seen_hashes: bool = Field(
        False,
        description=(
            "If True, the dedup wrapper appends new hashes to "
            "seen_hashes_path as records are yielded. Requires "
            "seen_hashes_path to be set."
        ),
    )
    hash_algo: str = Field(
        "sha256",
        pattern="^[a-z0-9_]+$",
        description=(
            "Hash algorithm for dedup. Any name accepted by "
            "hashlib.new works ('sha256', 'sha1', 'md5', 'blake2b', ...)."
        ),
    )


class PPOConfig(BaseModel):
    """PPO hyperparameters for RLHF training."""

    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    max_grad_norm: float = 1.0
    target_kl: float | None = None
    rollout_batch_size: int = 16
    response_max_len: int = 256
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    policy_lr: float | None = None
    value_lr: float | None = None
    use_ref_model: bool = True
    ref_model_update_freq: int = 0


class RLHFSettings(BaseModel):
    """RLHF-specific paths and options."""

    reward_model_path: str | None = None


class Config(BaseSettings):
    """Main configuration class combining all sub-configurations"""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    rlhf: RLHFSettings = Field(default_factory=RLHFSettings)

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    def save_to_yaml(self, path: str | Path):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump instead of asdict for Pydantic V2
        data = self.model_dump(mode="json")
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.model_validate(config_dict)

    # Note: from_args_and_env removed, CLI logic moves to Typed/CLI tool
