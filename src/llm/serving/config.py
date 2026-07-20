from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingConfig(BaseSettings):
    """
    Serving Configuration using environment variables.
    """

    # Model configuration
    model_path: str | None = None  # Path to training checkpoint (.pt)
    tokenizer_path: str | None = None  # Path to tokenizer pickle or HF repo
    tokenizer_type: str = Field("simple", pattern="^(simple|hf)$")
    device: str = "auto"

    # Security & Observability
    api_key: str | None = None  # If set, requires this key for access
    log_level: str = "INFO"
    host: str = "127.0.0.1"  # Set LLM_SERVING_HOST=0.0.0.0 for container bind-all

    # Generation
    generation_backend: str = "eager"  # eager | batched

    # Performance
    compile_model: bool = False  # Enable torch.compile for acceleration
    max_concurrent_requests: int = 4  # Max concurrent inference requests
    request_timeout: float = 60.0  # Request timeout in seconds

    # Chat template (OpenAI /v1/chat/completions).
    # Each message is rendered using ``chat_message_template.format(role=...,
    # content=...)``. The rendered messages are joined with newlines and the
    # ``chat_generation_prefix`` is appended at the end so the model knows
    # where to start producing assistant tokens. Override either field to
    # match your fine-tuned model's expected format (e.g., ChatML,
    # Llama-2-chat, Vicuna). None means: use the built-in defaults.
    chat_message_template: str | None = Field(
        default=None,
        description=(
            "Python format string applied to each chat message. "
            "Available placeholders: {role}, {content}. "
            "If None, falls back to '{role}: {content}'."
        ),
    )
    chat_generation_prefix: str | None = Field(
        default=None,
        description=(
            "String appended after all messages, signaling the model to "
            "start generating the assistant response. "
            "If None, falls back to 'Assistant: '."
        ),
    )

    # Paged Attention (block allocator sidecar; model forward still uses KVCache)
    use_paged_attention: bool = False
    max_blocks: int = 256
    block_size: int = 16

    # Prefix Cache
    enable_prefix_cache: bool = False
    max_prefixes: int = 10

    # Model Params (for dummy init if no ckpt)
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128

    # Advanced Arch Params
    num_kv_heads: int | None = None
    num_experts: int = 0
    top_k: int = 0
    attn_impl: str = "mha"
    mlp_impl: str = "mlp"

    # PEFT adapter loading (T2 PEFT #49).
    # Bridges the training-side PEFT save/load surface into the serving
    # loader: trained adapters (LoRA / IA³ / BitFit / Adapter / Pfeiffer
    # / AdaLoRA / QLoRA / Prefix Tuning) are applied to the base model
    # at startup and (optionally) folded into the base weights.
    #
    # Typical workflow:
    #   1. Train with ``TrainingConfig.peft_method="lora"`` +
    #      ``peft_save_path=...`` — the trainer's
    #      ``PEFTAdapterCheckpointCallback`` writes the sidecar on
    #      ``on_train_end``.
    #   2. Serve with ``LLM_SERVING_PEFT_METHOD=lora`` +
    #      ``LLM_SERVING_PEFT_ADAPTER_PATH=...`` — the loader applies
    #      the method and copies the saved adapter values into the
    #      model before the first request.
    peft_method: str | None = Field(
        default=None,
        description=(
            "Registered PEFT method name (e.g. 'lora', 'ia3', 'bitfit'). "
            "When set, the loader applies the method to the base model "
            "after loading the checkpoint. Must be a key in PEFT_REGISTRY."
        ),
    )
    peft_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Kwargs forwarded to ``apply_peft`` (e.g. ``{'rank': 8, "
            "'alpha': 16.0}`` for LoRA). Overridden by ``peft_kwargs`` "
            "stored inside the sidecar when ``peft_adapter_path`` is set."
        ),
    )
    peft_adapter_path: str | None = Field(
        default=None,
        description=(
            "Path to a PEFT sidecar file written by ``save_peft`` (typically "
            "``{checkpoint_dir}/peft_adapter_{method}.bin`` from "
            "``PEFTAdapterCheckpointCallback``). When set, the loader "
            "loads the saved adapter values into the model after applying "
            "the method."
        ),
    )
    peft_merge: bool = Field(
        default=False,
        description=(
            "If True, fold the adapter into the base weights at serve "
            "time (saves per-token routing overhead). Loses the ability "
            "to ``disable_peft`` / ``unmerge_peft`` at runtime. Refused "
            "for methods that don't expose a merge helper (bitfit / "
            "qlora / prefix_tuning)."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="LLM_SERVING_")

    # Methods that don't expose a merge helper. ``peft_merge=True`` is
    # rejected for these in the model_validator below — failing loud at
    # startup is better than silently no-op'ing at first-request time.
    _NON_MERGEABLE_METHODS: frozenset[str] = frozenset({"bitfit", "qlora", "prefix_tuning"})

    @field_validator("peft_method")
    @classmethod
    def _validate_peft_method(cls, value: str | None) -> str | None:
        """Reject unknown PEFT method names at config-load time.

        Built-ins are registered lazily — calling
        :func:`ensure_methods_registered` here is idempotent so the
        validator triggers the bootstrap without surprising the caller
        (subsequent calls return immediately).
        """
        if value is None:
            return None
        # Lazy import to avoid pulling PEFT into the serving import
        # graph just to instantiate the config (the auth / CLI guards
        # build a config at startup with no PEFT).
        from llm.core.peft import PEFT_REGISTRY
        from llm.core.peft.registry import ensure_methods_registered

        ensure_methods_registered()
        try:
            PEFT_REGISTRY.get(value)
        except ValueError as exc:
            raise ValueError(f"Unknown PEFT method {value!r}. Registered methods: {PEFT_REGISTRY.names()}.") from exc
        return value

    @model_validator(mode="after")
    def _validate_peft_field_consistency(self) -> ServingConfig:
        """Cross-field PEFT validation.

        - ``peft_adapter_path`` and ``peft_kwargs`` imply a method is set.
        - ``peft_merge=True`` requires a method that exposes a merge
          helper (raises ``ValueError`` for bitfit / qlora /
          prefix_tuning).
        """
        if self.peft_method is None:
            if self.peft_adapter_path is not None:
                raise ValueError(
                    "peft_adapter_path is set but peft_method is None. "
                    "Set peft_method (e.g. 'lora') to identify which "
                    "method the sidecar was saved with."
                )
            if self.peft_kwargs:
                raise ValueError(
                    "peft_kwargs is set but peft_method is None. Set peft_method (e.g. 'lora') to use the kwargs."
                )
            return self

        if self.peft_merge and self.peft_method in self._NON_MERGEABLE_METHODS:
            raise ValueError(
                f"peft_merge=True is not supported for method "
                f"{self.peft_method!r}: it does not expose a merge "
                f"helper. Either set peft_merge=False or pick a "
                f"different method."
            )
        return self
