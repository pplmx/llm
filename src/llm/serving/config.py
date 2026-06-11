from pydantic import Field
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

    # Generation
    generation_backend: str = "eager"  # eager | batched

    # Performance
    compile_model: bool = False  # Enable torch.compile for acceleration
    max_concurrent_requests: int = 4  # Max concurrent inference requests
    request_timeout: float = 60.0  # Request timeout in seconds

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

    model_config = SettingsConfigDict(env_prefix="LLM_SERVING_")
