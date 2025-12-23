from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingConfig(BaseSettings):
    """
    Serving Configuration using environment variables.
    """

    # Model configuration
    model_path: str | None = None  # Path to model checkpoint
    tokenizer_path: str | None = None  # Path to tokenizer (if different)
    device: str = "auto"

    # Security & Observability
    api_key: str | None = None  # If set, requires this key for access
    log_level: str = "INFO"

    # Performance
    compile_model: bool = False  # Enable torch.compile for acceleration

    # Model Params (for dummy init if no ckpt)
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128

    model_config = SettingsConfigDict(env_prefix="LLM_SERVING_")
