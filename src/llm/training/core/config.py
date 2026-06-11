from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
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
    use_moe: bool = Field(False, description="Whether to use MoE in TransformerBlocks")
    num_experts: int = Field(0, description="Number of experts if use_moe is True")
    top_k: int = Field(0, description="Number of top experts to select if use_moe is True")

    # Registry keys
    attn_impl: str = "mha"
    mlp_impl: str = "mlp"
    norm_impl: str = "layer_norm"  # Placeholder, currently not strictly used in config-to-model mapping yet except via kwargs or manual pass

    @model_validator(mode="after")
    def check_consistency(self) -> "ModelConfig":
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        if self.intermediate_size <= 0:
            raise ValueError("Intermediate size must be positive")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        if self.use_moe:
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive if use_moe is True.")
            if self.top_k <= 0 or self.top_k > self.num_experts:
                raise ValueError("top_k must be positive and less than or equal to num_experts if use_moe is True.")
        return self


class TrainingConfig(BaseModel):
    """Training configuration"""

    batch_size: int = Field(128, gt=0)
    epochs: int = Field(10, gt=0)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = 0.01
    num_samples: int = 20000
    scheduler_type: str = Field("cosine", pattern="^(cosine|step|plateau)$")
    warmup_epochs: int = 1
    gradient_clip_val: float = 1.0
    run_validation: bool = True


class DistributedConfig(BaseSettings):
    """Distributed configuration (aware of environment variables)"""

    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int | None = None  # Lazy init
    node_rank: int = 0
    backend: str = "nccl"
    parallel_strategy: str = Field("ddp", pattern="^(ddp|fsdp)$")

    @field_validator("gpus_per_node", mode="before")
    @classmethod
    def set_gpus_per_node(cls, v: int | None) -> int:
        available = torch.cuda.device_count()
        if v is None:
            return available
        if v > available:
            raise ValueError(f"Requested {v} GPUs but only {available} available")
        return v


class OptimizationConfig(BaseModel):
    """Performance optimization configuration"""

    use_compile: bool = True
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

    data_source: str = Field("local", pattern="^(local|hf)$")
    tokenizer_type: str = Field("simple", pattern="^(simple|hf)$")
    tokenizer_path: str | None = None  # Path to file (simple) or repo_id/path (hf)
    dataset_path: str | None = None
    val_dataset_path: str | None = None  # Optional explicit validation file
    dataset_name: str | None = None  # HuggingFace dataset id when data_source='hf'
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    max_seq_len: int = 512
    steps_per_epoch: int | None = Field(
        None,
        gt=0,
        description="Fixed optimizer steps per epoch for streaming DataModules",
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


# Backward-compatible alias for YAML configs and external imports.
PPOSettings = PPOConfig


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
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open() as f:
            config_dict = yaml.safe_load(f) or {}
        return cls.model_validate(config_dict)

    # Note: from_args_and_env removed, CLI logic moves to Typed/CLI tool
