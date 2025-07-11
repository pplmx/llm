import argparse
import os
import sys
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import torch
import yaml

# ============================================================================
# 配置管理
# ============================================================================


@dataclass
class ModelConfig:
    """模型配置"""

    hidden_size: int = 512
    ffn_hidden_size: int | None = None
    num_layers: int = 2
    dropout: float = 0.1
    use_moe: bool = False  # New: Whether to use MoE in TransformerBlocks
    num_experts: int = 0  # New: Number of experts if use_moe is True
    top_k: int = 0  # New: Number of top experts to select if use_moe is True

    def __post_init__(self):
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4

        if self.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if self.ffn_hidden_size <= 0:
            raise ValueError("FFN hidden size must be positive")
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if self.use_moe:
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive if use_moe is True.")
            if self.top_k <= 0 or self.top_k > self.num_experts:
                raise ValueError("top_k must be positive and less than or equal to num_experts if use_moe is True.")


@dataclass
class TrainingConfig:
    """训练配置"""

    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_samples: int = 20000
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 1
    gradient_clip_val: float = 1.0
    run_validation: bool = True  # Added for validation loop

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")


@dataclass
class DistributedConfig:
    """分布式配置"""

    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int = field(default_factory=lambda: torch.cuda.device_count())
    node_rank: int = 0
    backend: str = "nccl"

    def __post_init__(self):
        available_gpus = torch.cuda.device_count()
        if self.gpus_per_node > available_gpus:
            raise ValueError(f"Requested {self.gpus_per_node} GPUs but only {available_gpus} available")


@dataclass
class OptimizationConfig:
    """性能优化配置"""

    use_compile: bool = True
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False  # Changed default to False for experimentation

    def __post_init__(self):
        if self.num_workers < 0:
            raise ValueError("Number of workers must be non-negative")


@dataclass
class CheckpointConfig:
    """检查点配置"""

    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    save_interval: int = 1
    keep_last_n: int = 5  # 保留最近N个检查点
    save_best: bool = True

    def __post_init__(self):
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
        if self.keep_last_n <= 0:
            raise ValueError("Keep last N must be positive")


@dataclass
class LoggingConfig:
    """日志配置"""

    log_interval: int = 10  # 每隔多少个batch记录一次
    log_level: str = "INFO"
    log_dir: str = "logs"
    save_logs: bool = True


@dataclass
class Config:
    """主配置类，组合所有配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def save_to_yaml(self, path: str | Path):
        """将配置保存到 YAML 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """从 YAML 文件加载配置"""
        path = Path(path)
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Recursively reconstruct dataclasses
        def _from_dict(data_class, data_dict):
            if not isinstance(data_dict, dict):
                return data_dict

            field_values = {}
            for field_name, field_type in data_class.__annotations__.items():
                if field_name in data_dict:
                    if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:  # Handle Optional
                        actual_type = next((t for t in typing.get_args(field_type) if t is not types.NoneType), None)
                        if actual_type and hasattr(actual_type, "__annotations__"):  # Nested dataclass
                            field_values[field_name] = _from_dict(actual_type, data_dict[field_name])
                        else:
                            field_values[field_name] = data_dict[field_name]
                    elif hasattr(field_type, "__annotations__"):  # Nested dataclass
                        field_values[field_name] = _from_dict(field_type, data_dict[field_name])
                    else:
                        field_values[field_name] = data_dict[field_name]
            return data_class(**field_values)

        return _from_dict(cls, config_dict)

    @classmethod
    def from_args_and_env(cls) -> "Config":
        """从命令行参数和环境变量创建配置"""
        parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")

        # 动态添加参数，避免重复
        def add_args_from_dataclass(parser_, dc_name, dc_instance):
            for name, type_hint in dc_instance.__annotations__.items():
                arg_name = (
                    f"--{dc_name.lower().replace('config', '')}-{name.replace('_', '-')}"  # e.g., --model-hidden-size
                )

                # Determine the type for argparse
                type_for_argparse = type_hint
                origin = typing.get_origin(type_hint)
                if origin is types.UnionType or origin is typing.Union:
                    base_type = next((t for t in typing.get_args(type_hint) if t is not types.NoneType), None)
                    if base_type:
                        type_for_argparse = base_type
                    else:
                        continue  # Skip Optional[None] types

                # Handle boolean flags for argparse
                # Handle boolean flags for argparse
                if type_for_argparse is bool:
                    # Argparse automatically sets dest to the argument name without leading dashes
                    # For --no-flag, dest is 'flag'
                    # For --flag, dest is 'flag'
                    # So, we need to ensure the dest matches the field name
                    if getattr(dc_instance, name) is True:  # Default is True, so provide --no-flag
                        parser_.add_argument(
                            f"--no-{dc_name.lower().replace('config', '')}-{name.replace('_', '-')}",
                            action="store_false",
                            dest=f"{dc_name.lower().replace('config', '')}_{name}",
                            help=f"Disable {dc_name}.{name}",
                        )
                    else:  # Default is False, so provide --flag
                        parser_.add_argument(
                            arg_name,
                            action="store_true",
                            dest=f"{dc_name.lower().replace('config', '')}_{name}",
                            help=f"Enable {dc_name}.{name}",
                        )
                else:
                    parser_.add_argument(
                        arg_name, type=type_for_argparse, default=None, help=f"Override {dc_name}.{name}"
                    )

        config = cls()
        add_args_from_dataclass(parser, "training", config.training)
        add_args_from_dataclass(parser, "model", config.model)
        add_args_from_dataclass(parser, "optimization", config.optimization)
        add_args_from_dataclass(parser, "checkpoint", config.checkpoint)
        add_args_from_dataclass(parser, "logging", config.logging)

        parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
        parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

        args = parser.parse_args()

        # 从环境变量更新分布式配置
        config.distributed.node_rank = int(os.environ.get("NODE_RANK", config.distributed.node_rank))
        config.distributed.num_nodes = int(os.environ.get("NUM_NODES", config.distributed.num_nodes))
        config.distributed.master_addr = os.environ.get("MASTER_ADDR", config.distributed.master_addr)
        config.distributed.master_port = os.environ.get("MASTER_PORT", config.distributed.master_port)
        config.distributed.gpus_per_node = int(os.environ.get("GPUS_PER_NODE", torch.cuda.device_count()))

        # Apply parsed arguments to config
        for group_name, group_config in config.__dict__.items():
            if group_name.startswith("_"):
                continue
            for key, _type_hint in group_config.__annotations__.items():
                # Construct the expected attribute name in 'args'
                arg_attr_name = f"{group_name.lower().replace('config', '')}_{key}"

                # Check if the argument was actually provided in the command line
                # For boolean flags, argparse sets the attribute even if not explicitly provided
                # if it's a store_true/store_false action.
                # We need to check if the argument was *explicitly* set by the user.
                # This is tricky with argparse. A simpler approach for testing is to
                # ensure the test cases explicitly set all relevant args.

                # For now, let's assume if the attribute exists in args, it was set.
                # This might need refinement if more complex boolean logic is introduced.
                if hasattr(args, arg_attr_name):
                    arg_val = getattr(args, arg_attr_name)
                    # Only update if the value is not None (for non-boolean types)
                    # or if it's a boolean and was explicitly set (not default)
                    if arg_val is not None or (
                        isinstance(arg_val, bool) and arg_attr_name in sys.argv
                    ):  # This check is still problematic
                        setattr(group_config, key, arg_val)

        # Special handling for --no-compile and --no-amp flags
        # These are global flags, not tied to a specific dataclass field directly
        if hasattr(args, "no_compile") and args.no_compile:
            config.optimization.use_compile = False
        if hasattr(args, "no_amp") and args.no_amp:
            config.optimization.use_amp = False

        # 手动后处理
        config.model.__post_init__()
        config.training.__post_init__()

        return config
