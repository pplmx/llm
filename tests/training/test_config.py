from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)


@pytest.mark.quick
class TestConfig:
    def test_default_config_creation(self):
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.distributed, DistributedConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_model_config_post_init_ffn_size(self):
        model_config = ModelConfig(hidden_size=128, intermediate_size=None)
        assert model_config.intermediate_size == 128 * 4

        model_config_custom_ffn = ModelConfig(hidden_size=128, intermediate_size=512)
        assert model_config_custom_ffn.intermediate_size == 512

    @pytest.mark.parametrize(
        "hidden_size, intermediate_size, num_layers",
        [
            (100, 10, 1),  # Invalid hidden_size (not divisible by 8 heads default)
            (100, 0, 1),  # Invalid intermediate_size
            (128, 512, 0),  # Invalid num_layers
        ],
    )
    def test_model_config_validation_errors(self, hidden_size, intermediate_size, num_layers):
        with pytest.raises(ValueError):
            ModelConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, num_layers=num_layers)

    @pytest.mark.parametrize(
        "batch_size, lr, epochs",
        [
            (0, 1e-3, 1),  # Invalid batch_size
            (32, 0, 1),  # Invalid lr
            (32, 1e-3, 0),  # Invalid epochs
        ],
    )
    def test_training_config_validation_errors(self, batch_size, lr, epochs):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=batch_size, lr=lr, epochs=epochs)

    def test_distributed_config_gpus_per_node_default(self):
        # Test default behavior without mocking
        dist_config = DistributedConfig()
        # This will depend on the actual system's GPU count, so we can't assert a fixed value.
        # We can only assert it's non-negative (or None before init if lazy, but Pydantic validators run)
        # Our implementation sets default to None but validator uses torch.cuda.device_count() if None
        # So it should be an int >= 0
        assert dist_config.gpus_per_node >= 0

    def test_distributed_config_validation_errors(self):
        # Test validation directly
        with (
            patch("torch.cuda.device_count", return_value=2),
            pytest.raises(ValidationError, match="Requested 3 GPUs but only 2 available"),
        ):
            DistributedConfig(gpus_per_node=3)

    def test_optimization_config_validation_errors(self):
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            OptimizationConfig(num_workers=-1)

    def test_checkpoint_config_validation_errors(self):
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            CheckpointConfig(save_interval=0)
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            CheckpointConfig(keep_last_n=0)

    def test_model_config_moe_validation(self):
        # Test invalid num_experts
        with pytest.raises(ValidationError, match="num_experts must be positive if use_moe is True"):
            ModelConfig(use_moe=True, num_experts=0, top_k=2)

        # Test invalid top_k
        with pytest.raises(
            ValidationError, match="top_k must be positive and less than or equal to num_experts if use_moe is True"
        ):
            ModelConfig(use_moe=True, num_experts=4, top_k=0)

        with pytest.raises(
            ValidationError, match="top_k must be positive and less than or equal to num_experts if use_moe is True"
        ):
            ModelConfig(use_moe=True, num_experts=4, top_k=5)

    def test_config_save_and_load_yaml(self, tmp_path):
        config = Config(
            model=ModelConfig(hidden_size=128, use_moe=True, num_experts=8, top_k=4),
            training=TrainingConfig(batch_size=64, epochs=3),
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.save_to_yaml(yaml_path)

        loaded_config = Config.from_yaml(yaml_path)

        assert loaded_config.model.hidden_size == config.model.hidden_size
        assert loaded_config.model.use_moe == config.model.use_moe
        assert loaded_config.model.num_experts == config.model.num_experts
        assert loaded_config.model.top_k == config.model.top_k
        assert loaded_config.training.batch_size == config.training.batch_size
        assert loaded_config.training.epochs == config.training.epochs
        # Check other default values are preserved
        assert loaded_config.distributed.master_addr == config.distributed.master_addr

    def test_model_config_registry_params(self):
        # Test default values
        config = ModelConfig()
        assert config.attn_impl == "mha"
        assert config.mlp_impl == "mlp"

        # Test custom values
        config = ModelConfig(attn_impl="flash_attn", mlp_impl="moe", num_kv_heads=4)
        assert config.attn_impl == "flash_attn"
        assert config.mlp_impl == "moe"
        assert config.num_kv_heads == 4
