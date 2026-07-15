from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DistributedConfig,
    ModelConfig,
    OptimizationConfig,
    PPOConfig,
    TrainingConfig,
)


@pytest.mark.quick
class TestConfig:
    def test_default_config_creation(self):
        config = Config()
        assert config.model.hidden_size == 512
        assert config.training.batch_size == 128
        assert config.training.lr == 1e-3
        assert config.optimization.num_workers == 4
        assert config.checkpoint.save_interval == 1
        assert config.logging.log_interval == 10
        assert config.ppo.clip_epsilon == 0.2

    def test_nested_config_defaults_without_env(self):
        assert ModelConfig().attn_impl == "mha"
        assert ModelConfig().mlp_impl == "mlp"
        assert PPOConfig().ppo_epochs == 4

    def test_model_config_post_init_ffn_size(self):
        model_config = ModelConfig(hidden_size=128, intermediate_size=None)
        assert model_config.intermediate_size == 128 * 4

        model_config_custom_ffn = ModelConfig(hidden_size=128, intermediate_size=512)
        assert model_config_custom_ffn.intermediate_size == 512

    @pytest.mark.parametrize(
        ("hidden_size", "intermediate_size", "num_layers"),
        [
            (100, 10, 1),  # Invalid hidden_size (not divisible by 8 heads default)
            (100, 0, 1),  # Invalid intermediate_size
            (128, 512, 0),  # Invalid num_layers
        ],
    )
    def test_model_config_validation_errors(self, hidden_size, intermediate_size, num_layers):
        with pytest.raises((ValueError, ValidationError), match=r"hidden_size|Intermediate size|num_layers"):
            ModelConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, num_layers=num_layers)

    @pytest.mark.parametrize(
        ("batch_size", "lr", "epochs"),
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

    def test_optimization_compile_mode_default(self):
        """Default compile_mode is 'default', not 'reduce-overhead'.

        Regression for Finding AL: the hardcoded 'reduce-overhead' mode used
        CUDA graphs that broke variable-length training and KV-cache eviction.
        """
        cfg = OptimizationConfig()
        assert cfg.use_compile is True
        assert cfg.compile_mode == "default"
        assert cfg.compile_dynamic is None

    @pytest.mark.parametrize(
        "mode",
        ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    def test_optimization_compile_mode_accepts_valid(self, mode):
        OptimizationConfig(compile_mode=mode)

    def test_optimization_compile_mode_rejects_unknown(self):
        with pytest.raises(ValidationError, match="String should match pattern"):
            OptimizationConfig(compile_mode="bogus")

    def test_optimization_compile_dynamic_explicit(self):
        cfg = OptimizationConfig(compile_dynamic=False)
        assert cfg.compile_dynamic is False

    def test_checkpoint_config_validation_errors(self):
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            CheckpointConfig(save_interval=0)
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            CheckpointConfig(keep_last_n=0)

    def test_model_config_moe_validation(self):
        with pytest.raises(ValidationError, match="num_experts must be positive when mlp_impl='moe'"):
            ModelConfig(mlp_impl="moe", num_experts=0, top_k=2)

        with pytest.raises(ValidationError, match="top_k must be positive and <= num_experts when mlp_impl='moe'"):
            ModelConfig(mlp_impl="moe", num_experts=4, top_k=0)

        with pytest.raises(ValidationError, match="top_k must be positive and <= num_experts when mlp_impl='moe'"):
            ModelConfig(mlp_impl="moe", num_experts=4, top_k=5)

    def test_config_save_and_load_yaml(self, tmp_path):
        config = Config(
            model=ModelConfig(hidden_size=128, mlp_impl="moe", num_experts=8, top_k=4),
            training=TrainingConfig(batch_size=64, epochs=3),
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.save_to_yaml(yaml_path)

        loaded_config = Config.from_yaml(yaml_path)

        assert loaded_config.model.hidden_size == config.model.hidden_size
        assert loaded_config.model.mlp_impl == config.model.mlp_impl
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
        assert config.use_kv_cache is False

        # Test custom values (mla and flash_attn are both registered attention impls)
        config = ModelConfig(attn_impl="mla", mlp_impl="moe", num_kv_heads=4, num_experts=8, top_k=2)
        assert config.attn_impl == "mla"
        assert config.mlp_impl == "moe"
        assert config.num_kv_heads == 4

    def test_model_config_rejects_unknown_attn_impl(self):
        """Unknown attn_impl fails at config-load time, not deep in forward.

        Regression for Finding E (audit 2026-07-12). Note that ``mha``,
        ``mla``, and ``flash_attn`` are all valid registered impls; we
        pick a clearly-fake name to assert the rejection path.
        """
        with pytest.raises(ValueError, match="Unknown attn_impl"):
            ModelConfig(attn_impl="nonexistent_attention")

    def test_model_config_mla_with_kv_cache_accepted(self):
        """MLA + KV cache is supported (Tier 3 #31, audit 2026-07-12).

        The placeholder MLA caches K, V from ``input_kv_proj`` like MHA
        does; the latent attention then runs over the cached context.
        The architectural caveat is that MLA's output is a uniform-mean
        over latents — see ``MultiLatentAttention`` docstring.
        """
        config = ModelConfig(attn_impl="mla", use_kv_cache=True)
        assert config.attn_impl == "mla"
        assert config.use_kv_cache is True

    def test_model_config_mla_without_kv_cache_ok(self):
        """MLA without KV cache is fine — research/training use case."""
        config = ModelConfig(attn_impl="mla", use_kv_cache=False)
        assert config.attn_impl == "mla"
        assert config.use_kv_cache is False

    def test_model_config_mha_with_kv_cache_ok(self):
        """MHA + KV cache is the canonical serving path."""
        config = ModelConfig(attn_impl="mha", use_kv_cache=True)
        assert config.attn_impl == "mha"
        assert config.use_kv_cache is True

    def test_model_config_flash_attn_registered(self):
        """Flash Attention 2 is now a registered impl (Tier 3 #24).

        Regression for the test_data assumption that ``flash_attn`` was
        a placeholder for "unknown impl"; it is now a real registry entry.
        """
        from llm.core.registry import ATTENTION_KV_CACHE_CAPABILITY

        assert "flash_attn" in ATTENTION_KV_CACHE_CAPABILITY
        config = ModelConfig(attn_impl="flash_attn")
        assert config.attn_impl == "flash_attn"
        # ``use_kv_cache`` is accepted because flash_attn declared its
        # capability at import time.
