from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DataConfig,
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

    def test_config_from_yaml_missing_path_raises(self, tmp_path):
        """An explicitly-passed --config-path that doesn't exist is a user
        error, not a silent fall-back to defaults."""
        missing = tmp_path / "no-such-config.yaml"
        with pytest.raises(FileNotFoundError, match="config file not found"):
            Config.from_yaml(missing)

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

    @pytest.mark.parametrize(
        ("field", "bad", "valid"),
        [
            ("norm_impl", "layernorm", "rms_norm"),
            ("mlp_impl", "mlpp", "moe"),
            ("attn_impl", "mhaa", "mha"),
        ],
    )
    def test_registry_impl_names_validated_at_config_load(self, field, bad, valid):
        """All three registry keys must fail at load, not at model build.

        ``attn_impl`` was validated here while ``norm_impl`` / ``mlp_impl`` were
        only checked inside ``DecoderModel`` construction — a typo survived
        config load, CLI parsing and startup, then failed deep in the model.
        """
        with pytest.raises(ValidationError, match=f"Unknown {field}"):
            ModelConfig(**{field: bad})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("norm_impl", "rms_norm"),
            ("norm_impl", "layer_norm"),
            ("attn_impl", "mha"),
            ("attn_impl", "mla"),
        ],
    )
    def test_valid_registry_impl_names_accepted(self, field, value):
        """The guard must not reject legitimate registry entries."""
        assert getattr(ModelConfig(**{field: value}), field) == value

    def test_unknown_config_key_raises_with_suggestion(self, tmp_path):
        """A typo must not silently evaporate.

        ``Config`` is a ``BaseSettings`` with ``extra="ignore"`` so unrelated
        ``LLM_*`` environment variables cannot crash a run — but that same
        leniency made a typo in a YAML file a no-op: the run continued with the
        default and the user believed the setting applied.
        """
        p = tmp_path / "typo.yaml"
        p.write_text("model:\n  hidden_dim: 512\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"unknown key.*model\.hidden_dim"):
            Config.from_yaml(p)

    def test_unknown_config_key_suggests_the_real_name(self, tmp_path):
        """The message must point at the intended key, not just complain."""
        p = tmp_path / "typo.yaml"
        p.write_text("training:\n  epoch: 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"did you mean training\.epochs"):
            Config.from_yaml(p)

    def test_legacy_learning_rate_key_names_the_rename(self, tmp_path):
        """``learning_rate`` is an accepted ALIAS, so the message must say so.

        ``lr`` declares ``validation_alias=AliasChoices("lr", "learning_rate")``,
        so pydantic accepts both spellings when the model is built directly — but
        the YAML guard only inspects ``model_fields``, where ``learning_rate`` is
        invisible. The user saw the full known-key list and no hint, even though
        the field explicitly accepts that spelling. This is the exact typo that
        shipped in eight configs before anyone noticed.
        """
        p = tmp_path / "legacy.yaml"
        p.write_text("training:\n  learning_rate: 2.0e-5\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"training\.learning_rate \(rename it to training\.lr\)"):
            Config.from_yaml(p)

    def test_alias_hint_takes_precedence_over_fuzzy_match(self, tmp_path):
        """An exact alias must be reported as a rename, not a vague suggestion."""
        p = tmp_path / "legacy.yaml"
        p.write_text("training:\n  learning_rate: 1e-3\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"rename it to training\.lr") as excinfo:
            Config.from_yaml(p)
        message = str(excinfo.value)
        assert "rename it to training.lr" in message
        assert "did you mean" not in message, "an exact alias should not fall back to fuzzy matching"

    def test_serving_config_is_rejected_with_the_right_advice(self, tmp_path):
        """A serving file in `configs/` must not look like a bag of typos."""
        p = tmp_path / "serve.yaml"
        p.write_text("model_path: checkpoints/best\nhost: 0.0.0.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="looks like a SERVING config"):
            Config.from_yaml(p)

    def test_shipped_training_configs_all_load(self):
        """The repo's own configs must satisfy the same guard.

        This is the regression that matters: the guard found
        `training.learning_rate` (real key: `lr`) in EIGHT shipped configs,
        where the intended learning rate was silently replaced by the 1e-3
        default — a 50x error for `sft_alpaca.yaml`'s 2.0e-5.
        """
        import pathlib

        import yaml

        configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
        checked = 0
        for path in sorted(configs_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if {"model_path", "tokenizer_path", "host", "api_key"} & set(raw):
                continue  # serving schema, covered by ServingConfig
            Config.from_yaml(path)
            checked += 1
        assert checked >= 10, f"expected to check the shipped training configs, only saw {checked}"

    def test_log_level_validated_at_config_load(self):
        """An unknown level must fail here, not in a logging call later.

        The two consumers failed differently and late:
        ``configure_logging`` handed the string to
        ``logging.basicConfig(level=...)`` (``ValueError: Unknown level``)
        while ``Logger`` did ``getattr(logging, name)``
        (``AttributeError: module 'logging' has no attribute 'BOGUS'``).
        """
        from llm.training.core.config import LoggingConfig

        for bad in ("BOGUS", "", "not-a-level"):
            with pytest.raises(ValueError, match="unknown log_level"):
                LoggingConfig(log_level=bad)

    def test_log_level_normalised_to_upper(self):
        """A lowercase level is accepted and normalised, so ``debug`` works."""
        from llm.training.core.config import LoggingConfig

        assert LoggingConfig(log_level="debug").log_level == "DEBUG"
        assert LoggingConfig(log_level="Warning").log_level == "WARNING"

    def test_model_config_rejects_non_positive_num_kv_heads(self):
        """``num_kv_heads <= 0`` must fail at config load, not inside a backend.

        Without this check the failure mode depended on the attention backend:
        ``mha`` reached ``num_heads % 0`` and raised a bare
        ``ZeroDivisionError`` (no mention of which config key was wrong),
        while ``num_kv_heads=-2`` built a negative-width projection without
        complaint. Reject the value once, up front, for every backend.
        """
        for bad in (0, -2):
            with pytest.raises(ValueError, match="num_kv_heads must be positive"):
                ModelConfig(num_heads=4, num_kv_heads=bad)

    def test_model_config_rejects_num_kv_heads_exceeding_num_heads(self):
        """GQA uses FEWER KV heads than query heads; the reverse is nonsense.

        ``num_kv_heads > num_heads`` used to be accepted by
        ``ModelConfig`` and then rejected by the ``mha`` backend with a
        divisibility error — and silently ACCEPTED by the ``mla`` backend,
        which has no GQA repeat at all. Both paths now fail at config load.
        """
        for impl in ("mha", "mla"):
            with pytest.raises(ValueError, match="must not exceed num_heads"):
                ModelConfig(num_heads=4, num_kv_heads=8, attn_impl=impl)

    def test_model_config_rejects_indivisible_num_kv_heads(self):
        """``num_heads % num_kv_heads != 0`` cannot be split into equal groups."""
        with pytest.raises(ValueError, match="must be divisible by num_kv_heads"):
            ModelConfig(num_heads=4, num_kv_heads=3)

    def test_model_config_accepts_valid_gqa_split(self):
        """A genuine GQA split (fewer, divisor KV heads) still validates."""
        config = ModelConfig(num_heads=8, num_kv_heads=2, attn_impl="mha")
        assert config.num_kv_heads == 2

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

    def test_model_config_rejects_rope_for_mla(self):
        """``use_rope=True`` with ``attn_impl='mla'`` is
        rejected at config time. MLA's static-latent-query scheme performs no
        positional encoding, and its ``**`` catch-all silently swallowed the
        RoPE kwargs — a config validated, the model built, and the position
        encoding was simply ABSENT (silently wrong). Refuse loudly."""
        with pytest.raises(ValueError, match="use_rope"):
            ModelConfig(attn_impl="mla", use_rope=True)

    def test_model_config_accepts_rope_for_mha(self):
        """RoPE is a first-class option for MHA."""
        config = ModelConfig(attn_impl="mha", use_rope=True, use_kv_cache=True)
        assert config.use_rope is True


@pytest.mark.quick
class TestShippedConfigsNoDeadKeys:
    """r176 guard: shipped configs must never carry a key pydantic silently
    drops (``extra='ignore'``). The r175/r176 sweeps found
    ``training.gradient_accumulation_steps`` (the real key lives under
    ``optimization.*``) in dpo_ultrafeedback.yaml / sft_alpaca.yaml /
    example.yaml — users following those configs got the DEFAULT accumulation,
    not the documented effective batch. Encode the intended effective batch
    here so this class of silently-ignored knob can't ship again.
    """

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

    # config -> (intended optimization.gradient_accumulation_steps, intended training.lr)
    CONFIGS: ClassVar[dict[str, tuple[int, float]]] = {
        "configs/dpo_ultrafeedback.yaml": (8, 5.0e-7),
        "configs/sft_alpaca.yaml": (4, 2.0e-5),
        "configs/example.yaml": (1, 0.001),
    }

    def test_shipped_configs_apply_intended_gradient_accumulation(self):
        for rel, (want_steps, want_lr) in self.CONFIGS.items():
            cfg = Config.from_yaml(self.REPO_ROOT / rel)
            assert cfg.optimization.gradient_accumulation_steps == want_steps, rel
            assert cfg.training.lr == want_lr, rel


class TestDataSourceCustomSources:
    """r177: DataConfig's old hardcoded ``^(local|hf|dedup_local|dedup_hf)$``
    pattern rejected every third-party ``llm.data_sources`` plugin name at
    parse time, making the documented custom-source recipe (streaming.md)
    impossible. A name SOURCE_REGISTRY resolves must validate; unknown names
    still fail fast with the available list.
    """

    def test_builtin_sources_parse(self):
        for name in ("local", "hf", "dedup_local", "dedup_hf"):
            assert DataConfig(data_source=name).data_source == name

    def test_unknown_source_fails_fast(self):
        with pytest.raises(ValueError, match="unknown data_source"):
            DataConfig(data_source="bogus_source")

    def test_plugin_registered_name_is_accepted(self):
        from llm.data.sources import SOURCE_REGISTRY, ensure_sources_registered

        ensure_sources_registered()
        # replace() is idempotent, so repeat runs / collocated tests are fine.
        SOURCE_REGISTRY.replace("s3_test_plugin", lambda cfg: None)

        cfg = DataConfig(data_source="s3_test_plugin")
        assert cfg.data_source == "s3_test_plugin"


class TestNonFiniteOptimizerValues:
    """An infinite learning rate must be rejected, not applied to the weights."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_lr_must_be_finite(self, bad):
        """``gt=0`` alone lets ``inf`` through (``inf > 0`` is True).

        Measured: ``TrainingConfig(lr=inf)`` was accepted and one SGD step turned
        every weight into ``-inf``. ``nan`` happened to be rejected because a NaN
        comparison fails the ``gt=0`` check, but relying on that leaves the
        ``inf`` hole.
        """
        from llm.training.core.config import TrainingConfig

        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=bad)

    def test_normal_learning_rates_still_accepted(self):
        from llm.training.core.config import TrainingConfig

        for value in (1e-3, 2e-5, 5e-7, 1.0):
            assert TrainingConfig(lr=value).lr == value


class TestWarmupEpochs:
    """``warmup_epochs`` bounds and the over-long-warmup warning."""

    def test_negative_warmup_rejected(self):
        """A negative warmup has no meaning and produced a nonsense schedule.

        Measured before the guard: ``warmup_epochs=-1`` made the cosine schedule
        START higher (0.00091) than with ``warmup_epochs=0`` (0.00087) — the LR
        rose out of nothing, with no error.
        """
        from llm.training.core.config import TrainingConfig

        with pytest.raises(ValueError, match="warmup_epochs"):
            TrainingConfig(warmup_epochs=-1)

    def test_zero_warmup_is_valid(self):
        """``0`` disables warmup and is the documented flat-LR choice."""
        from llm.training.core.config import TrainingConfig

        assert TrainingConfig(warmup_epochs=0).warmup_epochs == 0

    def test_warmup_longer_than_epochs_warns(self, caplog):
        """``warmup_epochs > epochs`` must warn: the decay never starts.

        Measured: ``epochs=3, warmup_epochs=5`` leaves the LR on the rising half
        for the whole run (peak 0.0006 against a 0.001 base), so training happens
        at 60% of the requested rate with nothing in the logs.
        """
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            config = TrainingConfig(epochs=3, warmup_epochs=5)

        assert config.warmup_epochs == 5  # accepted: a pure-warmup run is legal
        assert any("warmup_epochs" in record.getMessage() for record in caplog.records), (
            "an over-long warmup must be reported"
        )

    def test_normal_warmup_does_not_warn(self, caplog):
        """No false positive on the shipped-style configuration."""
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            TrainingConfig(epochs=3, warmup_epochs=1)

        assert not any("warmup_epochs" in record.getMessage() for record in caplog.records)

    def test_warmup_equal_to_epochs_still_reaches_the_peak(self):
        """``warmup_epochs == epochs`` is allowed and ends AT the peak LR.

        Pins the boundary the ``>`` comparison relies on: one more epoch would
        warn, exactly equal does not, and the schedule still peaks.
        """
        import torch
        import torch.nn as nn

        from llm.training.core.config import Config
        from llm.training.tasks.lm_task import LanguageModelingTask

        config = Config(training=TrainingConfig(epochs=3, warmup_epochs=3, scheduler_type="cosine"))
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        task = LanguageModelingTask.__new__(LanguageModelingTask)
        task.config = config
        scheduler = task.build_scheduler(optimizer)

        peak = 0.0
        for _ in range(3):
            scheduler.step()
            peak = max(peak, optimizer.param_groups[0]["lr"])
        assert peak == pytest.approx(1e-3, rel=1e-6), "warmup == epochs should end at the base LR"


class TestPlateauWithoutValidation:
    """``scheduler_type='plateau'`` needs a metric that actually plateaus."""

    def test_plateau_without_validation_warns(self, caplog):
        """ReduceLROnPlateau fed the TRAIN loss barely ever reduces the LR.

        Measured over a real 10-epoch run with ``run_validation=False``, the LR
        stayed at its base value for every epoch — training loss keeps inching
        down while a model learns, so ``ReduceLROnPlateau`` never sees a plateau.
        Not fatal (in-sample convergence does eventually flatten, and then the
        reduction fires), which is why this warns instead of raising.
        """
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            config = TrainingConfig(scheduler_type="plateau", run_validation=False)

        assert config.scheduler_type == "plateau"  # still accepted
        assert any("plateau" in record.getMessage() for record in caplog.records), (
            "the ineffective combination must be reported"
        )

    def test_plateau_with_validation_does_not_warn(self, caplog):
        """The intended setup stays quiet."""
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            TrainingConfig(scheduler_type="plateau", run_validation=True)

        assert not any("plateau" in record.getMessage() for record in caplog.records)

    def test_cosine_without_validation_does_not_warn(self, caplog):
        """Schedules that do not need a metric must not trip the guard."""
        import logging

        from llm.training.core.config import TrainingConfig

        for scheduler in ("cosine", "step"):
            with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
                TrainingConfig(scheduler_type=scheduler, run_validation=False)
            assert not any("plateau" in record.getMessage() for record in caplog.records), (
                f"{scheduler} should not trigger the plateau warning"
            )


class TestInertSwitchSubOptions:
    """Sub-options set while their master switch is off change nothing."""

    @pytest.mark.parametrize(
        ("master", "suboption", "value"),
        [
            ("use_qat", "qat_bits", 4),
            ("use_qat", "qat_quant_activation", True),
            ("use_adalora", "adalora_init_rank", 16),
            ("use_prefix_tuning", "prefix_tuning_len", 20),
            ("use_ia3", "ia3_init_scale", 0.05),
            ("use_adapter", "adapter_bottleneck_dim", 32),
        ],
    )
    def test_configured_suboption_without_its_switch_warns(self, master, suboption, value, caplog):
        """Each group must report the ignored option by name.

        The sub-options are only read inside ``if use_<x>`` in the task's
        ``build_model``, so setting one with the switch off produces a config
        that LOOKS configured while behaving as if every knob were default.
        """
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            TrainingConfig(**{master: False, suboption: value})

        messages = [record.getMessage() for record in caplog.records]
        assert any(master in message and suboption in message for message in messages), (
            f"setting {suboption} with {master}=False was not reported: {messages}"
        )

    def test_default_config_does_not_warn(self, caplog):
        """A config that never mentions the sub-options stays quiet.

        The guard keys on ``model_fields_set`` (intent), not on value comparison,
        so the defaults — which are all valid — cannot produce noise.
        """
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            TrainingConfig()

        assert not any("no effect" in record.getMessage() for record in caplog.records)

    def test_enabling_the_switch_silences_the_warning(self, caplog):
        """The same sub-option is fine once its switch is on."""
        import logging

        from llm.training.core.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            TrainingConfig(use_qat=True, qat_bits=4)

        assert not any("no effect" in record.getMessage() for record in caplog.records)

    def test_yaml_path_reports_inert_suboptions(self, tmp_path, caplog):
        """The warning must also fire for YAML, the path users actually take.

        ``model_fields_set`` has to survive ``Config.from_yaml`` for the intent
        check to work there — verified, and pinned here so a loader change that
        stopped tracking explicit keys would be caught.
        """
        import logging

        from llm.training.core.config import Config

        config_path = tmp_path / "inert.yaml"
        config_path.write_text(
            "training:\n  use_qat: false\n  qat_bits: 4\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            config = Config.from_yaml(str(config_path))

        assert config.training.qat_bits == 4  # the value IS stored, just unused
        emitted = [record.getMessage() for record in caplog.records]
        assert any("no effect" in message for message in emitted), (
            f"the YAML load path dropped the explicit-set information; captured: {emitted}"
        )


class TestOptimizationInertSubOptions:
    """``OptimizationConfig`` has the same switch/sub-option shape."""

    @pytest.mark.parametrize(
        ("switch", "suboption", "value"),
        [
            ("use_compile", "compile_mode", "max-autotune"),
            ("use_compile", "compile_dynamic", True),
            ("use_amp", "amp_dtype", "float16"),
        ],
    )
    def test_suboption_without_its_switch_warns(self, switch, suboption, value, caplog):
        """``compile_mode`` / ``amp_dtype`` are only read inside their switch branch.

        Setting one with the switch off produced a config that looks tuned while
        the engine never reads the value — the same silent no-op the
        ``TrainingConfig`` guard covers for the PEFT/QAT groups.
        """
        import logging

        from llm.training.core.config import OptimizationConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            OptimizationConfig(**{switch: False, suboption: value})

        messages = [record.getMessage() for record in caplog.records]
        assert any(switch in message and suboption in message for message in messages), (
            f"setting {suboption} with {switch}=False was not reported: {messages}"
        )

    def test_default_optimization_config_does_not_warn(self, caplog):
        """Defaults must stay quiet (they are all valid)."""
        import logging

        from llm.training.core.config import OptimizationConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            OptimizationConfig()

        assert not any("no effect" in record.getMessage() for record in caplog.records)

    def test_enabled_switches_do_not_warn(self, caplog):
        """The options are legitimate once their switch is on."""
        import logging

        from llm.training.core.config import OptimizationConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            OptimizationConfig(use_compile=True, compile_mode="max-autotune", use_amp=True, amp_dtype="float16")

        assert not any("no effect" in record.getMessage() for record in caplog.records)

    def test_num_workers_zero_with_persistent_workers_still_builds_a_loader(self):
        """``num_workers=0`` + ``persistent_workers=True`` must not reach torch.

        torch's DataLoader raises ``ValueError: persistent_workers option needs
        num_workers > 0``. The config layer accepts the combination, so the data
        module has to downgrade the flag — and this builds the real DataLoader
        rather than restating the boolean, so the guard is exercised for real.
        """
        import torch
        from torch.utils.data import DistributedSampler, TensorDataset

        from llm.data.modules.map_base import SamplerMapDataModule
        from llm.training.core.config import Config

        class _Concrete(SamplerMapDataModule):
            """Minimal concrete subclass — only ``build_dataloader`` is exercised."""

            def prepare_data(self) -> None:
                return None

            def setup(self) -> None:
                return None

        config = Config(
            optimization=OptimizationConfig(num_workers=0, persistent_workers=True, prefetch_factor=4),
        )
        module = _Concrete(config)
        dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)

        # Must not raise: torch rejects both persistent_workers and
        # prefetch_factor when num_workers == 0.
        loader = module.build_dataloader(dataset, sampler)

        assert loader.num_workers == 0
        assert list(loader), "the loader must still yield batches"


class TestUnusedParallelSizes:
    """A parallel size knob set under a strategy that never reads it."""

    @pytest.mark.parametrize(
        ("strategy", "field", "value"),
        [
            ("ddp", "tp_size", 2),
            ("fsdp", "tp_size", 2),
            ("ddp", "pp_size", 2),
            ("tp", "pp_size", 2),
        ],
    )
    def test_size_under_the_wrong_strategy_warns(self, strategy, field, value, caplog):
        """``wrap_model_for_training`` reads tp_size only in 'tp'/'3d' and pp_size only in 'pp'/'3d'.

        Every combination is accepted silently, so ``parallel_strategy: ddp``
        plus ``tp_size: 2`` runs plain DDP while looking like tensor parallelism
        was enabled — the kind of config that costs a debugging session.
        """
        import logging

        from llm.training.core.config import DistributedConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DistributedConfig(parallel_strategy=strategy, **{field: value})

        messages = [record.getMessage() for record in caplog.records]
        assert any(field in message and strategy in message for message in messages), (
            f"{field}={value} under strategy {strategy!r} was not reported: {messages}"
        )

    @pytest.mark.parametrize(
        ("strategy", "field"),
        [("tp", "tp_size"), ("pp", "pp_size"), ("3d", "tp_size"), ("3d", "pp_size")],
    )
    def test_size_under_its_own_strategy_is_silent(self, strategy, field, caplog):
        """The legitimate pairings must not warn."""
        import logging

        from llm.training.core.config import DistributedConfig

        kwargs = {field: 2}
        if strategy == "3d":
            # 3d needs the full grid; supply the other dims so the grid
            # validator (not this warning) is what would complain.
            kwargs = {"tp_size": 2, "pp_size": 2, "dp_size": 1}
        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DistributedConfig(parallel_strategy=strategy, **kwargs)

        assert not any("is only read when parallel_strategy" in record.getMessage() for record in caplog.records)

    def test_default_values_do_not_warn(self, caplog):
        """``tp_size=0`` means 'derive from world_size'; it is not lost intent."""
        import logging

        from llm.training.core.config import DistributedConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DistributedConfig(parallel_strategy="ddp", tp_size=0, pp_size=0)

        assert not any("is only read when parallel_strategy" in record.getMessage() for record in caplog.records)

    def test_3d_grid_is_validated_at_runtime_not_config_time(self):
        """The config cannot check the product (world_size is unknown then).

        The real guard lives in ``three_d_layout``, which raises when
        ``dp_size * pp_size * tp_size != world_size`` — pinned here so nobody
        moves that check into the config where it cannot see ``world_size``.
        """
        from llm.training.core.config import DistributedConfig
        from llm.training.distributed.parallel import three_d_layout

        # Accepted at config time even though the grid cannot tile a real world.
        DistributedConfig(parallel_strategy="3d", tp_size=2, pp_size=2, dp_size=2)

        with pytest.raises(ValueError, match="must tile the world exactly"):
            three_d_layout(world_size=4, dp_size=2, pp_size=2, tp_size=2)
