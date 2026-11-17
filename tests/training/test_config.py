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

    def test_single_worker_is_not_a_hardware_requirement(self):
        """``gpus_per_node=1`` must load on a host with NO GPUs at all.

        The field is a WORKER COUNT, not a hardware demand: the consumers
        (``DistributedManager.get_world_size`` and ``train.main``) read it as
        "how many local processes to spawn" and fall back to a single CPU
        process when no GPU is usable. The validator compared it against
        ``torch.cuda.device_count()`` unconditionally, so ``1 > 0`` was
        rejected — and every shipped CPU demo pins exactly ``1``, precisely to
        stop an idle multi-GPU host from fanning the tutorial out into one DDP
        rank per card. The result was that the documented demos could not be
        loaded on a CPU-only machine, which failed four e2e preset-config tests
        and both shipped-config validation tests on GPU-less CI.
        """
        with patch("torch.cuda.device_count", return_value=0):
            assert DistributedConfig(gpus_per_node=1).gpus_per_node == 1

    def test_multi_worker_still_requires_that_many_gpus(self):
        """The ``1`` exemption must not become a blanket exemption.

        Asking for more processes than the host has GPUs is a real
        configuration error and has to keep raising — otherwise the fix above
        would silently turn an unsatisfiable DDP launch into a single-process
        run.
        """
        with (
            patch("torch.cuda.device_count", return_value=0),
            pytest.raises(ValidationError, match="Requested 2 GPUs but only 0 available"),
        ):
            DistributedConfig(gpus_per_node=2)

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

        from tests.support.devices import cuda_device_count

        available_gpus = cuda_device_count()
        configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
        checked = 0
        skipped_for_gpus: list[str] = []
        for path in sorted(configs_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if {"model_path", "tokenizer_path", "host", "api_key"} & set(raw):
                continue  # serving schema, covered by ServingConfig
            # ``gpus_per_node <= 1`` is a worker count, not a hardware demand, so
            # it loads anywhere (the demo configs pin 1 precisely to stay
            # single-process). A config asking for MORE than one worker really
            # does describe a multi-GPU run, and ``DistributedConfig`` rightly
            # refuses it on a host that cannot provide it — that is the
            # validator working, not a broken YAML, so it is only exercised
            # where the hardware exists.
            requested_gpus = (raw.get("distributed") or {}).get("gpus_per_node")
            if isinstance(requested_gpus, int) and requested_gpus > 1 and requested_gpus > available_gpus:
                skipped_for_gpus.append(path.name)
                continue
            Config.from_yaml(path)
            checked += 1
        assert checked >= 10, f"expected to check the shipped training configs, only saw {checked}"
        assert len(skipped_for_gpus) < 10, (
            f"the GPU filter excused too many configs to be meaningful: {skipped_for_gpus}"
        )

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


class TestDedupKnobsNeedADedupSource:
    """Dedup knobs are only read by the ``dedup_*`` source factories."""

    @pytest.mark.parametrize(
        ("source", "knob", "value"),
        [
            ("local", "seen_hashes_path", "hashes.txt"),
            ("local", "write_seen_hashes", True),
            ("local", "hash_algo", "md5"),
            ("hf", "seen_hashes_path", "hashes.txt"),
        ],
    )
    def test_knob_without_a_dedup_source_warns(self, source, knob, value, caplog):
        """A plain source never wraps ``DedupTextSource``, so the knob is inert.

        Measured against the registry: ``local`` plus ``seen_hashes_path`` yields
        every row (3 duplicates stay 3), while ``dedup_local`` yields one. The
        old field comment claimed setting ``seen_hashes_path`` was itself enough
        to opt in — it is not, and two shipped configs relied on that.
        """
        import logging

        from llm.training.core.config import DataConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DataConfig(data_source=source, **{knob: value})

        messages = [record.getMessage() for record in caplog.records]
        assert any(knob in message and source in message for message in messages), (
            f"{knob} with data_source={source!r} was not reported: {messages}"
        )

    def test_dedup_source_is_silent(self, caplog):
        """The legitimate pairing must not warn."""
        import logging

        from llm.training.core.config import DataConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DataConfig(data_source="dedup_local", seen_hashes_path="hashes.txt", write_seen_hashes=True)

        assert not any("is not a dedup variant" in record.getMessage() for record in caplog.records)

    def test_explicit_null_does_not_warn(self, caplog):
        """``seen_hashes_path: null`` states the same thing as omission.

        ``configs/streaming_local_demo.yaml`` writes exactly that, annotated
        "not used with data_source=local" — a deliberate no-op, not a mistake.
        """
        import logging

        from llm.training.core.config import DataConfig

        with caplog.at_level(logging.WARNING, logger="llm.training.core.config"):
            DataConfig(data_source="local", seen_hashes_path=None)

        assert not any("is not a dedup variant" in record.getMessage() for record in caplog.records)

    def test_shipped_streaming_configs_actually_enable_dedup(self):
        """The streaming presets must use a ``dedup_*`` source, not plain ``hf``.

        They set ``seen_hashes_path`` + ``write_seen_hashes`` and their header
        claims that enables the wrapper. It did not: ``hf`` never wraps
        ``DedupTextSource``, so those presets silently skipped deduplication and
        never wrote the hash file.
        """
        import pathlib

        import yaml

        configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"

        # Discover the configs instead of listing them. The hard-coded pair
        # ("streaming_c4.yaml", "streaming_c4_demo.yaml") meant a NEW preset
        # could set dedup knobs beside a plain ``hf`` source and this test would
        # not look at it at all — verified by dropping such a file into
        # ``configs/``: the suite stayed green. The property being asserted is
        # about every shipped streaming preset, so the test has to enumerate
        # them the way a user would.
        candidates = sorted(configs_dir.glob("streaming*.yaml"))
        assert candidates, "no streaming configs found; the glob or the directory moved"

        checked = []
        for path in candidates:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data = raw.get("data") or {}
            knobs = [k for k in ("seen_hashes_path", "write_seen_hashes") if k in data]
            if not knobs:
                continue
            # An explicit ``null`` is the documented "not applicable here" form
            # (``streaming_local_demo.yaml`` writes it under data_source=local).
            # It must not be reported as inert configuration.
            if all(data[k] is None for k in knobs):
                continue
            checked.append(path.name)
            assert data.get("data_source", "").startswith("dedup_"), (
                f"{path.name} sets {knobs} but data_source={data.get('data_source')!r}, "
                "which ignores them (only dedup_* factories wrap the source)"
            )
        assert checked, "no streaming config with active dedup knobs was checked — the guard is vacuous"


class TestKvCacheCapabilityMessageStaysAccurate:
    """The "switch to an impl that supports KV cache" hint must reflect reality.

    ``ModelConfig`` rejects ``use_kv_cache=True`` on an impl whose capability
    record is ``False``, and the message used to end with a hard-coded
    ``(currently: mha)``. By then ``flash_attn`` and ``mla`` supported KV cache
    as well, so the one actionable sentence in the error steered the user to the
    least attractive of three working options — and the list re-staled every time
    an impl gained support, silently.

    Nothing asserted the text, which is why it survived. These tests derive the
    expectation from the same registry the check uses, so the assertion cannot
    drift away from the truth either.
    """

    @staticmethod
    def _non_kv_capable_impl(monkeypatch):
        """Register a throwaway impl whose capability record is False.

        ``replace`` rather than ``register`` so a previous test in the same
        process cannot make this one fail on a duplicate name; the registry
        entry and the capability record are both restored by ``monkeypatch`` so
        nothing leaks into other tests.
        """
        from llm.core.registry import ATTENTION_KV_CACHE_CAPABILITY, ATTENTION_REGISTRY

        name = "test_no_kv_cache_impl"
        monkeypatch.setattr(ATTENTION_REGISTRY, "_entries", dict(ATTENTION_REGISTRY._entries))
        ATTENTION_REGISTRY.replace(name, object)
        monkeypatch.setitem(ATTENTION_KV_CACHE_CAPABILITY, name, False)
        return name

    def test_message_lists_every_kv_capable_impl(self, monkeypatch):
        from llm.core.registry import ATTENTION_KV_CACHE_CAPABILITY
        from llm.training.core.config import ModelConfig

        name = self._non_kv_capable_impl(monkeypatch)
        expected = sorted(impl for impl, supports in ATTENTION_KV_CACHE_CAPABILITY.items() if supports)
        assert len(expected) >= 2, (
            "this test needs more than one KV-capable impl, otherwise the "
            "'(currently: ...)' list could be right by accident"
        )

        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(attn_impl=name, use_kv_cache=True)

        message = str(excinfo.value)
        for impl in expected:
            assert impl in message, f"KV-capable impl {impl!r} missing from the error message"
        assert message.count("currently:") == 1

    def test_message_does_not_hardcode_a_single_impl(self, monkeypatch):
        """Regression guard for the literal that was there: ``(currently: mha)``.

        Computed from the registry rather than comparing to a pasted string, so
        adding a KV-capable impl updates the expectation automatically — the
        failure mode is a stale message, and a hard-coded expectation would
        re-introduce the same staleness one level up.
        """
        from llm.core.registry import ATTENTION_KV_CACHE_CAPABILITY
        from llm.training.core.config import ModelConfig

        kv_capable = sorted(impl for impl, supports in ATTENTION_KV_CACHE_CAPABILITY.items() if supports)
        if kv_capable == ["mha"]:  # pragma: no cover - only if support regresses
            pytest.skip("only one KV-capable impl exists; nothing to disagree with")

        name = self._non_kv_capable_impl(monkeypatch)
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(attn_impl=name, use_kv_cache=True)

        assert "(currently: mha)." not in str(excinfo.value), (
            "the message still ends with the hard-coded '(currently: mha)' list"
        )


class TestShippedConfigsCanActuallyStart:
    """A config that PARSES is not a config that RUNS.

    `test_shipped_training_configs_all_load` only proves the YAML satisfies the
    schema. The four distributed presets (`ddp`, `zero`, `tp-pretrain`,
    `pp-pretrain`, `fsdp-pretrain`) each held nothing but a `distributed` block
    — 11 to 13 lines — while their own header and the distributed guide tell the
    reader to run `uv run llm train --task stream_lm --config-path <that file>`.
    Everything the run needs was missing: no `data.dataset_path`, so it died in
    `StreamingTextDataModule.setup` with "data.dataset_path is required when
    data_source='local'" — before the parallel strategy the file exists to
    demonstrate was ever exercised.

    The gap is structural, not a typo: schema validation cannot see "the
    runnable shape is absent", because every missing field has a default. These
    tests check the two things a reader needs to be true — the referenced corpus
    is reachable, and the text source the run builds from it really constructs.
    """

    @staticmethod
    def _shipped_configs():
        import pathlib

        import yaml

        configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
        for path in sorted(configs_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if {"model_path", "tokenizer_path", "host", "api_key"} & set(raw):
                continue  # serving schema, not a training run
            yield path, raw

    #: Configs whose `data` section is filled in AFTER loading, by design.
    #: `c4-pretrain.yaml` documents "load the YAML, then apply C4_PRESET on top";
    #: the preset writes the four HF dataset fields, so an empty `dataset_name`
    #: here is the documented contract, not a defect.
    _FILLED_BY_A_PRESET = frozenset({"c4-pretrain.yaml"})

    def test_every_runnable_shipped_config_names_a_text_source_it_can_resolve(self):
        """A config the docs tell you to RUN must name the inputs that run needs.

        This is the check that would have caught the four distributed presets,
        which shipped with nothing but a `distributed` block: they parsed, then
        died at `data.dataset_path is required when data_source='local'` — after
        the reader had already waited through model construction.

        It checks the STRUCTURE the run needs, not the filesystem. Most shipped
        configs deliberately point at corpora the reader creates (the header of
        each file shows the command), so requiring the file to exist would fail
        on every fresh clone — including on the repo's own CI. What must hold
        unconditionally is that a `local` source names a `dataset_path` and an
        `hf` source names a `dataset_name`; a config missing both is a skeleton,
        not an example.
        """
        from llm.training.core.config import Config

        preset_problems: list[str] = []
        checked = 0
        for path, raw in self._shipped_configs():
            data = raw.get("data")
            if not data:
                preset_problems.append(f"{path.name}: has no `data` section at all")
                continue
            # The schema itself already rejects an unparseable file; building
            # the Config proves the section survives the real validators.
            config = Config.from_yaml(path)
            source = config.data.data_source
            needs_path = source in {"local", "dedup_local"} and not config.data.dataset_path
            needs_name = (
                source in {"hf", "dedup_hf"}
                and not config.data.dataset_name
                and path.name not in self._FILLED_BY_A_PRESET
            )
            if needs_path or needs_name:
                field = "dataset_path" if needs_path else "dataset_name"
                preset_problems.append(f"{path.name}: data_source={source!r} needs `{field}`")
                continue
            checked += 1

        assert checked >= 10, f"only {checked} shipped configs declared usable inputs"
        assert not preset_problems, "shipped training configs that cannot start:\n  " + "\n  ".join(preset_problems)

    def test_distributed_presets_declare_a_runnable_shape(self):
        """The distributed presets must carry the sections their own usage block promises.

        Each preset's header says
        ``uv run llm train --task stream_lm --config-path configs/<name>.yaml``.
        That command needs a model, a training schedule and a data source; the
        files shipped with only a `distributed` block, so the documented command
        failed with "data.dataset_path is required" before reaching the
        parallelism the file exists to demonstrate.

        Other shipped configs (SFT / DPO / wikitext) legitimately expect a
        user-supplied corpus, so this is scoped to the presets whose header
        command the repo itself provides a corpus for.
        """
        import pathlib

        import yaml

        configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
        presets = [
            "ddp.yaml",
            "zero-pretrain.yaml",
            "tp-pretrain.yaml",
            "pp-pretrain.yaml",
            "fsdp-pretrain.yaml",
            "moe-demo.yaml",
        ]
        problems: list[str] = []
        for name in presets:
            path = configs_dir / name
            assert path.exists(), f"{name} is referenced by the docs but missing"
            text = path.read_text(encoding="utf-8")
            raw = yaml.safe_load(text) or {}
            # `moe-demo.yaml` is a single-process demo: it needs no
            # `distributed` block (the default DDP-of-one path runs it).
            required = ("model", "training", "data")
            if name != "moe-demo.yaml":
                required += ("distributed",)
            problems.extend(f"{name}: missing `{section}` section" for section in required if not raw.get(section))
            dataset_path = (raw.get("data") or {}).get("dataset_path")
            if not dataset_path:
                problems.append(f"{name}: `data.dataset_path` is required for data_source=local")
            elif dataset_path not in text:
                # The corpus the usage block tells the reader to create must be
                # the one the config reads, or the header command is wrong even
                # when every section is present.
                problems.append(f"{name}: usage block never names its own dataset_path ({dataset_path})")
            elif "print(" not in text:
                problems.append(f"{name}: uses {dataset_path} without showing how to create it")
        assert not problems, "distributed presets that cannot run as documented:\n  " + "\n  ".join(problems)


class TestDataWindowFitsTheModelContext:
    """The loader window must not exceed the model's context, checked at TRAIN time.

    ``data.max_seq_len`` and ``model.max_seq_len`` live in different YAML
    sections and both default to 512, so they agree unless exactly one is
    overridden. The mismatch used to surface from inside the forward pass as
    ``Sequence endpoint 2048 exceeds maximum sequence length 512`` from
    ``positional_encoding.py`` — the right numbers, but not the two config keys
    behind them. ``configs/c4-pretrain.yaml`` shipped in exactly that state.

    The check deliberately lives at the training entry point, NOT in
    ``Config``'s own validator: serving / generation / RLHF paths build a
    ``Config`` for a small model without ever constructing a dataloader, and a
    config-level rule rejected those perfectly valid setups (measured: it broke
    32 tests and 209 fixtures across ``serving``, ``generation`` and
    ``training/rlhf``).
    """

    def test_mismatch_is_rejected(self):
        from llm.models.config import ModelConfig
        from llm.training.core.config import Config, DataConfig
        from llm.training.train import _check_data_window_fits_model_context

        cfg = Config(model=ModelConfig(max_seq_len=32), data=DataConfig(max_seq_len=512))
        with pytest.raises(ValueError, match=r"data\.max_seq_len \(512\) exceeds model\.max_seq_len \(32\)"):
            _check_data_window_fits_model_context(cfg)

    def test_matching_lengths_pass(self):
        from llm.models.config import ModelConfig
        from llm.training.core.config import Config, DataConfig
        from llm.training.train import _check_data_window_fits_model_context

        _check_data_window_fits_model_context(
            Config(model=ModelConfig(max_seq_len=64), data=DataConfig(max_seq_len=64))
        )

    def test_a_shorter_data_window_is_allowed(self):
        """The data window may be SMALLER than the context — that is legitimate."""
        from llm.models.config import ModelConfig
        from llm.training.core.config import Config, DataConfig
        from llm.training.train import _check_data_window_fits_model_context

        _check_data_window_fits_model_context(
            Config(model=ModelConfig(max_seq_len=512), data=DataConfig(max_seq_len=128))
        )

    def test_config_alone_still_builds_for_inference_paths(self):
        """Counterpart to the placement decision: ``Config`` must not reject this.

        Serving and generation build a ``Config`` around a small model and never
        touch ``data.max_seq_len``; making this construct-time would break them.
        """
        from llm.models.config import ModelConfig
        from llm.training.core.config import Config, DataConfig

        cfg = Config(model=ModelConfig(max_seq_len=16), data=DataConfig(max_seq_len=512))
        assert cfg.model.max_seq_len == 16

    def test_c4_preset_pins_a_model_context_that_can_hold_its_sequences(self):
        """The shipped C4 preset must load AND pass the training-time check."""
        from llm.training.core.config import Config
        from llm.training.train import _check_data_window_fits_model_context

        repo_root = Path(__file__).resolve().parent.parent.parent
        cfg = Config.from_yaml(repo_root / "configs" / "c4-pretrain.yaml")
        assert cfg.model.max_seq_len >= cfg.data.max_seq_len
        _check_data_window_fits_model_context(cfg)


class TestNonPositiveCadenceKnobs:
    """The two "every N" knobs that were only validated at use time.

    Both fields are read as divisors/forwarded verbatim, and both accepted a
    zero that survived config validation, model construction and dataloader
    construction — the failure arrived much later, in a place that no longer
    names the config key:

    * ``logging.log_interval`` is the modulus in three call sites (``engine.py``
      twice, ``ppo_task.py`` once). ``0`` raised a bare
      ``ZeroDivisionError: division by zero`` on the first logged batch.
    * ``optimization.prefetch_factor`` is forwarded to ``DataLoader``. ``0`` was
      accepted by the constructor (torch only validates negatives there) and
      raised ``AssertionError: prefetch_factor must be greater than 0 for
      MultiProcessingDataLoaderIter`` on the FIRST iteration, i.e. only once
      training had started.

    Neither zero was ever a working setting: ``callbacks.py`` did guard its own
    ``% log_interval`` with ``> 0``, but no call site implemented "0 = off" for
    the engine, so the value crashed instead of disabling anything.
    """

    @pytest.mark.parametrize("bad", [0, -1, -25])
    def test_log_interval_must_be_positive(self, bad: int):
        from llm.training.core.config import LoggingConfig

        with pytest.raises(ValidationError, match="log_interval"):
            LoggingConfig(log_interval=bad)

    @pytest.mark.parametrize("good", [1, 10, 50])
    def test_log_interval_accepts_positive_values(self, good: int):
        """Counterpart: ordinary cadences still construct."""
        from llm.training.core.config import LoggingConfig

        assert LoggingConfig(log_interval=good).log_interval == good

    @pytest.mark.parametrize("bad", [0, -1, -8])
    def test_prefetch_factor_must_be_positive(self, bad: int):
        with pytest.raises(ValidationError, match="prefetch_factor"):
            OptimizationConfig(prefetch_factor=bad)

    @pytest.mark.parametrize("good", [1, 2, 4])
    def test_prefetch_factor_accepts_positive_values(self, good: int):
        assert OptimizationConfig(prefetch_factor=good).prefetch_factor == good

    def test_prefetch_factor_is_forwarded_to_the_loader_when_workers_run(self):
        """The guard is only worth having if the field still reaches torch.

        ``num_workers == 0`` deliberately drops the kwarg, so the existing
        zero-worker test cannot show the field is live. This builds the real
        ``DataLoader`` for the other branch and reads back what torch received —
        without iterating it, so no worker processes are spawned.
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

        config = Config(optimization=OptimizationConfig(num_workers=2, prefetch_factor=3))
        module = _Concrete(config)
        dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)

        loader = module.build_dataloader(dataset, sampler)

        assert loader.num_workers == 2
        assert loader.prefetch_factor == 3, (
            f"optimization.prefetch_factor must reach the DataLoader when workers run, got {loader.prefetch_factor}"
        )


class TestGradientClipVal:
    """``training.gradient_clip_val`` must not be negative, and 0 must mean "off".

    The field had no constraint and was passed straight to torch. Two values
    were silently wrong:

    * ``0`` zeroed every gradient (``clip_coef = 0 / (norm + eps) == 0``), so
      the optimizer step was an identity while ``global_step`` still advanced.
      Measured over 3 epochs on a real engine: ``||dW|| = 0.000000``.
    * a NEGATIVE value multiplied every gradient by that factor
      (``clip_coef = -1 / (norm + eps) < 0``), i.e. gradient ASCENT. Measured:
      the weight displacement of a ``gradient_clip_val=-1.0`` run had cosine
      ``-0.7644`` against the ``+1.0`` run — the step went the wrong way.

    ``0`` now resolves to "no clipping" (torch's ``max_norm=inf``), matching
    ``ppo.max_grad_norm``'s existing ``if max_grad_norm > 0`` convention, and a
    negative value is rejected at config load. ``nan`` is rejected by ``ge=0``
    as a side effect (a NaN comparison fails the bound); ``inf`` stays legal
    because it is exactly "no clipping" spelled out.
    """

    @pytest.mark.parametrize("bad", [-0.1, -1.0, -50.0])
    def test_negative_gradient_clip_val_is_rejected(self, bad: float):
        from llm.training.core.config import TrainingConfig

        with pytest.raises(ValidationError, match="gradient_clip_val"):
            TrainingConfig(gradient_clip_val=bad)

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0, 10.0, float("inf")])
    def test_non_negative_gradient_clip_val_is_accepted(self, good: float):
        """0 and inf both mean "do not clip", and ordinary values still work."""
        from llm.training.core.config import TrainingConfig

        assert TrainingConfig(gradient_clip_val=good).gradient_clip_val == good

    def test_zero_no_longer_zeroes_the_gradients(self):
        """The regression itself: ``clip_val=0`` must leave the step alone.

        Drives the real engine so the assertion covers the call site, not just
        torch's semantics. Compares the weight displacement of a 0-clipped run
        against the unclipped (``inf``) run — they must be identical, whereas a
        linearly-scaled gradient would move the weights differently.
        """
        zero = _weight_displacement(gradient_clip_val=0.0)
        unclipped = _weight_displacement(gradient_clip_val=float("inf"))
        clipped = _weight_displacement(gradient_clip_val=1e-3)

        assert zero > 0, "clip_val=0 must still train — it froze every weight before the fix"
        assert zero == pytest.approx(unclipped, rel=1e-6), (
            f"clip_val=0 must be identical to no clipping; got {zero} vs {unclipped} (a zero clip froze the weights)"
        )
        assert clipped < unclipped, (
            "sanity: a tiny positive clip must shrink the step, or this test cannot "
            f"distinguish clipping from no clipping (got {clipped} vs {unclipped})"
        )


def _weight_displacement(*, gradient_clip_val: float) -> float:
    """L2 norm of the total weight change after one epoch on a fixed batch.

    ``torch.cuda.is_available`` is patched off so the engine does not spend
    ~25 s probing free VRAM on a shared host; the clip logic is device-agnostic.
    """
    from unittest.mock import patch

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.config import Config, LoggingConfig, ModelConfig, OptimizationConfig, TrainingConfig
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask

    torch.manual_seed(11)
    config = Config(
        model=ModelConfig(vocab_size=8, hidden_size=32, num_layers=1, num_heads=2, max_seq_len=8),
        training=TrainingConfig(
            batch_size=4,
            num_samples=8,
            epochs=1,
            lr=0.01,
            run_validation=False,
            gradient_clip_val=gradient_clip_val,
            weight_decay=0.0,
            warmup_epochs=0,
        ),
        optimization=OptimizationConfig(num_workers=0, use_compile=False, use_amp=False),
        logging=LoggingConfig(log_interval=100, log_level="ERROR", save_logs=False),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    task = LanguageModelingTask(config, data_module)
    with patch("torch.cuda.is_available", return_value=False):
        engine = TrainingEngine(config, task, rank=0, world_size=1, data_module=data_module)
    # One fixed, fittable LM batch: the only variable between runs is the clip.
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]] * 4)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=4, shuffle=False)
    engine.val_dataloader = None

    before = [p.detach().clone() for p in engine.model.parameters()]
    engine._run_epoch(0)
    delta = torch.cat([(p.detach() - b).flatten() for p, b in zip(engine.model.parameters(), before, strict=True)])
    return delta.norm().item()


class TestNodeTopology:
    """``num_nodes`` / ``node_rank`` must describe a runnable topology.

    Both fields were unconstrained. Two silent failures followed:

    * ``num_nodes <= 0`` was coerced to 1 by ``DistributedManager.get_world_size``
      (``num_nodes if num_nodes > 0 else 1``), so a config claiming 0 nodes
      trained as a single node with nothing in the logs.
    * ``node_rank`` outside ``[0, num_nodes)`` produced out-of-range global
      ranks via ``node_rank * local_world_size + local_rank``. Measured with
      ``num_nodes=2, gpus_per_node=4``: ``node_rank=2`` gave global ranks
      ``[8..11]`` against ``world_size=8``, and ``node_rank=-1`` gave
      ``[-4..-1]``. ``init_process_group`` does not reject those eagerly — it
      blocks waiting for peers that were never meant to exist, so the symptom is
      a hang rather than a traceback.
    """

    @pytest.mark.parametrize("bad", [0, -1, -8])
    def test_num_nodes_must_be_positive(self, bad: int):
        from llm.training.core.config import DistributedConfig

        with pytest.raises(ValidationError, match="num_nodes"):
            DistributedConfig(num_nodes=bad)

    @pytest.mark.parametrize("good", [1, 2, 16])
    def test_num_nodes_accepts_positive_values(self, good: int):
        from llm.training.core.config import DistributedConfig

        assert DistributedConfig(num_nodes=good).num_nodes == good

    def test_negative_node_rank_is_rejected(self):
        from llm.training.core.config import DistributedConfig

        with pytest.raises(ValidationError, match="node_rank"):
            DistributedConfig(num_nodes=4, node_rank=-1)

    @pytest.mark.parametrize(("num_nodes", "node_rank"), [(2, 2), (1, 3), (4, 4), (4, 9)])
    def test_node_rank_past_the_last_node_is_rejected(self, num_nodes: int, node_rank: int):
        """The upper bound is ``num_nodes``, so it needs the cross-field check."""
        from llm.training.core.config import DistributedConfig

        with pytest.raises(ValidationError, match="outside the run"):
            DistributedConfig(num_nodes=num_nodes, node_rank=node_rank)

    @pytest.mark.parametrize(("num_nodes", "node_rank"), [(1, 0), (2, 0), (2, 1), (8, 7)])
    def test_in_range_topologies_still_load(self, num_nodes: int, node_rank: int):
        """Counterpart: every real multi-node topology must keep working."""
        from llm.training.core.config import DistributedConfig

        config = DistributedConfig(num_nodes=num_nodes, node_rank=node_rank)
        assert (config.num_nodes, config.node_rank) == (num_nodes, node_rank)

    def test_the_rejected_ranks_are_the_ones_that_used_to_hang(self):
        """Records the mechanism, so the constraint's reason stays checkable.

        A node_rank one past the last node is off the end of ``world_size`` for
        every one of its local workers — the precise condition that leaves a
        process group waiting for a peer schedule that includes nobody.
        """
        from llm.training.train import _global_rank

        local_world_size = 4
        world_size = 2 * local_world_size
        in_range = [_global_rank(1, local, local_world_size) for local in range(local_world_size)]
        out_of_range = [_global_rank(2, local, local_world_size) for local in range(local_world_size)]

        assert all(0 <= rank < world_size for rank in in_range), in_range
        assert not any(0 <= rank < world_size for rank in out_of_range), out_of_range


class TestGpusPerNodeLowerBound:
    """A negative worker count must be rejected; ``0`` stays the "derive" spelling.

    ``gpus_per_node`` goes through a ``mode="before"`` validator that already
    rejects asking for more workers than the host has GPUs. Any NEGATIVE value
    slipped past that check — ``requested > 1`` is simply False — and then:
    the field stored the negative verbatim, while
    ``DistributedManager.get_world_size`` collapsed it to 1 through its own
    ``if gpus_per_node and gpus_per_node > 0 else 1`` fallback. The run
    silently used ONE process regardless of what the YAML said, and the two
    layers disagreed about what the value meant.

    ``0`` is deliberately still accepted: it is the documented "derive from
    the host" spelling and takes the same fallback path, so pinning it prevents
    a blanket positivity rule from breaking that.
    """

    @pytest.mark.parametrize("bad", [-1, -8])
    def test_negative_worker_count_is_rejected(self, bad: int):
        from llm.training.core.config import DistributedConfig

        with pytest.raises(ValidationError, match="gpus_per_node must be >= 0"):
            DistributedConfig(gpus_per_node=bad)

    @pytest.mark.parametrize("good", [0, 1, 2])
    def test_zero_and_positive_still_load(self, good: int):
        """``0`` means "derive from the host" and must keep working."""
        from unittest.mock import patch

        from llm.training.core.config import DistributedConfig

        # Pin the CUDA count so the ``> available`` check is host-independent.
        with patch("llm.training.core.config.torch.cuda.device_count", return_value=8):
            assert DistributedConfig(gpus_per_node=good).gpus_per_node == good

    def test_none_derives_from_the_device_count(self):
        from unittest.mock import patch

        from llm.training.core.config import DistributedConfig

        with patch("llm.training.core.config.torch.cuda.device_count", return_value=5):
            assert DistributedConfig(gpus_per_node=None).gpus_per_node == 5


class TestWeightDecay:
    """``training.weight_decay`` must be non-negative and finite.

    The field was unconstrained and reaches ``torch.optim.AdamW`` through three
    tasks (``lm_task``, ``regression_task``, ``reward_task``). Two values got
    past the config and behaved differently, and neither named the field:

    * a NEGATIVE value made torch raise
      ``ValueError: Invalid weight_decay value: -0.1`` from inside the optimizer
      constructor, i.e. after the model was built;
    * ``inf`` is ACCEPTED by torch and turns every weight into ``-inf`` on the
      first step — measured on a two-parameter parameter with an all-zero
      gradient (pure decay): ``[1.0, 2.0]`` -> ``[-inf, -inf]``.

    ``training.lr`` already carries ``allow_inf_nan=False`` for the same reason,
    so these bounds match it.
    """

    @pytest.mark.parametrize("bad", [-0.1, -1.0])
    def test_negative_weight_decay_is_rejected(self, bad: float):
        from llm.training.core.config import TrainingConfig

        with pytest.raises(ValidationError, match="weight_decay"):
            TrainingConfig(weight_decay=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_weight_decay_is_rejected(self, bad: float):
        """``ge=0`` alone lets ``inf`` through (``inf >= 0`` is True)."""
        from llm.training.core.config import TrainingConfig

        with pytest.raises(ValidationError, match="weight_decay"):
            TrainingConfig(weight_decay=bad)

    @pytest.mark.parametrize("good", [0.0, 0.01, 0.1, 1.0])
    def test_zero_and_positive_weight_decay_is_accepted(self, good: float):
        """0 is the documented "no decay" setting and must stay legal."""
        from llm.training.core.config import TrainingConfig

        assert TrainingConfig(weight_decay=good).weight_decay == good

    def test_the_infinite_value_poisons_weights_in_one_step(self):
        """Record what ``allow_inf_nan=False`` prevents, using the real optimizer."""
        import torch

        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.zeros(2)  # pure decay: no data gradient at all
        torch.optim.AdamW([param], lr=0.1, weight_decay=float("inf")).step()
        assert param.detach().tolist() == [float("-inf"), float("-inf")], (
            "weight_decay=inf must poison the weights, which is why the config rejects it"
        )

        # The negative case is rejected by torch itself, just later and anonymously.
        other = torch.nn.Parameter(torch.zeros(1))
        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            torch.optim.AdamW([other], lr=1e-3, weight_decay=-0.1)
