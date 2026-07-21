"""Integration tests for BitFit wiring in the training stack.

Mirrors the IA³ / Prefix Tuning trainer-integration tests: the
master switch lives on ``TrainingConfig``,
``LanguageModelingTask.build_model`` calls ``apply_bitfit`` when the
flag is on, SFT inherits the behaviour without any extra wiring,
and the resulting model has every bias trainable while every weight
is frozen.

BitFit has no scheduler / tracker and no inference-time merge
(unlike IA³ which has ``merge_ia3``, Prefix Tuning which has
``fold_reparameterization``). The test surface reflects that: no
callback registration, no merge roundtrip, just the structural /
gradient-flow contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.core.bitfit import apply_bitfit, get_bitfit_parameters
from llm.training.core.config import Config, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask

# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


def _tiny_config(*, use_bitfit: bool = True, **bitfit_kwargs) -> Config:
    """Build a config with a tiny model that fits the BitFit shape."""
    cfg = Config()
    cfg.model.hidden_size = 16
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 32
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.use_bitfit = use_bitfit
    cfg.training.bitfit_target_modules = bitfit_kwargs.get("bitfit_target_modules")
    cfg.training.epochs = 1
    return cfg


def _tiny_model_with_biases(cfg: Config) -> nn.Module:
    """Build a tiny model containing biases for the BitFit toggle.

    Two Linears with bias=True and one LayerNorm — gives us three
    ``.bias``-suffixed params (two Linear biases + LayerNorm bias) plus
    one LayerNorm ``.weight`` param (NOT a bias, so BitFit freezes it).
    """

    class _TinyBiasHolder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)
            self.fc2 = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size, bias=True)
            self.norm = nn.LayerNorm(cfg.model.hidden_size)

        def forward(self, x):  # pragma: no cover — not used in tests
            return self.norm(self.fc2(self.fc1(x)))

    return _TinyBiasHolder()


# ---------------------------------------------------------------------------
# TrainingConfig structural tests
# ---------------------------------------------------------------------------


class TestTrainingConfigBitFitFields:
    """Defaults and opt-in wiring via TrainingConfig."""

    def test_use_bitfit_off_by_default(self):
        cfg = TrainingConfig()
        assert cfg.use_bitfit is False

    def test_bitfit_target_modules_default_is_none(self):
        """``None`` → every ``.bias`` is enabled (BitFit default)."""
        cfg = TrainingConfig()
        assert cfg.bitfit_target_modules is None

    def test_opt_in_via_kwargs(self):
        cfg = TrainingConfig(
            use_bitfit=True,
            bitfit_target_modules=["fc1"],
        )
        assert cfg.use_bitfit is True
        assert cfg.bitfit_target_modules == ["fc1"]

    def test_bitfit_coexists_with_adalora_ia3_and_prefix_tuning(self):
        """Four PEFT methods can coexist in the config. In practice the
        user picks one, but the config must not implicitly forbid the
        combination — selecting the right method is a user decision.
        """
        cfg = TrainingConfig(
            use_bitfit=True,
            use_ia3=True,
            use_adalora=True,
            use_prefix_tuning=True,
        )
        assert cfg.use_bitfit is True
        assert cfg.use_ia3 is True
        assert cfg.use_adalora is True
        assert cfg.use_prefix_tuning is True


# ---------------------------------------------------------------------------
# LanguageModelingTask wiring
# ---------------------------------------------------------------------------


class TestLanguageModelingTaskAppliesBitFit:
    """``build_model`` wires ``apply_bitfit`` when ``use_bitfit=True``."""

    def test_off_by_default_no_freezing(self):
        """When ``use_bitfit=False`` (default), the model is returned
        with all parameters in their default state (everything
        trainable). No BitFit snapshot is saved.
        """
        from unittest.mock import patch

        from llm.core.bitfit import is_bitfit_applied

        cfg = _tiny_config(use_bitfit=False)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # BitFit was NOT applied.
        assert is_bitfit_applied(model) is False
        # Every weight is still trainable (default state preserved).
        for p in model.parameters():
            assert p.requires_grad is True

    def test_opt_in_freezes_weights_enables_biases(self):
        from unittest.mock import patch

        from llm.core.bitfit import is_bitfit_applied

        cfg = _tiny_config(use_bitfit=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # BitFit was applied.
        assert is_bitfit_applied(model) is True
        # Every weight is frozen.
        for name, p in model.named_parameters():
            if name.endswith(".weight"):
                assert p.requires_grad is False
            if name.endswith(".bias") or name == "bias":
                assert p.requires_grad is True

    def test_opt_in_passes_target_modules_through(self):
        """``bitfit_target_modules`` is forwarded to ``apply_bitfit``."""
        from unittest.mock import patch

        cfg = _tiny_config(use_bitfit=True, bitfit_target_modules=["fc1"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # fc1.bias matches → trainable.
        assert model.fc1.bias.requires_grad is True
        # fc2.bias and norm.bias do NOT match → frozen.
        assert model.fc2.bias.requires_grad is False
        assert model.norm.bias.requires_grad is False


# ---------------------------------------------------------------------------
# Gradient / parameter contract
# ---------------------------------------------------------------------------


class TestBitFitGradientContract:
    """After ``apply_bitfit``, every weight is frozen and every bias trains."""

    def test_optimizer_only_updates_biases(self):
        """When the user wires the optimizer via
        ``get_bitfit_parameters``, every weight is byte-identical across
        an Adam step.
        """
        cfg = _tiny_config(use_bitfit=True)
        tiny_model = _tiny_model_with_biases(cfg)
        apply_bitfit(tiny_model)

        weight_before = tiny_model.fc1.weight.detach().clone()
        bias_before = tiny_model.fc1.bias.detach().clone()

        opt = torch.optim.Adam(get_bitfit_parameters(tiny_model), lr=1e-2)

        x = torch.randn(1, 4, cfg.model.hidden_size)
        out = tiny_model(x).sum()
        out.backward()
        opt.step()

        # Weight unchanged (no grad → Adam is a no-op).
        assert torch.allclose(tiny_model.fc1.weight, weight_before, atol=1e-6)
        # Bias updated.
        assert not torch.allclose(tiny_model.fc1.bias, bias_before, atol=1e-6)

    def test_helper_yields_only_biases_after_build_model(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_bitfit=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        params = list(get_bitfit_parameters(model))
        # Three biases: fc1.bias, fc2.bias, norm.bias.
        assert len(params) == 3
        for p in params:
            assert p.requires_grad is True


# ---------------------------------------------------------------------------
# SFT / DPO inheritance
# ---------------------------------------------------------------------------


class TestSFTInheritsBitFit:
    """``SFTTask`` inherits the wiring via
    ``LanguageModelingTask.build_model``.
    """

    def test_sft_build_model_applies_bitfit(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_bitfit=True)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # Every bias is trainable, every weight is frozen.
        for name, p in model.named_parameters():
            if name.endswith(".weight"):
                assert p.requires_grad is False
            if name.endswith(".bias") or name == "bias":
                assert p.requires_grad is True

    def test_sft_off_by_default(self):
        from unittest.mock import patch

        from llm.core.bitfit import is_bitfit_applied

        cfg = _tiny_config(use_bitfit=False)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_biases(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # BitFit was NOT applied.
        assert is_bitfit_applied(model) is False


class TestDPOInheritsBitFit:
    """``DPOTask`` builds policy AND reference model — both get BitFit.

    DPO calls ``super().build_model()`` twice (once for the policy,
    once for the reference), so BitFit must run on both — otherwise
    the reference model's biases would not match the policy's.
    """

    def test_dpo_applies_bitfit_to_policy_and_reference(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_bitfit=True)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_biases(cfg)
        reference = _tiny_model_with_biases(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        # Policy has BitFit applied.
        for name, p in built_policy.named_parameters():
            if name.endswith(".weight"):
                assert p.requires_grad is False

        assert task.ref_model is not None
        assert task.ref_model is not built_policy
        # Reference has BitFit applied too.
        for name, p in task.ref_model.named_parameters():
            if name.endswith(".weight"):
                assert p.requires_grad is False

    def test_dpo_off_by_default(self):
        from unittest.mock import patch

        from llm.core.bitfit import is_bitfit_applied
        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_bitfit=False)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_biases(cfg)
        reference = _tiny_model_with_biases(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        # No BitFit applied to either side. (DPO then freezes the
        # reference model entirely, which is its standard policy —
        # we only check that BitFit was NOT one of the steps that
        # touched it.)
        assert is_bitfit_applied(built_policy) is False
        assert is_bitfit_applied(task.ref_model) is False
        # Policy biases are still trainable (no BitFit freezing).
        assert built_policy.fc1.bias.requires_grad is True
        assert built_policy.norm.bias.requires_grad is True


# ---------------------------------------------------------------------------
# Empty / bias-free models
# ---------------------------------------------------------------------------


class TestEmptyModelIsNoop:
    """Models with no biases leave ``apply_bitfit`` a no-op."""

    def test_bias_free_model_no_trainable(self):
        """A model with no biases at all must not break ``build_model`` —
        it just produces a fully-frozen model (zero trainable params).
        """
        from unittest.mock import patch

        class _BiasFree(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 16, bias=False)

        no_bias = _BiasFree()
        cfg = _tiny_config(use_bitfit=True)
        task = LanguageModelingTask(cfg, data_module=None)

        with patch("llm.runtime.ModelFactory.from_config", return_value=no_bias):
            model = task.build_model()

        # Every parameter is frozen (no biases to enable).
        for p in model.parameters():
            assert p.requires_grad is False
