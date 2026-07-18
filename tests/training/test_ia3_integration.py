"""Integration tests for IA³ wiring in the training stack.

Mirrors the Prefix Tuning trainer-integration tests: the master
switch lives on ``TrainingConfig``, ``LanguageModelingTask.build_model``
applies the wrapper when the flag is on, SFT inherits the behaviour
without any extra wiring, and the resulting model has the IA³ path
trainable while the base Linear stays frozen.

Unlike AdaLoRA there's no scheduler / tracker — IA³ is a one-shot
wrap at ``build_model`` time and the user calls ``merge_ia3`` at
inference time. The test surface reflects that: no callback
registration, no rank-shrinkage assertions, just the structural /
gradient-flow contract.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.ia3 import IA3Linear, apply_ia3, get_ia3_parameters
from llm.training.core.config import Config, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask


# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


def _tiny_config(*, use_ia3: bool = True, **ia3_kwargs) -> Config:
    """Build a config with a tiny model that fits the IA³ shape."""
    cfg = Config()
    cfg.model.hidden_size = 16
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 32
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.use_ia3 = use_ia3
    cfg.training.ia3_init_scale = ia3_kwargs.get("ia3_init_scale", 1.0)
    cfg.training.ia3_target_modules = ia3_kwargs.get("ia3_target_modules", None)
    cfg.training.epochs = 1
    return cfg


def _tiny_model_with_linear(cfg: Config) -> nn.Module:
    """Build a tiny model containing an ``nn.Linear`` for the wrapper.

    We bypass ``ModelFactory.from_config`` so the test does not depend
    on the full decoder plumbing — only on the contract ``apply_ia3``
    actually consults (i.e. presence of Linear submodules).
    """

    class _TinyLinearHolder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size)

        def forward(self, x):
            return self.fc(x)

    return _TinyLinearHolder()


# ---------------------------------------------------------------------------
# TrainingConfig structural tests
# ---------------------------------------------------------------------------


class TestTrainingConfigIA3Fields:
    """Defaults and opt-in wiring via TrainingConfig."""

    def test_use_ia3_off_by_default(self):
        cfg = TrainingConfig()
        assert cfg.use_ia3 is False

    def test_ia3_init_scale_default_is_one(self):
        """``init_scale=1.0`` → wrapper is the identity transform at step 1."""
        cfg = TrainingConfig()
        assert cfg.ia3_init_scale == 1.0

    def test_ia3_target_modules_default_is_none(self):
        """``None`` → every ``nn.Linear`` is wrapped."""
        cfg = TrainingConfig()
        assert cfg.ia3_target_modules is None

    def test_init_scale_must_be_positive(self):
        with pytest.raises(ValueError):
            TrainingConfig(ia3_init_scale=0.0)
        with pytest.raises(ValueError):
            TrainingConfig(ia3_init_scale=-1.0)

    def test_opt_in_via_kwargs(self):
        cfg = TrainingConfig(
            use_ia3=True,
            ia3_init_scale=0.5,
            ia3_target_modules=["q_proj"],
        )
        assert cfg.use_ia3 is True
        assert cfg.ia3_init_scale == 0.5
        assert cfg.ia3_target_modules == ["q_proj"]

    def test_ia3_coexists_with_adalora_and_prefix_tuning(self):
        """Three PEFT methods can coexist in the config — they wrap
        different module classes (AdaLoRA wraps ``nn.Linear``, IA³
        wraps ``nn.Linear``, Prefix Tuning wraps ``MultiHeadAttention``).
        At most one of AdaLoRA / IA³ should be active in practice
        (both wrap ``nn.Linear``), but the config must not implicitly
        forbid the combination — the user picks.
        """
        cfg = TrainingConfig(
            use_ia3=True,
            use_adalora=True,
            use_prefix_tuning=True,
        )
        assert cfg.use_ia3 is True
        assert cfg.use_adalora is True
        assert cfg.use_prefix_tuning is True


# ---------------------------------------------------------------------------
# LanguageModelingTask wiring
# ---------------------------------------------------------------------------


class TestLanguageModelingTaskAppliesIA3:
    """``build_model`` wires ``apply_ia3`` when ``use_ia3=True``."""

    def test_off_by_default_no_wrapping(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=False)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert linears, "test fixture should have at least one Linear"
        assert not any(isinstance(m, IA3Linear) for m in model.modules()), (
            "IA3Linear appeared without opt-in"
        )

    def test_opt_in_wraps_every_linear(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, IA3Linear)]
        assert len(wrappers) == 1, (
            f"expected exactly 1 IA3Linear, got {len(wrappers)}"
        )

    def test_opt_in_passes_target_modules_through(self):
        """``ia3_target_modules`` is forwarded to ``apply_ia3``."""
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=True, ia3_target_modules=["fc"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, IA3Linear)]
        assert len(wrappers) == 1

    def test_target_modules_filters_non_matching_modules(self):
        """A substring that does NOT match the Linear module name leaves it alone."""
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=True, ia3_target_modules=["no_such_module"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, IA3Linear)]
        assert wrappers == []

    def test_init_scale_is_forwarded(self):
        """``ia3_init_scale`` flows into the wrapper's ``ia3_l`` initial value."""
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=True, ia3_init_scale=0.5)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrapper = next(m for m in model.modules() if isinstance(m, IA3Linear))
        assert torch.allclose(wrapper.ia3_l, torch.full_like(wrapper.ia3_l, 0.5))


# ---------------------------------------------------------------------------
# Gradient / parameter contract
# ---------------------------------------------------------------------------


class TestIA3GradientContract:
    """After ``apply_ia3``, the base Linear is frozen and ``ia3_l`` is trainable."""

    def test_base_linear_frozen_ia3_l_trainable(self):
        cfg = _tiny_config(use_ia3=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)
        apply_ia3(
            tiny_model,
            init_scale=cfg.training.ia3_init_scale,
            target_modules=cfg.training.ia3_target_modules,
        )

        wrapper = next(m for m in tiny_model.modules() if isinstance(m, IA3Linear))
        assert wrapper.base_layer.weight.requires_grad is False
        if wrapper.base_layer.bias is not None:
            assert wrapper.base_layer.bias.requires_grad is False

        assert wrapper.ia3_l.requires_grad is True

    def test_optimizer_only_updates_ia3_l(self):
        """When the user wires the optimizer via ``get_ia3_parameters``,
        the base Linear weight is never touched.
        """
        cfg = _tiny_config(use_ia3=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)
        apply_ia3(tiny_model)

        wrapper = next(m for m in tiny_model.modules() if isinstance(m, IA3Linear))

        base_weight_before = wrapper.base_layer.weight.detach().clone()
        ia3_l_before = wrapper.ia3_l.detach().clone()

        opt = torch.optim.Adam(get_ia3_parameters(tiny_model), lr=1e-2)

        x = torch.randn(1, 4, cfg.model.hidden_size)
        out = tiny_model(x)
        loss = out.sum()
        loss.backward()
        opt.step()

        # Base weight unchanged (no grad → Adam is a no-op).
        assert torch.allclose(wrapper.base_layer.weight, base_weight_before)
        # ia3_l updated.
        assert not torch.allclose(wrapper.ia3_l, ia3_l_before)


# ---------------------------------------------------------------------------
# SFT / DPO inheritance
# ---------------------------------------------------------------------------


class TestSFTInheritsIA3:
    """``SFTTask`` inherits the wrapping via ``LanguageModelingTask.build_model``."""

    def test_sft_build_model_wraps_with_ia3(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=True)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, IA3Linear)]
        assert len(wrappers) == 1

    def test_sft_off_by_default(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_ia3=False)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        assert not any(isinstance(m, IA3Linear) for m in model.modules())


class TestDPOInheritsIA3:
    """``DPOTask`` builds policy AND reference model — both get IA³.

    DPO calls ``super().build_model()`` twice (once for the policy,
    once for the reference), so IA³ must wrap both — otherwise the
    reference logits would be computed on an unwrapped Linear and the
    policy / ref comparison would be apples-to-oranges.
    """

    def test_dpo_wraps_policy_and_reference(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_ia3=True)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_linear(cfg)
        reference = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        policy_wrappers = [
            m for m in built_policy.modules() if isinstance(m, IA3Linear)
        ]
        assert len(policy_wrappers) == 1

        assert task.ref_model is not None
        assert task.ref_model is not built_policy
        ref_wrappers = [
            m for m in task.ref_model.modules() if isinstance(m, IA3Linear)
        ]
        assert len(ref_wrappers) == 1

    def test_dpo_off_by_default(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_ia3=False)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_linear(cfg)
        reference = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        for m in built_policy.modules():
            assert not isinstance(m, IA3Linear)
        for m in task.ref_model.modules():
            assert not isinstance(m, IA3Linear)


# ---------------------------------------------------------------------------
# Empty / no-Linear models
# ---------------------------------------------------------------------------


class TestEmptyModelIsNoop:
    """Models with no ``nn.Linear`` modules leave ``apply_ia3`` a no-op."""

    def test_linear_only_model_no_wrap(self):
        """A model with no Linear at all must not break ``build_model``."""
        from unittest.mock import patch

        # Build a model with only LayerNorm + activation — no Linear.
        class _NoLinearModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(self.norm(x))

        no_linear = _NoLinearModel()
        cfg = _tiny_config(use_ia3=True)
        task = LanguageModelingTask(cfg, data_module=None)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=no_linear
        ):
            model = task.build_model()

        # No IA³ wrappers were produced (nothing to wrap).
        wrappers = [m for m in model.modules() if isinstance(m, IA3Linear)]
        assert wrappers == []

