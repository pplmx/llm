"""Integration tests for Adapter wiring in the training stack.

Mirrors the IA³ / BitFit / Prefix Tuning trainer-integration tests:
the master switch lives on ``TrainingConfig``,
``LanguageModelingTask.build_model`` calls ``apply_adapter`` when the
flag is on, SFT inherits the behaviour without any extra wiring, and
the resulting model has the adapter path trainable while the base
Linear stays frozen.

Adapter has no scheduler / tracker and no inference-time merge (the
up projection is zero, so the wrapper contributes zero unless
trained). The test surface reflects that: no callback registration,
no merge roundtrip, just the structural / gradient-flow contract.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.adapter import (
    AdapterLinear,
    apply_adapter,
    get_adapter_parameters,
)
from llm.training.core.config import Config, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask


# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


def _tiny_config(*, use_adapter: bool = True, **adapter_kwargs) -> Config:
    """Build a config with a tiny model that fits the adapter shape."""
    cfg = Config()
    cfg.model.hidden_size = 16
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 32
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.use_adapter = use_adapter
    cfg.training.adapter_bottleneck_dim = adapter_kwargs.get(
        "adapter_bottleneck_dim", 4
    )
    cfg.training.adapter_target_modules = adapter_kwargs.get(
        "adapter_target_modules", None
    )
    cfg.training.epochs = 1
    return cfg


def _tiny_model_with_linear(cfg: Config) -> nn.Module:
    """Build a tiny model containing an ``nn.Linear`` for the wrapper.

    We bypass ``ModelFactory.from_config`` so the test does not depend
    on the full decoder plumbing — only on the contract
    ``apply_adapter`` actually consults (i.e. presence of Linear
    submodules).
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


class TestTrainingConfigAdapterFields:
    """Defaults and opt-in wiring via TrainingConfig."""

    def test_use_adapter_off_by_default(self):
        cfg = TrainingConfig()
        assert cfg.use_adapter is False

    def test_adapter_bottleneck_dim_default(self):
        """Defaults to 64 (Houlsby 2019 paper convention)."""
        cfg = TrainingConfig()
        assert cfg.adapter_bottleneck_dim == 64

    def test_adapter_target_modules_default_is_none(self):
        """``None`` → every ``nn.Linear`` is wrapped."""
        cfg = TrainingConfig()
        assert cfg.adapter_target_modules is None

    def test_bottleneck_dim_must_be_positive(self):
        with pytest.raises(ValueError):
            TrainingConfig(adapter_bottleneck_dim=0)
        with pytest.raises(ValueError):
            TrainingConfig(adapter_bottleneck_dim=-1)

    def test_opt_in_via_kwargs(self):
        cfg = TrainingConfig(
            use_adapter=True,
            adapter_bottleneck_dim=8,
            adapter_target_modules=["q_proj"],
        )
        assert cfg.use_adapter is True
        assert cfg.adapter_bottleneck_dim == 8
        assert cfg.adapter_target_modules == ["q_proj"]

    def test_adapter_coexists_with_all_other_peft_flags(self):
        """Five PEFT methods can coexist in the config — each wraps
        different module classes (Prefix Tuning wraps MultiHeadAttention,
        BitFit toggles requires_grad, the others wrap nn.Linear). In
        practice the user picks one, but the config must not implicitly
        forbid the combination.
        """
        cfg = TrainingConfig(
            use_adapter=True,
            use_bitfit=True,
            use_ia3=True,
            use_adalora=True,
            use_prefix_tuning=True,
        )
        assert cfg.use_adapter is True
        assert cfg.use_bitfit is True
        assert cfg.use_ia3 is True
        assert cfg.use_adalora is True
        assert cfg.use_prefix_tuning is True


# ---------------------------------------------------------------------------
# LanguageModelingTask wiring
# ---------------------------------------------------------------------------


class TestLanguageModelingTaskAppliesAdapter:
    """``build_model`` wires ``apply_adapter`` when ``use_adapter=True``."""

    def test_off_by_default_no_wrapping(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=False)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert linears, "test fixture should have at least one Linear"
        assert not any(isinstance(m, AdapterLinear) for m in model.modules()), (
            "AdapterLinear appeared without opt-in"
        )

    def test_opt_in_wraps_every_linear(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        assert len(wrappers) == 1, (
            f"expected exactly 1 AdapterLinear, got {len(wrappers)}"
        )

    def test_opt_in_passes_target_modules_through(self):
        """``adapter_target_modules`` is forwarded to ``apply_adapter``."""
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=True, adapter_target_modules=["fc"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        assert len(wrappers) == 1

    def test_target_modules_filters_non_matching_modules(self):
        """A substring that does NOT match the Linear module name leaves it alone."""
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=True, adapter_target_modules=["no_such_module"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        assert wrappers == []

    def test_bottleneck_dim_is_forwarded(self):
        """``adapter_bottleneck_dim`` flows into the wrapper's bottleneck size."""
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=True, adapter_bottleneck_dim=4)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrapper = next(m for m in model.modules() if isinstance(m, AdapterLinear))
        assert wrapper.bottleneck_dim == 4


# ---------------------------------------------------------------------------
# Gradient / parameter contract
# ---------------------------------------------------------------------------


class TestAdapterGradientContract:
    """After ``apply_adapter``, the base Linear is frozen and the adapter trains."""

    def test_base_linear_frozen_adapter_trainable(self):
        cfg = _tiny_config(use_adapter=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)
        apply_adapter(
            tiny_model,
            bottleneck_dim=cfg.training.adapter_bottleneck_dim,
            target_modules=cfg.training.adapter_target_modules,
        )

        wrapper = next(m for m in tiny_model.modules() if isinstance(m, AdapterLinear))
        assert wrapper.base_layer.weight.requires_grad is False
        if wrapper.base_layer.bias is not None:
            assert wrapper.base_layer.bias.requires_grad is False

        assert wrapper.down.weight.requires_grad is True
        assert wrapper.down.bias.requires_grad is True
        assert wrapper.up.weight.requires_grad is True
        assert wrapper.up.bias.requires_grad is True

    def test_optimizer_only_updates_adapter_params(self):
        """When the user wires the optimizer via
        ``get_adapter_parameters``, the base Linear weight is never
        touched.
        """
        cfg = _tiny_config(use_adapter=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)
        apply_adapter(tiny_model)

        wrapper = next(m for m in tiny_model.modules() if isinstance(m, AdapterLinear))

        base_weight_before = wrapper.base_layer.weight.detach().clone()

        opt = torch.optim.Adam(get_adapter_parameters(tiny_model), lr=1e-2)
        x = torch.randn(1, 4, cfg.model.hidden_size)
        out = tiny_model(x).sum()
        out.backward()
        opt.step()

        # Base weight unchanged (no grad → Adam is a no-op).
        assert torch.allclose(wrapper.base_layer.weight, base_weight_before, atol=1e-6)


# ---------------------------------------------------------------------------
# SFT / DPO inheritance
# ---------------------------------------------------------------------------


class TestSFTInheritsAdapter:
    """``SFTTask`` inherits the wrapping via
    ``LanguageModelingTask.build_model``.
    """

    def test_sft_build_model_wraps_with_adapter(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=True)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        assert len(wrappers) == 1

    def test_sft_off_by_default(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_adapter=False)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=tiny_model
        ):
            model = task.build_model()

        assert not any(isinstance(m, AdapterLinear) for m in model.modules())


class TestDPOInheritsAdapter:
    """``DPOTask`` builds policy AND reference model — both get adapter.

    DPO calls ``super().build_model()`` twice (once for the policy,
    once for the reference), so the adapter must wrap both —
    otherwise the reference logits would be computed on an unwrapped
    Linear and the policy / ref comparison would be apples-to-oranges.
    """

    def test_dpo_wraps_policy_and_reference(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_adapter=True)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_linear(cfg)
        reference = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        policy_wrappers = [
            m for m in built_policy.modules() if isinstance(m, AdapterLinear)
        ]
        assert len(policy_wrappers) == 1

        assert task.ref_model is not None
        assert task.ref_model is not built_policy
        ref_wrappers = [
            m for m in task.ref_model.modules() if isinstance(m, AdapterLinear)
        ]
        assert len(ref_wrappers) == 1

    def test_dpo_off_by_default(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_adapter=False)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_linear(cfg)
        reference = _tiny_model_with_linear(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        for m in built_policy.modules():
            assert not isinstance(m, AdapterLinear)
        for m in task.ref_model.modules():
            assert not isinstance(m, AdapterLinear)


# ---------------------------------------------------------------------------
# Empty / no-Linear models
# ---------------------------------------------------------------------------


class TestEmptyModelIsNoop:
    """Models with no ``nn.Linear`` modules leave ``apply_adapter`` a no-op."""

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
        cfg = _tiny_config(use_adapter=True)
        task = LanguageModelingTask(cfg, data_module=None)

        with patch(
            "llm.runtime.ModelFactory.from_config", return_value=no_linear
        ):
            model = task.build_model()

        # No AdapterLinear wrappers were produced (nothing to wrap).
        wrappers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        assert wrappers == []
