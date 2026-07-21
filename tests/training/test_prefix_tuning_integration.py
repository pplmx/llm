"""Integration tests for Prefix Tuning wiring in the training stack.

Mirrors the AdaLoRA trainer-integration slice (T3 #42): the master
switch lives on ``TrainingConfig``, ``LanguageModelingTask.build_model``
applies the wrapper when the flag is on, SFT/DPO tasks inherit the
behaviour without any extra wiring, and the resulting model has the
prefix path trainable while the base MHA stays frozen.

Unlike AdaLoRA there's no scheduler / tracker — Prefix Tuning is a
one-shot wrap at ``build_model`` time and the user calls
``fold_reparameterization`` at inference time. The test surface
reflects that: no callback registration, no rank-shrinkage assertions,
just the structural / gradient-flow contract.
"""

from __future__ import annotations

import torch

from llm.core.attn.mha import MultiHeadAttention
from llm.core.prefix_tuning import (
    PrefixTuningAttention,
    apply_prefix_tuning,
    get_prefix_parameters,
)
from llm.training.core.config import Config, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask

# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


def _tiny_config(*, use_prefix_tuning: bool = True, **prefix_kwargs) -> Config:
    """Build a config with a tiny model that fits the prefix tuning shape."""
    cfg = Config()
    cfg.model.hidden_size = 16
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 32
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.use_prefix_tuning = use_prefix_tuning
    cfg.training.prefix_tuning_len = prefix_kwargs.get("prefix_tuning_len", 4)
    cfg.training.prefix_reparam_hidden = prefix_kwargs.get("prefix_reparam_hidden", 8)
    cfg.training.prefix_target_modules = prefix_kwargs.get("prefix_target_modules")
    cfg.training.epochs = 1
    return cfg


def _tiny_model_with_mha(cfg: Config) -> torch.nn.Module:  # type: ignore[name-defined]
    """Build a tiny model containing a ``MultiHeadAttention`` for the wrapper.

    We bypass ``ModelFactory.from_config`` so the test does not depend on
    the full decoder plumbing — only on the contract
    ``apply_prefix_tuning`` actually consults (i.e. presence of MHA
    submodules).
    """

    class TinyModelWithMHA:
        pass

    import torch.nn as nn

    class _TinyMHAHolder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = MultiHeadAttention(
                hidden_size=cfg.model.hidden_size,
                num_heads=cfg.model.num_heads,
                include_norm_residual=False,
            )

        def forward(self, x):
            return self.attn(x)

    return _TinyMHAHolder()


# ---------------------------------------------------------------------------
# TrainingConfig structural tests (already covered by test_prefix_tuning_config)
# These tests focus on the build_model wiring.
# ---------------------------------------------------------------------------


class TestTrainingConfigPrefixTuningFields:
    """Defaults and opt-in wiring via TrainingConfig."""

    def test_use_prefix_tuning_off_by_default(self):
        cfg = TrainingConfig()
        assert cfg.use_prefix_tuning is False

    def test_opt_in_via_kwargs(self):
        cfg = TrainingConfig(
            use_prefix_tuning=True,
            prefix_tuning_len=20,
            prefix_reparam_hidden=64,
        )
        assert cfg.use_prefix_tuning is True
        assert cfg.prefix_tuning_len == 20
        assert cfg.prefix_reparam_hidden == 64


# ---------------------------------------------------------------------------
# LanguageModelingTask wiring
# ---------------------------------------------------------------------------


class TestLanguageModelingTaskAppliesPrefixTuning:
    """``build_model`` wires ``apply_prefix_tuning`` when ``use_prefix_tuning=True``."""

    def test_off_by_default_no_wrapping(self):
        """When ``use_prefix_tuning=False`` (default), build_model does NOT wrap."""
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=False)
        task = LanguageModelingTask(cfg, data_module=None)

        # Bypass the factory; we only care whether build_model calls
        # apply_prefix_tuning when off.
        tiny_model = _tiny_model_with_mha(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # The MHA layer stays unwrapped.
        mha_modules = [m for m in model.modules() if isinstance(m, MultiHeadAttention)]
        assert mha_modules, "test fixture should have at least one MHA"
        assert not any(isinstance(m, PrefixTuningAttention) for m in model.modules()), (
            "PrefixTuningAttention appeared without opt-in"
        )

    def test_opt_in_wraps_every_mha(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=True)
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_mha(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # The MHA layer was replaced by a PrefixTuningAttention.
        wrappers = [m for m in model.modules() if isinstance(m, PrefixTuningAttention)]
        assert len(wrappers) == 1, f"expected exactly 1 PrefixTuningAttention, got {len(wrappers)}"

    def test_opt_in_passes_target_modules_through(self):
        """``prefix_target_modules`` is forwarded to ``apply_prefix_tuning``."""
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=True, prefix_target_modules=["attn"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_mha(cfg)

        # ``prefix_target_modules=["attn"]`` matches the module name
        # ``attn`` (substring), so the wrap should fire.
        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, PrefixTuningAttention)]
        assert len(wrappers) == 1

    def test_target_modules_filters_non_matching_modules(self):
        """A substring that does NOT match the MHA module name leaves it alone."""
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=True, prefix_target_modules=["no_such_module"])
        task = LanguageModelingTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_mha(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        # No wrap fired because no module name matched the pattern.
        wrappers = [m for m in model.modules() if isinstance(m, PrefixTuningAttention)]
        assert wrappers == []


# ---------------------------------------------------------------------------
# Gradient / parameter contract
# ---------------------------------------------------------------------------


class TestPrefixTuningGradientContract:
    """After ``apply_prefix_tuning``, the base MHA is frozen and the prefix path trains."""

    def test_base_mha_frozen_prefix_params_trainable(self):
        cfg = _tiny_config(use_prefix_tuning=True)
        tiny_model = _tiny_model_with_mha(cfg)
        apply_prefix_tuning(
            tiny_model,
            prefix_len=cfg.training.prefix_tuning_len,
            reparam_hidden=cfg.training.prefix_reparam_hidden,
        )

        # Base MHA weights are frozen.
        wrapper = next(m for m in tiny_model.modules() if isinstance(m, PrefixTuningAttention))
        assert wrapper.base_attn.qkv_proj.weight.requires_grad is False
        assert wrapper.base_attn.out_proj.weight.requires_grad is False

        # All 5 prefix params are trainable.
        prefix_params = list(get_prefix_parameters(tiny_model))
        assert len(prefix_params) == 5
        for p in prefix_params:
            assert p.requires_grad is True

    def test_optimizer_only_updates_prefix_params(self):
        """When the user wires the optimizer via ``get_prefix_parameters``, the
        base MHA weights are never touched."""
        cfg = _tiny_config(use_prefix_tuning=True)
        tiny_model = _tiny_model_with_mha(cfg)
        apply_prefix_tuning(
            tiny_model,
            prefix_len=cfg.training.prefix_tuning_len,
            reparam_hidden=cfg.training.prefix_reparam_hidden,
        )

        wrapper = next(m for m in tiny_model.modules() if isinstance(m, PrefixTuningAttention))
        import torch

        # Snapshot base MHA weights before any optimizer step.
        base_qkv_before = wrapper.base_attn.qkv_proj.weight.detach().clone()
        prefix_small_before = wrapper.prefix_small.detach().clone()

        # Build an optimizer from prefix-only params (this is the
        # documented user pattern).
        opt = torch.optim.Adam(get_prefix_parameters(tiny_model), lr=1e-2)

        # Forward + backward on a fake loss.
        x = torch.randn(1, 4, cfg.model.hidden_size)
        out = tiny_model(x)
        loss = out.sum()
        loss.backward()
        opt.step()

        # Base MHA weights unchanged (no grad → Adam is a no-op).
        assert torch.allclose(wrapper.base_attn.qkv_proj.weight, base_qkv_before)
        # Prefix params updated.
        assert not torch.allclose(wrapper.prefix_small, prefix_small_before)


# ---------------------------------------------------------------------------
# SFT / DPO inheritance
# ---------------------------------------------------------------------------


class TestSFTInheritsPrefixTuning:
    """``SFTTask`` inherits the wrapping via ``LanguageModelingTask.build_model``."""

    def test_sft_build_model_wraps_with_prefix_tuning(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=True)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_mha(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        wrappers = [m for m in model.modules() if isinstance(m, PrefixTuningAttention)]
        assert len(wrappers) == 1

    def test_sft_off_by_default(self):
        from unittest.mock import patch

        cfg = _tiny_config(use_prefix_tuning=False)
        task = SFTTask(cfg, data_module=None)
        tiny_model = _tiny_model_with_mha(cfg)

        with patch("llm.runtime.ModelFactory.from_config", return_value=tiny_model):
            model = task.build_model()

        assert not any(isinstance(m, PrefixTuningAttention) for m in model.modules())


class TestDPOInheritsPrefixTuning:
    """``DPOTask`` builds policy AND reference model — both get prefix tuning.

    DPO calls ``super().build_model()`` twice (once for the policy, once
    for the reference), so prefix tuning must wrap both — otherwise the
    reference logits would be computed on an unwrapped MHA and the
    policy / ref comparison would be apples-to-oranges.
    """

    def test_dpo_wraps_policy_and_reference(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_prefix_tuning=True)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_mha(cfg)
        reference = _tiny_model_with_mha(cfg)

        # DPO calls build_model() twice → factory returns two distinct models.
        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        # Policy is wrapped.
        policy_wrappers = [m for m in built_policy.modules() if isinstance(m, PrefixTuningAttention)]
        assert len(policy_wrappers) == 1

        # Reference is wrapped (and distinct from policy).
        assert task.ref_model is not None
        assert task.ref_model is not built_policy
        ref_wrappers = [m for m in task.ref_model.modules() if isinstance(m, PrefixTuningAttention)]
        assert len(ref_wrappers) == 1

    def test_dpo_off_by_default(self):
        from unittest.mock import patch

        from llm.training.tasks.dpo_task import DPOTask

        cfg = _tiny_config(use_prefix_tuning=False)
        task = DPOTask(cfg, data_module=None)
        policy = _tiny_model_with_mha(cfg)
        reference = _tiny_model_with_mha(cfg)

        with patch(
            "llm.runtime.ModelFactory.from_config",
            side_effect=[policy, reference],
        ):
            built_policy = task.build_model()

        # Neither side was wrapped.
        for m in built_policy.modules():
            assert not isinstance(m, PrefixTuningAttention)
        for m in task.ref_model.modules():
            assert not isinstance(m, PrefixTuningAttention)


# ---------------------------------------------------------------------------
# Empty / no-MHA models
# ---------------------------------------------------------------------------


class TestEmptyModelIsNoop:
    """Models with no ``MultiHeadAttention`` modules leave ``apply_prefix_tuning`` a no-op."""

    def test_linear_only_model_no_wrap(self):
        """A model with no MHA at all must not break ``build_model``."""
        from unittest.mock import patch

        import torch.nn as nn

        linear_only = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))

        cfg = _tiny_config(use_prefix_tuning=True)
        task = LanguageModelingTask(cfg, data_module=None)

        with patch("llm.runtime.ModelFactory.from_config", return_value=linear_only):
            model = task.build_model()

        # No prefix wrappers were produced (nothing to wrap).
        wrappers = [m for m in model.modules() if isinstance(m, PrefixTuningAttention)]
        assert wrappers == []
        # The Linear layers are untouched.
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2
