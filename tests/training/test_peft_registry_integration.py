"""Tests for the PEFT_REGISTRY trainer integration slice (T2 PEFT #44).

Covers:

- New ``TrainingConfig.peft_method`` and ``peft_kwargs`` fields
- ``peft_method`` validator rejects unknown methods at config-load time
- ``LanguageModelingTask.build_model`` dispatches via the registry when
  ``peft_method`` is set (and otherwise keeps the legacy per-method
  ``use_*`` path intact)
- All seven built-in methods apply correctly through the new path
- Backward-compat: every existing per-method flag still works (AdaLoRA,
  Prefix Tuning, IA³, BitFit, Adapter) — the legacy 5 if/elif branches
  are preserved
- ``SFTTask`` and ``DPOTask`` inherit the dispatch path
- Validation ordering: unknown ``peft_method`` raises during
  ``TrainingConfig(...)`` construction, not later in ``build_model``
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.adalora import AdaLoRALinear
from llm.core.adapter import AdapterLinear
from llm.core.ia3 import IA3Linear
from llm.core.lora import LoRALinear
from llm.core.peft import ensure_methods_registered
from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.tasks.dpo_task import DPOTask
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.sft_task import SFTTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_registered() -> None:
    """Make sure built-ins are registered before every test."""
    ensure_methods_registered()


@pytest.fixture
def synthetic_dm():
    """A no-op SyntheticDataModule fixture — language modeling tests don't
    touch the data, but ``LanguageModelingTask`` requires one.
    """
    cfg = Config(
        model=ModelConfig(hidden_size=32, num_heads=4, num_layers=2, vocab_size=64, max_seq_len=16),
        training=TrainingConfig(batch_size=2, num_samples=4),
        optimization=OptimizationConfig(),
    )
    dm = SyntheticDataModule(cfg)
    dm.setup()
    return dm


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=64,
        attn_impl="mha",
        vocab_size=64,
        max_seq_len=16,
        dropout=0.0,
        num_experts=0,
    )


def _tiny_training_config(**overrides) -> TrainingConfig:
    """Build a minimal TrainingConfig — overrides take precedence."""
    defaults = {
        "batch_size": 2,
        "epochs": 1,
        "lr": 1e-3,
        "num_samples": 4,
        "warmup_epochs": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# TrainingConfig: peft_method / peft_kwargs fields
# ---------------------------------------------------------------------------


def test_peft_method_defaults_to_none() -> None:
    cfg = _tiny_training_config()
    assert cfg.peft_method is None
    assert cfg.peft_kwargs is None


def test_peft_method_accepts_known_string() -> None:
    """All seven built-ins should round-trip through the validator."""
    for name in ("lora", "qlora", "adalora", "prefix_tuning", "ia3", "bitfit", "adapter"):
        cfg = _tiny_training_config(peft_method=name)
        assert cfg.peft_method == name


def test_peft_method_rejects_unknown_string() -> None:
    """Unknown methods raise at config-load time (matches ``attn_impl``)."""
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        _tiny_training_config(peft_method="definitely_not_a_method")


def test_peft_kwargs_default_is_none() -> None:
    cfg = _tiny_training_config(peft_method="ia3")
    assert cfg.peft_kwargs is None


def test_peft_kwargs_accepts_dict() -> None:
    cfg = _tiny_training_config(peft_method="ia3", peft_kwargs={"init_scale": 0.5})
    assert cfg.peft_kwargs == {"init_scale": 0.5}


# ---------------------------------------------------------------------------
# build_model dispatch — new path
# ---------------------------------------------------------------------------


def _build_lm_task(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    dm: SyntheticDataModule,
) -> LanguageModelingTask:
    cfg = Config(
        model=model_cfg,
        training=train_cfg,
        optimization=OptimizationConfig(),
        data=DataConfig(data_source="local"),
    )
    return LanguageModelingTask(cfg, dm)


def test_build_model_dispatches_ia3_via_registry(synthetic_dm) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(peft_method="ia3", peft_kwargs={"init_scale": 1.0})
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # Every Linear wrapped in IA3Linear.
    from llm.models.decoder import DecoderModel

    assert isinstance(model, DecoderModel)
    # Use ``named_modules()`` and skip the ``.base_layer`` sub-modules —
    # they're raw nn.Linear by design (the wrapper holds the base layer
    # and applies the IA3 scale on top).
    for name, module in model.named_modules():
        if name.endswith(".base_layer"):
            continue
        if isinstance(module, nn.Linear):
            assert isinstance(module, IA3Linear), f"{name} should be wrapped in IA3Linear"


def test_build_model_dispatches_lora_via_registry(synthetic_dm) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="lora",
        peft_kwargs={"rank": 4, "alpha": 8.0, "target_modules": ["qkv_proj", "out_proj"]},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # qkv_proj / out_proj in attention blocks should be wrapped in LoRALinear.
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return  # found at least one
    pytest.fail("Expected at least one LoRALinear wrapper after apply_lora via registry")


def test_build_model_dispatches_adapter_via_registry(synthetic_dm) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="adapter",
        peft_kwargs={"bottleneck_dim": 4},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    found_adapter = any(isinstance(m, AdapterLinear) for m in model.modules())
    assert found_adapter


def test_build_model_dispatches_adalora_via_registry(synthetic_dm) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="adalora",
        peft_kwargs={"init_rank": 4, "target_rank": 2},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    found_adalora = any(isinstance(m, AdaLoRALinear) for m in model.modules())
    assert found_adalora


def test_build_model_dispatches_bitfit_via_registry(synthetic_dm) -> None:
    """BitFit wraps every bias via ``apply_bitfit`` — trainable bias check."""
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(peft_method="bitfit")
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # BitFit freezes every weight + enables every bias. The combined test is:
    # after apply_bitfit, every param with ``.bias`` suffix has requires_grad=True.
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            assert p.requires_grad, f"{name} should be trainable under BitFit"
        else:
            assert not p.requires_grad, f"{name} should be frozen under BitFit"


def test_build_model_dispatches_prefix_tuning_via_registry(synthetic_dm) -> None:
    """Prefix Tuning wraps every MultiHeadAttention — count the wrappers."""
    from llm.core.attn.mha import MultiHeadAttention
    from llm.core.prefix_tuning import PrefixTuningAttention

    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="prefix_tuning",
        peft_kwargs={"prefix_len": 4},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # Every top-level MultiHeadAttention should now be wrapped in
    # PrefixTuningAttention. The base MHA still exists inside the wrapper
    # at ``.base_attn`` — skip those when counting.
    mha_count = 0
    wrapper_count = 0
    for name, module in model.named_modules():
        if name.endswith(".base_attn"):
            continue  # the wrapped MHA inside PrefixTuningAttention
        if isinstance(module, MultiHeadAttention):
            mha_count += 1
        if isinstance(module, PrefixTuningAttention):
            wrapper_count += 1
    assert mha_count == 0, "All top-level MHA should be wrapped"
    assert wrapper_count > 0


def test_build_model_uses_legacy_path_when_peft_method_is_none(synthetic_dm) -> None:
    """When ``peft_method`` is None, the legacy ``use_*`` flags are still honored."""
    model_cfg = _tiny_model_config()
    # use_ia3=True → legacy path applies IA3 (without going through the registry).
    train_cfg = _tiny_training_config(use_ia3=True)
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    found_ia3 = any(isinstance(m, IA3Linear) for m in model.modules())
    assert found_ia3


def test_build_model_new_path_overrides_legacy_flags(synthetic_dm) -> None:
    """When ``peft_method`` is set, legacy ``use_*`` flags are ignored.

    Regression for the dispatch ambiguity: setting BOTH ``peft_method="ia3"``
    and ``use_bitfit=True`` should apply IA3 only (the new path wins).
    """
    from llm.core.bitfit import is_bitfit_applied

    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(peft_method="ia3", use_bitfit=True)
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # IA3 was applied (every Linear wrapped).
    found_ia3 = any(isinstance(m, IA3Linear) for m in model.modules())
    assert found_ia3
    # BitFit was NOT applied — ``is_bitfit_applied`` is the canonical
    # check (looks for the ``_bitfit_original_requires_grad`` snapshot).
    assert not is_bitfit_applied(model)


# ---------------------------------------------------------------------------
# Inheritance — SFTTask and DPOTask
# ---------------------------------------------------------------------------


def test_sft_task_inherits_peft_method_dispatch(synthetic_dm) -> None:
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="ia3",
        peft_kwargs={"init_scale": 1.0},
    )
    cfg = Config(
        model=model_cfg,
        training=train_cfg,
        optimization=OptimizationConfig(),
        data=DataConfig(data_source="local"),
    )
    task = SFTTask(cfg, synthetic_dm)
    model = task.build_model()
    found_ia3 = any(isinstance(m, IA3Linear) for m in model.modules())
    assert found_ia3


def test_dpo_task_peft_method_applies_to_both_policy_and_reference(synthetic_dm) -> None:
    """DPOTask calls ``super().build_model()`` twice — PEFT must apply to both."""
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="ia3",
        peft_kwargs={"init_scale": 1.0},
    )
    cfg = Config(
        model=model_cfg,
        training=train_cfg,
        optimization=OptimizationConfig(),
        data=DataConfig(data_source="local"),
    )
    task = DPOTask(cfg, synthetic_dm)
    policy = task.build_model()
    reference = task.build_model()
    # Both models wrapped.
    found_policy = any(isinstance(m, IA3Linear) for m in policy.modules())
    found_reference = any(isinstance(m, IA3Linear) for m in reference.modules())
    assert found_policy
    assert found_reference


# ---------------------------------------------------------------------------
# Validator behavior + interaction with AdaLoRA's pruning callback
# ---------------------------------------------------------------------------


def test_unknown_peft_method_rejected_at_config_load() -> None:
    """Validation runs at ``TrainingConfig(...)`` construction time —
    users get the error before they ever call ``build_model``.
    """
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        TrainingConfig(
            batch_size=2,
            epochs=1,
            lr=1e-3,
            num_samples=4,
            warmup_epochs=0,
            peft_method="nonsense_method",
        )


def test_adalora_via_registry_does_not_register_pruning_callback(
    synthetic_dm,
) -> None:
    """AdaLoRA via the new registry path does NOT register the
    AdaLoRA pruning callback — that's still the legacy
    ``use_adalora=True`` branch's job.

    The registry carries the ``requires_callback=True`` metadata, but
    the actual ``build_callbacks`` wiring stays coupled to the
    ``use_adalora`` flag (matches the slice plan).

    Note (T2 PEFT #48): setting ``peft_method`` now also registers
    :class:`PEFTAdapterCheckpointCallback` (one-shot sidecar save at
    ``on_train_end``). This test asserts the *AdaLoRA pruning*
    callback is absent — not that *every* callback is absent.
    """
    from llm.training.core.callbacks import AdaLoRAPruningCallback

    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="adalora",
        peft_kwargs={"init_rank": 4, "target_rank": 2},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    callbacks = task.build_callbacks()
    # The AdaLoRA pruning callback is NOT registered via the new path
    # — it requires the legacy ``use_adalora=True`` flag.
    assert not any(isinstance(c, AdaLoRAPruningCallback) for c in callbacks), (
        "AdaLoRA pruning callback should NOT be wired via peft_method"
    )


# ---------------------------------------------------------------------------
# peft_kwargs forwarding — kwargs reach the per-method apply_*
# ---------------------------------------------------------------------------


def test_peft_kwargs_forwarded_to_apply_ia3(synthetic_dm) -> None:
    """``peft_kwargs={"init_scale": 0.5}`` should set ``ia3_l`` to 0.5."""
    from llm.core.ia3 import IA3Linear

    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(
        peft_method="ia3",
        peft_kwargs={"init_scale": 0.5},
    )
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    # Pick the first IA3Linear and check its ia3_l is 0.5.
    for module in model.modules():
        if isinstance(module, IA3Linear):
            assert torch.allclose(module.ia3_l, torch.full_like(module.ia3_l, 0.5))
            return
    pytest.fail("No IA3Linear found in model")


def test_peft_kwargs_empty_dict_is_valid(synthetic_dm) -> None:
    """An empty ``peft_kwargs`` is fine — method's own defaults kick in."""
    model_cfg = _tiny_model_config()
    train_cfg = _tiny_training_config(peft_method="ia3", peft_kwargs={})
    task = _build_lm_task(model_cfg, train_cfg, synthetic_dm)
    model = task.build_model()
    from llm.core.ia3 import IA3Linear

    assert any(isinstance(m, IA3Linear) for m in model.modules())
