"""Tests for Pfeiffer Adapter end-to-end trainer integration (T2 PEFT #46).

Validates that ``peft_method=\"pfeiffer_adapter\"`` flows through:

- ``TrainingConfig.peft_method`` field + validator
- ``LanguageModelingTask.build_model`` → ``apply_peft(...)`` dispatch
- SFTTask / DPOTask inheritance

No code changes are needed in ``lm_task.py`` or ``config.py`` — the
Slice 1 registry entry wires everything through the existing
``peft_method`` dispatch path. This test file just locks down the
end-to-end behavior so a future refactor can't silently break the
FFN-only design choice.

Covers:

- ``peft_method=\"pfeiffer_adapter\"`` is accepted at config-load time
- ``peft_kwargs`` flows into the wrapper (``bottleneck_dim`` etc.)
- The MLP layers are wrapped, attention layers are NOT (the FFN-only
  design choice — without this guard Pfeiffer collapses to Houlsby)
- SFT and DPO inherit the dispatch
- Pfeiffer coexists with Houlsby at the config level (the registry
  doesn't prevent both being enabled — project convention is one
  PEFT method per training run)
- The legacy ``use_adapter`` flag still works (backward compat)
- An invalid ``peft_method`` raises at config-load time, not at
  ``build_model`` time
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from llm.core.adapter import AdapterLinear
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
    ensure_methods_registered()


@pytest.fixture
def synthetic_dm():
    """No-op SyntheticDataModule — language modeling tests don't touch data."""
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
    defaults = {
        "batch_size": 2,
        "epochs": 1,
        "lr": 1e-3,
        "num_samples": 4,
        "warmup_epochs": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


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


# ---------------------------------------------------------------------------
# TrainingConfig: peft_method accepts "pfeiffer_adapter"
# ---------------------------------------------------------------------------


def test_peft_method_pfeiffer_adapter_accepted() -> None:
    """Slice 1 registered 'pfeiffer_adapter' — config validator must accept it."""
    cfg = _tiny_training_config(peft_method="pfeiffer_adapter")
    assert cfg.peft_method == "pfeiffer_adapter"


def test_peft_kwargs_forwarded_to_pfeiffer_adapter() -> None:
    """``peft_kwargs`` is stored on the config verbatim — the
    ``build_model`` dispatcher forwards it to ``apply_peft``.
    """
    cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 16},
    )
    assert cfg.peft_kwargs == {"bottleneck_dim": 16}


def test_unknown_peft_method_rejected_at_config_load() -> None:
    """Validation runs at ``TrainingConfig(...)`` construction — users
    get the error before they ever call ``build_model``.
    """
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        _tiny_training_config(peft_method="definitely_not_a_method")


# ---------------------------------------------------------------------------
# End-to-end dispatch — FFN-only behavior verified on a real DecoderModel
# ---------------------------------------------------------------------------


def test_build_model_applies_pfeiffer_to_mlp_only(synthetic_dm) -> None:
    """The headline FFN-only design choice — verified on the real model.

    Without this guard, Pfeiffer collapses to Houlsby (full stack). The
    test pins: ``mlp.fc1`` / ``mlp.fc2`` are wrapped, but
    ``self_attn.qkv_proj`` / ``self_attn.out_proj`` are NOT.
    """
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 8},
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    # MLP layers wrapped.
    assert isinstance(model.transformer_blocks[0].mlp.fc1, AdapterLinear)
    assert isinstance(model.transformer_blocks[0].mlp.fc2, AdapterLinear)
    # Attention layers NOT wrapped.
    assert isinstance(model.transformer_blocks[0].self_attn.qkv_proj, nn.Linear)
    assert isinstance(model.transformer_blocks[0].self_attn.out_proj, nn.Linear)
    assert not isinstance(model.transformer_blocks[0].self_attn.qkv_proj, AdapterLinear)


def test_pfeiffer_wraps_every_transformer_block(synthetic_dm) -> None:
    """All blocks get the FFN-only wrap — not just the first one."""
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 4},
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    for i, block in enumerate(model.transformer_blocks):
        assert isinstance(block.mlp.fc1, AdapterLinear), f"block {i} mlp.fc1 not wrapped"
        assert isinstance(block.mlp.fc2, AdapterLinear), f"block {i} mlp.fc2 not wrapped"


def test_bottleneck_dim_flows_into_wrapper(synthetic_dm) -> None:
    """``peft_kwargs['bottleneck_dim']`` ends up in the wrapper's field."""
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 12},
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    assert model.transformer_blocks[0].mlp.fc1.bottleneck_dim == 12
    assert model.transformer_blocks[0].mlp.fc2.bottleneck_dim == 12


def test_pfeiffer_target_modules_override(synthetic_dm) -> None:
    """A custom ``target_modules`` in ``peft_kwargs`` flows through.

    For instance, if a user wants Pfeiffer on a single MLP layer
    (``fc2`` only, for an extreme parameter-efficient setting), they
    can pass ``peft_kwargs={'target_modules': ['fc2']}``.
    """
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 4, "target_modules": ["fc2"]},
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    # Only fc2 wrapped, fc1 left alone.
    assert isinstance(model.transformer_blocks[0].mlp.fc1, nn.Linear)
    assert isinstance(model.transformer_blocks[0].mlp.fc2, AdapterLinear)


# ---------------------------------------------------------------------------
# Inheritance — SFTTask and DPOTask
# ---------------------------------------------------------------------------


def test_sft_task_inherits_pfeiffer_dispatch(synthetic_dm) -> None:
    """SFTTask.build_model reaches the same ``apply_peft`` path."""
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 8},
    )
    cfg = Config(
        model=_tiny_model_config(),
        training=train_cfg,
        optimization=OptimizationConfig(),
        data=DataConfig(data_source="local"),
    )
    task = SFTTask(cfg, synthetic_dm)
    model = task.build_model()
    assert isinstance(model.transformer_blocks[0].mlp.fc1, AdapterLinear)
    assert isinstance(model.transformer_blocks[0].mlp.fc2, AdapterLinear)
    # FFN-only preserved.
    assert isinstance(model.transformer_blocks[0].self_attn.qkv_proj, nn.Linear)


def test_dpo_task_pfeiffer_applies_to_both_models(synthetic_dm) -> None:
    """DPOTask calls ``super().build_model()`` twice — Pfeiffer must
    apply to both the policy and the reference model.
    """
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 8},
    )
    cfg = Config(
        model=_tiny_model_config(),
        training=train_cfg,
        optimization=OptimizationConfig(),
        data=DataConfig(data_source="local"),
    )
    task = DPOTask(cfg, synthetic_dm)
    policy = task.build_model()
    reference = task.build_model()
    # Both models have their MLP layers wrapped.
    assert isinstance(policy.transformer_blocks[0].mlp.fc1, AdapterLinear)
    assert isinstance(reference.transformer_blocks[0].mlp.fc1, AdapterLinear)


# ---------------------------------------------------------------------------
# Backward compat — legacy use_adapter flag still works (Houlsby path)
# ---------------------------------------------------------------------------


def test_legacy_use_adapter_still_wraps_full_stack(synthetic_dm) -> None:
    """The legacy ``use_adapter=True`` flag (Houlsby) must still wrap
    EVERY Linear including attention. The new peft_method path is
    additive — it doesn't replace the legacy flags.
    """
    train_cfg = _tiny_training_config(use_adapter=True, adapter_bottleneck_dim=8)
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    # Houlsby: attention AND MLP wrapped.
    assert isinstance(model.transformer_blocks[0].self_attn.qkv_proj, AdapterLinear)
    assert isinstance(model.transformer_blocks[0].mlp.fc1, AdapterLinear)


def test_peft_method_overrides_legacy_flags(synthetic_dm) -> None:
    """When ``peft_method`` is set, the legacy flags are ignored —
    matches the Slice 44 dispatch contract.
    """
    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 8},
        # Legacy flags also set — must be ignored when peft_method is set.
        use_adapter=True,
        adapter_bottleneck_dim=4,
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    model = task.build_model()
    # The new path wins: peft_method='pfeiffer_adapter' → FFN-only wrap.
    assert isinstance(model.transformer_blocks[0].mlp.fc1, AdapterLinear)
    assert isinstance(model.transformer_blocks[0].mlp.fc2, AdapterLinear)
    # Attention NOT wrapped — this distinguishes Pfeiffer from Houlsby.
    # If the legacy use_adapter=True had been honored, qkv_proj would be
    # wrapped too.
    assert isinstance(model.transformer_blocks[0].self_attn.qkv_proj, nn.Linear)


# ---------------------------------------------------------------------------
# No-callback guarantee — Pfeiffer has no periodic action (unlike AdaLoRA)
# ---------------------------------------------------------------------------


def test_pfeiffer_does_not_register_pruning_callback(synthetic_dm) -> None:
    """Pfeiffer has no periodic action — ``build_callbacks`` does NOT
    register :class:`AdaLoRAPruningCallback`.

    Contrast with AdaLoRA which registers the pruning callback via
    the legacy ``use_adalora=True`` path. Pfeiffer is a one-shot wrap
    with no scheduler / tracker.

    Note (T2 PEFT #48): setting ``peft_method`` now also registers
    :class:`PEFTAdapterCheckpointCallback` (one-shot sidecar save at
    ``on_train_end``). This test asserts the *pruning* callback is
    absent — not that *every* callback is absent.
    """
    from llm.training.core.callbacks import AdaLoRAPruningCallback

    train_cfg = _tiny_training_config(
        peft_method="pfeiffer_adapter",
        peft_kwargs={"bottleneck_dim": 8},
    )
    task = _build_lm_task(_tiny_model_config(), train_cfg, synthetic_dm)
    callbacks = task.build_callbacks()
    assert not any(isinstance(c, AdaLoRAPruningCallback) for c in callbacks), (
        "Pfeiffer should NOT register the AdaLoRA pruning callback"
    )
