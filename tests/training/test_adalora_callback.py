"""Tests for ``AdaLoRAPruningCallback`` and SFT/DPO integration (T3 #42).

Two concerns:

1. ``AdaLoRAPruningCallback`` wires the EMA tracker + prune step into
   the standard training loop, including checkpoint roundtrip.
2. ``LanguageModelingTask.build_model`` applies AdaLoRA and the task
   registers the pruning callback on the engine.

The schema-only tests live in ``test_adalora_config.py`` so the two
modules can be RED/GREEN'd independently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from llm.core.adalora import AdaLoRAGradientEMA, AdaLoRALinear, apply_adalora
from llm.training.core.callbacks import AdaLoRAPruningCallback


def _make_engine_stub(model: nn.Module | None = None, *, rank: int = 0) -> MagicMock:
    """A minimal engine stub that satisfies the callback's surface needs."""
    engine = MagicMock()
    engine.rank = rank
    engine.model = model if model is not None else nn.Sequential(nn.Linear(8, 8))
    engine.global_step = 0
    engine.logger = MagicMock()
    return engine


class TestAdaLoRAPruningCallbackNoOp:
    """The callback is a no-op when ``use_adalora=False``."""

    def test_use_adalora_false_skips_tracker_construction(self):
        cb = AdaLoRAPruningCallback(use_adalora=False)
        cb.set_engine(_make_engine_stub())
        cb.on_train_start()  # must not raise
        assert cb._tracker is None

    def test_use_adalora_false_does_not_prune(self):
        cb = AdaLoRAPruningCallback(use_adalora=False)
        engine = _make_engine_stub()
        cb.set_engine(engine)
        cb.on_train_start()
        cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
        # No tracker, no prune call.
        assert cb._tracker is None


class TestAdaLoRAPruningCallbackSetup:
    """With ``use_adalora=True``, the tracker is built against engine.model."""

    def test_tracker_built_on_train_start(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_ema_alpha=0.9)
        cb.set_engine(_make_engine_stub(model=model))
        cb.on_train_start()
        assert isinstance(cb._tracker, AdaLoRAGradientEMA)
        assert cb._tracker.alpha == 0.9
        # Two AdaLoRA layers in a Sequential → two EMAs.
        assert len(cb._tracker.as_dict()) == 2

    def test_callback_uses_engine_global_step(self):
        cb = AdaLoRAPruningCallback(
            use_adalora=True,
            adalora_prune_every=10,
            adalora_tinit=0,
            adalora_tfinal=20,  # budget collapses to target_rank at step >= 20
        )
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model)
        cb.set_engine(engine)
        cb.on_train_start()

        # Step 1: too early → no prune
        cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        assert layer.effective_rank == 4  # unchanged

        # Step 10 → prune fires (10 % 10 == 0). Schedule (0, 20) at step 10
        # gives budget = round(4 - 0.5*2) = 3.
        engine.global_step = 10
        cb.on_train_step_end(epoch=0, batch_idx=9, loss=torch.tensor(0.0), metrics={})
        assert layer.effective_rank == 3

        # Step 20 → budget collapses to target_rank=2.
        engine.global_step = 20
        cb.on_train_step_end(epoch=0, batch_idx=19, loss=torch.tensor(0.0), metrics={})
        assert layer.effective_rank == 2

    def test_prune_only_fires_on_cadence(self):
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=3)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model)
        cb.set_engine(engine)
        cb.on_train_start()
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))

        # Steps 1, 2: no prune (rank stays at init_rank=4).
        for step in (1, 2):
            engine.global_step = step
            cb.on_train_step_end(epoch=0, batch_idx=step - 1, loss=torch.tensor(0.0), metrics={})
            assert layer.effective_rank == 4

        # Step 3: prune fires.
        engine.global_step = 3
        cb.on_train_step_end(epoch=0, batch_idx=2, loss=torch.tensor(0.0), metrics={})
        assert layer.effective_rank == 2


class TestAdaLoRAPruningCallbackEMA:
    """Update is called on every optimizer step, regardless of prune cadence."""

    def test_update_called_every_step(self):
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=100)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model)
        cb.set_engine(engine)
        cb.on_train_start()

        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        layer.lora_lambda.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Hook is on_optimizer_step (fires BEFORE zero_grad clears the grads).
        cb.on_optimizer_step(epoch=0, batch_idx=0)
        ema = cb._tracker.as_dict()[id(layer)]
        assert torch.allclose(ema, 0.05 * torch.ones(4))  # a=0.95 -> 0.05*1.0

    def test_update_via_on_train_step_end_is_a_noop(self):
        """Regression: zero_grad runs between optimizer.step() and
        on_train_step_end, so the EMA must NOT rely on the latter hook.
        If it did, the EMA would always be zero in production.
        """
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=100)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model)
        cb.set_engine(engine)
        cb.on_train_start()

        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        layer.lora_lambda.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Simulate the engine: zero_grad runs first, then on_train_step_end.
        layer.lora_lambda.grad = None

        cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
        ema = cb._tracker.as_dict()[id(layer)]
        # EMA stays at zero because we never got a real gradient to fold in.
        assert torch.allclose(ema, torch.zeros(4))


class TestAdaLoRAPruningCallbackLogging:
    """Effective-rank logging at rank 0 only."""

    def test_effective_rank_logged_at_rank_0(self):
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=1)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model, rank=0)
        cb.set_engine(engine)
        cb.on_train_start()

        engine.global_step = 1
        cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
        engine.logger.info.assert_any_call("adalora/effective_rank=2.0")

    def test_no_log_at_rank_nonzero(self):
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=1)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        engine = _make_engine_stub(model=model, rank=1)
        cb.set_engine(engine)
        cb.on_train_start()

        engine.global_step = 1
        cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
        for call in engine.logger.info.call_args_list:
            assert "adalora/effective_rank" not in call.args[0]


class TestAdaLoRAPruningCallbackCheckpoint:
    """Checkpoint roundtrip via the existing CheckpointContributor protocol."""

    def test_get_checkpoint_state_returns_ema_state(self):
        cb = AdaLoRAPruningCallback(use_adalora=True, adalora_prune_every=10)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        cb.set_engine(_make_engine_stub(model=model))
        cb.on_train_start()

        # Manually populate EMAs (the live buffers — state_dict() returns
        # a snapshot of these).
        for ema in cb._tracker._emas.values():
            ema.fill_(0.7)

        state = cb.get_checkpoint_state()
        assert state is not None
        # State is keyed by qualified name (e.g. "0") not id().
        for tensor in state.values():
            assert torch.allclose(tensor, 0.7 * torch.ones(4))

    def test_get_checkpoint_state_none_when_disabled(self):
        cb = AdaLoRAPruningCallback(use_adalora=False)
        cb.set_engine(_make_engine_stub())
        # Not even on_train_start has been called → must still be safe.
        assert cb.get_checkpoint_state() is None

    def test_load_checkpoint_state_restores_ema(self):
        cb = AdaLoRAPruningCallback(use_adalora=True)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        cb.set_engine(_make_engine_stub(model=model))
        cb.on_train_start()

        # Simulate a saved checkpoint.
        saved = {name: torch.full((4,), float(idx + 1) * 0.1) for idx, name in enumerate(cb._tracker.state_dict())}
        cb.load_checkpoint_state(saved)

        for name, expected in saved.items():
            assert torch.allclose(cb._tracker.state_dict()[name], expected, atol=1e-7)

    def test_load_checkpoint_state_none_is_safe(self):
        cb = AdaLoRAPruningCallback(use_adalora=True)
        model = nn.Sequential(nn.Linear(8, 8))
        apply_adalora(model, init_rank=4, target_rank=2)
        cb.set_engine(_make_engine_stub(model=model))
        cb.on_train_start()
        # Must not raise.
        cb.load_checkpoint_state(None)


# ---------------------------------------------------------------------------
# Task integration (LM / SFT / DPO)
# ---------------------------------------------------------------------------


def _tiny_config(use_adalora: bool = True, **overrides):
    """Build a Config that turns AdaLoRA on (or off) at the right scope."""
    from llm.training.core.config import Config

    base = Config()
    base.training.use_adalora = use_adalora
    for key, value in overrides.items():
        setattr(base.training, key, value)
    return base


class TestLanguageModelingTaskAdaLoRAWiring:
    """``LanguageModelingTask.build_model`` applies AdaLoRA when configured."""

    def test_apply_adalora_when_use_adalora_true(self):
        from llm.training.tasks.lm_task import LanguageModelingTask

        config = _tiny_config(use_adalora=True)
        task = LanguageModelingTask(config, data_module=None)

        model = task.build_model()
        # The model should now contain AdaLoRA layers (since hidden_size is
        # small enough for init_rank=12 to fit).
        ada_layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
        assert len(ada_layers) > 0

    def test_no_adalora_when_use_adalora_false(self):
        from llm.training.tasks.lm_task import LanguageModelingTask

        config = _tiny_config(use_adalora=False)
        task = LanguageModelingTask(config, data_module=None)
        model = task.build_model()
        ada_layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
        assert ada_layers == []

    def test_target_modules_forwarded_to_apply_adalora(self):
        from llm.training.tasks.lm_task import LanguageModelingTask

        # Hand-built model with named projections so we can match by substring.
        config = _tiny_config(use_adalora=True, adalora_target_modules=["q_proj"])
        task = LanguageModelingTask(config, data_module=None)
        model = task.build_model()
        # Only modules whose qualified name contains "q_proj" should be wrapped.
        names = [name for name, m in model.named_modules() if isinstance(m, AdaLoRALinear)]
        for name in names:
            assert "q_proj" in name


class TestSFTDPOTaskInheritance:
    """SFT and DPO inherit AdaLoRA wiring from ``LanguageModelingTask``."""

    def test_sft_task_applies_adalora(self):
        from llm.training.tasks.sft_task import SFTTask

        config = _tiny_config(use_adalora=True)
        task = SFTTask(config, data_module=None)
        model = task.build_model()
        assert any(isinstance(m, AdaLoRALinear) for m in model.modules())

    def test_dpo_task_applies_adalora_to_policy(self):
        """DPO builds both a policy and a reference model — both get AdaLoRA."""
        from llm.training.tasks.dpo_task import DPOTask

        config = _tiny_config(use_adalora=True)
        task = DPOTask(config, data_module=None)
        policy = task.build_model()
        assert any(isinstance(m, AdaLoRALinear) for m in policy.modules())
        assert task.ref_model is not None
        assert any(isinstance(m, AdaLoRALinear) for m in task.ref_model.modules())
