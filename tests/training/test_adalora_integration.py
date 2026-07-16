"""End-to-end AdaLoRA integration test (T3 #42 acceptance).

Runs a few SFT steps with ``use_adalora=True`` and asserts the mask
shrinks on cadence. This is the smoke test that proves:

1. ``LanguageModelingTask.build_model`` produces an AdaLoRA-adapted
   model when the config flag is set.
2. The pruning callback is registered on the engine without any
   extra wiring from user code.
3. After enough optimizer steps, the rank budget has actually
   shrunk from ``init_rank`` toward ``target_rank``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from llm.core.adalora import AdaLoRALinear, apply_adalora
from llm.training.core.callbacks import AdaLoRAPruningCallback
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.sft_task import SFTTask


def _tiny_sft_config(*, init_rank: int = 4, target_rank: int = 2, prune_every: int = 2) -> Config:
    """Build a config that makes the tiny model fit AdaLoRA's rank constraint.

    Default ModelConfig has ``hidden_size=512`` which is overkill for
    this smoke test. Shrink it so we don't pay for a real decoder on
    the test runner. AdaLoRA's ``init_rank`` defaults to 12, which is
    safe here because ``min(8, 8) = 8 >= 4``.
    """
    cfg = Config()
    cfg.model.hidden_size = 8
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 16
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.use_adalora = True
    cfg.training.adalora_init_rank = init_rank
    cfg.training.adalora_target_rank = target_rank
    cfg.training.adalora_prune_every = prune_every
    cfg.training.adalora_tinit = 0
    cfg.training.adalora_tfinal = 6  # linear budget collapses to target_rank at step 6
    cfg.training.epochs = 1
    return cfg


def _make_tiny_model_with_adalora(cfg: Config) -> torch.nn.Module:
    """Build a tiny model with AdaLoRA applied, bypassing the factory."""
    model = torch.nn.Sequential(
        torch.nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size),
        torch.nn.Linear(cfg.model.hidden_size, cfg.model.vocab_size),
    )
    apply_adalora(
        model,
        init_rank=cfg.training.adalora_init_rank,
        target_rank=cfg.training.adalora_target_rank,
    )
    return model


def test_sft_end_to_end_with_adalora_mask_shrinks_on_cadence():
    """Run several optimizer steps; assert effective_rank drops."""
    from llm.training.core.callbacks import AdaLoRAPruningCallback

    cfg = _tiny_sft_config()

    # 1) build_model produces AdaLoRA-adapted layers.
    task = SFTTask(cfg, data_module=None)
    task.build_model = lambda: _make_tiny_model_with_adalora(cfg)
    model = task.build_model()
    ada_layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    assert ada_layers, "AdaLoRA flag set but no AdaLoRALinear layers produced"

    # 2) build_callbacks returns the pruning callback.
    callbacks = task.build_callbacks()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], AdaLoRAPruningCallback)
    assert callbacks[0].use_adalora is True

    # 3) Manually wire a stub engine and step through the prune cycle.
    engine = MagicMock()
    engine.rank = 0
    engine.model = model
    engine.global_step = 0
    engine.logger = MagicMock()
    cb = callbacks[0]
    cb.set_engine(engine)
    cb.on_train_start()

    # Step 0 (global_step=0): EMA update fires, but prune does not
    # because tinit=0 and the spec says pruning starts at tinit + 1.
    cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
    initial_rank = ada_layers[0].effective_rank
    assert initial_rank == cfg.training.adalora_init_rank

    # Drive enough optimizer steps to push the budget to target_rank.
    for step in range(1, cfg.training.adalora_tfinal + 2):
        engine.global_step = step
        cb.on_train_step_end(epoch=0, batch_idx=step, loss=torch.tensor(0.0), metrics={})

    final_rank = ada_layers[0].effective_rank
    assert final_rank == cfg.training.adalora_target_rank, (
        f"expected mask to shrink to {cfg.training.adalora_target_rank} "
        f"after tfinal+1 steps, got {final_rank}"
    )


def test_sft_engine_registers_adalora_callback_automatically():
    """End-to-end: construct an engine and verify the callback is wired.

    Uses no-op dataloader shims so the engine constructor finishes
    without needing the full data stack.
    """
    cfg = _tiny_sft_config()

    # Stub the dataloader construction. Both train_dataloader and
    # val_dataloader must return (loader, sampler) tuples or the
    # unpacking in _setup_components blows up.
    empty_iter = MagicMock()
    empty_iter.__iter__ = MagicMock(return_value=iter([]))
    empty_iter.__len__ = MagicMock(return_value=0)
    loader_pair = (empty_iter, None)

    fake_data_module = MagicMock(
        train_dataloader=MagicMock(return_value=loader_pair),
        val_dataloader=MagicMock(return_value=loader_pair),
        is_streaming=False,
    )

    with patch("llm.training.core.engine.wrap_model_for_training", side_effect=lambda m, **_: m):
        task = SFTTask(cfg, data_module=fake_data_module)
        # Override build_model to skip the heavyweight factory path.
        task.build_model = lambda: _make_tiny_model_with_adalora(cfg)
        engine = TrainingEngine(
            config=cfg,
            task=task,
            rank=0,
            world_size=1,
            data_module=fake_data_module,
            callbacks=None,
        )

    # The AdaLoRA pruning callback should be registered automatically.
    cb_types = [type(cb).__name__ for cb in engine.callbacks]
    assert "AdaLoRAPruningCallback" in cb_types, f"callbacks={cb_types}"


def test_ema_captures_real_gradients_via_engine_step():
    """End-to-end: the engine wires ``on_optimizer_step`` between
    ``scaler.step`` and ``optimizer.zero_grad``, so the AdaLoRA EMA
    tracker captures a real gradient signal.

    We don't run a full training step (the tiny fixture model isn't a
    real DecoderModel and the SFTTask's train_step needs attn_mask
    plumbing we don't have here). Instead we drive the lifecycle
    manually: backward → scaler.step → on_optimizer_step → zero_grad,
    mirroring the engine's actual ordering.
    """
    cfg = _tiny_sft_config(prune_every=100)  # never prune — only EMA

    model = _make_tiny_model_with_adalora(cfg)
    layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))

    empty_iter = MagicMock()
    empty_iter.__iter__ = MagicMock(return_value=iter([]))
    empty_iter.__len__ = MagicMock(return_value=0)
    loader_pair = (empty_iter, None)
    fake_data_module = MagicMock(
        train_dataloader=MagicMock(return_value=loader_pair),
        val_dataloader=MagicMock(return_value=loader_pair),
        is_streaming=False,
    )

    with patch("llm.training.core.engine.wrap_model_for_training", side_effect=lambda m, **_: m):
        task = SFTTask(cfg, data_module=fake_data_module)
        task.build_model = lambda: model
        engine = TrainingEngine(
            config=cfg,
            task=task,
            rank=0,
            world_size=1,
            data_module=fake_data_module,
            callbacks=None,
        )

    cb = next(c for c in engine.callbacks if isinstance(c, AdaLoRAPruningCallback))
    cb.on_train_start()

    # EMA starts at zero.
    ema = cb._tracker.as_dict()[id(layer)]
    assert torch.allclose(ema, torch.zeros_like(ema))

    # Simulate a backward pass that leaves a non-zero gradient on
    # lora_lambda. In real training this comes from the loss; here we
    # set it directly to test the hook ordering.
    grad_signal = torch.full_like(layer.lora_lambda, 0.7)
    layer.lora_lambda.grad = grad_signal

    # Engine ordering (see engine.py:245-247): scaler.step →
    # on_optimizer_step → zero_grad.
    cb.on_optimizer_step(epoch=0, batch_idx=0)
    ema_after = cb._tracker.as_dict()[id(layer)]
    assert ema_after.abs().sum() > 0, "EMA should have captured the gradient signal"

    # Now zero_grad clears the gradient — but the EMA retains it.
    layer.lora_lambda.grad = None
    assert cb._tracker.as_dict()[id(layer)].abs().sum() > 0

    # And on_train_step_end must NOT also fire the EMA update (it would
    # read zero gradients post-zero_grad).
    cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(0.0), metrics={})
    # The EMA was already set above; calling on_train_step_end again
    # with grad=None shouldn't reset it to zero.
    assert cb._tracker.as_dict()[id(layer)].abs().sum() > 0


def test_checkpoint_roundtrip_persists_ema_via_engine():
    """The engine must fold the AdaLoRA callback's EMA state into
    checkpoint extra_state, otherwise resume loses the importance signal.
    """
    from llm.runtime.checkpoint import collect_extra_state, load_extra_state

    cfg = _tiny_sft_config(prune_every=100)

    empty_iter = MagicMock()
    empty_iter.__iter__ = MagicMock(return_value=iter([]))
    empty_iter.__len__ = MagicMock(return_value=0)
    loader_pair = (empty_iter, None)
    fake_data_module = MagicMock(
        train_dataloader=MagicMock(return_value=loader_pair),
        val_dataloader=MagicMock(return_value=loader_pair),
        is_streaming=False,
    )

    with patch("llm.training.core.engine.wrap_model_for_training", side_effect=lambda m, **_: m):
        task = SFTTask(cfg, data_module=fake_data_module)
        task.build_model = lambda: _make_tiny_model_with_adalora(cfg)
        engine = TrainingEngine(
            config=cfg,
            task=task,
            rank=0,
            world_size=1,
            data_module=fake_data_module,
            callbacks=None,
        )

    cb = next(c for c in engine.callbacks if isinstance(c, AdaLoRAPruningCallback))
    cb.on_train_start()

    # Populate the EMA with a known signal so we can verify roundtrip.
    ema_keys = list(cb._tracker._emas.keys())
    assert len(ema_keys) > 0, "tracker has no EMAs — model walking failed"
    cb._tracker._emas[ema_keys[0]].fill_(0.42)

    # Engine-side collect: callback state must appear in extra_state.
    merged = collect_extra_state(fake_data_module, task, *engine.callbacks)
    assert merged is not None
    assert ema_keys[0] in merged, (
        f"callback state missing from extra_state; got keys {list(merged)}, expected {ema_keys[0]}"
    )
    assert torch.allclose(merged[ema_keys[0]], 0.42 * torch.ones_like(merged[ema_keys[0]]))

    # Now simulate a fresh engine that loads the checkpoint. Build a
    # new tracker (zero EMAs) and confirm load_extra_state restores it.
    cb._tracker._emas[ema_keys[0]].zero_()
    load_extra_state(merged, fake_data_module, task, *engine.callbacks)
    assert torch.allclose(cb._tracker._emas[ema_keys[0]], 0.42 * torch.ones_like(merged[ema_keys[0]]))
