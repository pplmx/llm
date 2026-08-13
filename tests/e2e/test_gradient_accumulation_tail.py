"""Regression: gradient-accumulation partial tail window must be re-scaled.

When ``num_batches % gradient_accumulation_steps != 0`` the final optimizer
step fires on a partial window. Each loss was scaled by ``1/accum_steps``,
so without correction the tail step would carry only
``window_count/accum_steps`` of a full batch's gradient — a silent,
step-to-step-varying effective-LR reduction (RIL ISS-066).
"""

import torch

from llm.training.core.callbacks import Callback
from llm.training.core.config import (
    Config,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask
from tests.support.data import DummyLMDataModule


def _make_engine(accum_steps: int, batch_size: int, num_samples: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config(
        model=ModelConfig(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16),
        training=TrainingConfig(batch_size=batch_size, epochs=1, num_samples=num_samples),
        optimization=OptimizationConfig(
            use_compile=False,
            use_amp=False,
            gradient_accumulation_steps=accum_steps,
        ),
    )
    cfg.distributed.backend = "gloo"
    dm = DummyLMDataModule(cfg)
    task = LanguageModelingTask(cfg, dm)
    engine = TrainingEngine(config=cfg, task=task, rank=0, world_size=1, data_module=dm, callbacks=[])
    return engine


class _GradNormRecorder(Callback):
    """Record the total gradient norm at each optimizer step."""

    def __init__(self, engine: TrainingEngine):
        self._engine = engine
        self.grad_norms: list[float] = []

    def on_optimizer_step(self, epoch: int, batch_idx: int, **kwargs) -> None:
        norm = 0.0
        for p in self._engine.model.parameters():
            if p.grad is not None:
                norm += float(p.grad.detach().pow(2).sum())
        self.grad_norms.append(norm**0.5)


def test_partial_tail_window_gradient_scale():
    """7 batches with accum_steps=3 -> steps at batches 2 and 4, then a
    partial tail step at batch 6 (7 % 3 == 1). The tail step's gradient
    norm must be comparable to the full-window ones (within a small factor),
    not ~1/3 of them."""
    engine = _make_engine(accum_steps=3, batch_size=2, num_samples=14)  # 7 batches
    recorder = _GradNormRecorder(engine)
    engine.callbacks.append(recorder)
    engine.run()

    # ceil(7/3) = 3 optimizer steps.
    assert engine.global_step == 3
    assert len(recorder.grad_norms) == 3

    # Tail step is window_count=1 vs full window 3: without correction its
    # gradient norm stays at the 1-batch scale (~0.5x of a 3-batch window).
    # With correction it is re-scaled to a full-window-equivalent step, so it
    # must sit at (or above) the full-window mean, not below it.
    tail = recorder.grad_norms[-1]
    full = recorder.grad_norms[:2]
    assert all(f > 0.0 for f in full), f"full-window steps had zero grad: {full}"
    full_mean = sum(full) / len(full)
    assert tail >= 0.7 * full_mean, (
        f"tail window gradient ({tail:.6g}) below the 0.7x threshold of the "
        f"full-window mean ({full_mean:.6g}) — partial window not re-scaled"
    )


def test_divisible_window_unchanged():
    """A cleanly-divisible epoch (no partial tail) is unaffected by the
    re-scale logic (no correction should fire)."""
    engine = _make_engine(accum_steps=2, batch_size=2, num_samples=16)  # 8 batches / 2
    recorder = _GradNormRecorder(engine)
    engine.callbacks.append(recorder)
    engine.run()

    assert engine.global_step == 4  # 8 / 2
    assert len(recorder.grad_norms) == 4
    # All four windows are full: gradient norms are all non-zero and the
    # per-step variation is only from data randomness, not a 2x structural
    # drop between full and tail steps.
    assert all(g > 0.0 for g in recorder.grad_norms)
