"""The random stream must CONTINUE across a checkpoint resume, not restart.

``DistributedManager.setup`` seeds ``torch.manual_seed(42 + global_rank)`` on
every process start, and nothing restored the RNG position from the checkpoint.
A resumed run therefore replayed the stream from the beginning: the DataLoader
shuffle produced the SAME batch orders and dropout re-drew the same masks, so
the resumed epochs re-trained on data already seen (measured: two
``randperm(8)`` draws after a restart were identical to the first run's).

These tests pin the continuation. They exercise the engine's checkpoint
contributor methods directly rather than a full save/load round trip, because
the RNG state is the only thing under test and a real checkpoint also writes
weights to disk.
"""

from __future__ import annotations

import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.regression_task import RegressionTask


def _engine() -> TrainingEngine:
    config = Config(
        training=TrainingConfig(task="regression", epochs=1, batch_size=2, num_samples=8, lr=1e-3),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", use_compile=False),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    return TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)


def test_a_restart_without_restoring_replays_the_stream():
    """Precondition: re-seeding alone DOES replay — this is the bug's shape.

    Without this, a passing continuation test could mean nothing happened at all.
    """
    torch.manual_seed(42)
    first = [torch.randn(1).item() for _ in range(3)]

    torch.manual_seed(42)  # exactly what a process restart does
    replayed = [torch.randn(1).item() for _ in range(3)]

    assert replayed == first, "re-seeding must reproduce the stream for this test to be meaningful"


def test_the_checkpoint_records_the_rng_state():
    """``get_checkpoint_state`` must carry the generators, not just the counter."""
    engine = _engine()
    state = engine.get_checkpoint_state()

    assert state is not None
    assert "engine.rng_state" in state, "the checkpoint does not persist the RNG state"
    cpu_state = state["engine.rng_state"]["cpu"]
    assert cpu_state.shape == torch.get_rng_state().shape
    assert cpu_state.dtype == torch.get_rng_state().dtype


def test_a_resume_continues_the_stream_exactly():
    """The resumed draws must equal an uninterrupted run's next draws.

    The comparison point matters. ``_engine()`` itself draws random numbers, so
    two engines built from the same seed land on the same stream position — and
    merely re-seeding then constructing the second engine reproduces that
    position "by accident", which is why an earlier version of this test passed
    even with the RNG restore removed. Here the state is captured AFTER the
    first engine exists, and the resumed engine's stream is forced somewhere
    else before loading, so only a real restore can land on the right position.
    """
    torch.manual_seed(42)
    engine = _engine()
    saved = engine.get_checkpoint_state()
    expected = [torch.randn(1).item() for _ in range(3)]

    resumed_engine = _engine()
    torch.manual_seed(999)  # move the stream far away from the saved position
    assert [torch.randn(1).item() for _ in range(3)] != expected  # precondition

    torch.manual_seed(999)
    resumed_engine.load_checkpoint_state(saved)
    actual = [torch.randn(1).item() for _ in range(3)]

    assert actual == expected, (
        f"the resumed stream diverged from an uninterrupted run: {actual} vs {expected}. "
        "A restart that does not restore the RNG state instead replays from a fresh seed."
    )


def test_a_checkpoint_without_rng_state_still_loads():
    """An older checkpoint has no RNG key; that must not be an error."""
    torch.manual_seed(7)
    engine = _engine()
    before = torch.get_rng_state().clone()

    engine.load_checkpoint_state({"engine.global_step": 5})

    assert engine.global_step == 5
    assert torch.equal(torch.get_rng_state(), before), (
        "loading a checkpoint with no RNG state must leave the current stream untouched"
    )
