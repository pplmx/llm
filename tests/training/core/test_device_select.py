"""Most-free-VRAM CUDA device selection (RIL TASK-266 / DEC-094).

The runtime (TrainingEngine + distributed launcher) must map each rank to the
GPU with the most free VRAM instead of the historical ``rank % device_count``
(always ``cuda:0`` for rank 0), so runs land on the fattest device on shared /
busy hosts instead of OOMing on a contended GPU 0.
"""

from __future__ import annotations

import pytest
import torch

from llm.training.core import device_select as ds

MIN = ds.MIN_FREE_VRAM_BYTES


def _fake_free(entries: dict[int, int | None]):
    """A ``free_bytes_fn`` serving fixed values (``None`` = inaccessible)."""

    def fn(index: int) -> int | None:
        return entries.get(index)

    return fn


# ---------------------------------------------------------------------------
# pure selection logic (no CUDA required)
# ---------------------------------------------------------------------------


def test_sort_prefers_most_free_vram_then_lower_index():
    """Sorting puts the fattest GPU first and breaks ties by lower index."""
    fn = _fake_free({0: 2 * MIN, 1: 5 * MIN, 2: 5 * MIN})
    assert ds.sort_by_free_vram([0, 1, 2], free_bytes_fn=fn) == [1, 2, 0]


def test_sort_filters_below_floor_and_inaccessible():
    fn = _fake_free({0: MIN, 1: MIN - 1, 2: 8 * MIN, 3: None})
    assert ds.sort_by_free_vram([0, 1, 2, 3], free_bytes_fn=fn) == [2, 0]


def test_select_rank_mapping_fattest_first():
    """rank 0 -> most free, rank 1 -> second most, rank 2 -> least (ascending index)."""
    fn = _fake_free({0: 2 * MIN, 1: 5 * MIN, 2: 9 * MIN})
    assert ds.select_cuda_index(0, free_bytes_fn=fn) == 2
    assert ds.select_cuda_index(1, free_bytes_fn=fn) == 1
    assert ds.select_cuda_index(2, free_bytes_fn=fn) == 0


def test_select_round_robins_over_usable_gpus():
    """A rank beyond the usable count wraps around (matching old modulo behaviour)."""
    fn = _fake_free({0: 9 * MIN, 1: 5 * MIN, 2: MIN - 1})  # only 0 and 1 usable
    assert ds.select_cuda_index(0, free_bytes_fn=fn) == 0
    assert ds.select_cuda_index(1, free_bytes_fn=fn) == 1
    assert ds.select_cuda_index(2, free_bytes_fn=fn) == 0


def test_select_returns_none_when_all_below_floor():
    fn = _fake_free({0: MIN - 1, 1: 0})
    assert ds.select_cuda_index(0, free_bytes_fn=fn) is None


def test_select_respects_explicit_device_count():
    fn = _fake_free({0: MIN, 1: 9 * MIN, 2: 9 * MIN})
    assert ds.select_cuda_index(0, n_devices=2, free_bytes_fn=fn) == 1


def test_cuda_usable_impl_rejects_when_not_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ds._cuda_usable_impl(0, free_bytes_fn=_fake_free({0: 9 * MIN})) is False


def test_cuda_usable_impl_requires_free_floor(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert ds._cuda_usable_impl(0, free_bytes_fn=_fake_free({0: MIN - 1})) is False
    assert ds._cuda_usable_impl(1, free_bytes_fn=_fake_free({1: MIN})) is True


# ---------------------------------------------------------------------------
# TrainingEngine integration (only meaningful with CUDA visible)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_engine_parts(tmp_path):
    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.config import Config, LoggingConfig, ModelConfig, OptimizationConfig, TrainingConfig
    from llm.training.tasks.regression_task import RegressionTask

    config = Config(
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            lr=1e-3,
            output_dir=str(tmp_path / "output"),
            num_samples=20,  # Required by SyntheticDataModule
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        logging=LoggingConfig(log_interval=1, log_level="ERROR"),
        optimization=OptimizationConfig(num_workers=0),  # Avoid multiprocessing in tests
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    return config, data_module, RegressionTask


def _engine_device(parts, rank, world_size, monkeypatch, free_entries):
    from llm.training.core.engine import TrainingEngine

    config, data_module, task_cls = parts
    task = task_cls(config, data_module=data_module)
    monkeypatch.setattr(ds, "_free_bytes", _fake_free(free_entries))
    engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
    return engine.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_engine_rank0_lands_on_most_free_gpu(tiny_engine_parts, monkeypatch):
    """rank 0 must pick the GPU with the most free VRAM, not cuda:0."""
    indices = list(range(torch.cuda.device_count()))
    free_entries = {i: MIN * (3 if i == 0 else 2 + i) for i in indices}
    fattest = ds.sort_by_free_vram(indices, free_bytes_fn=_fake_free(free_entries))[0]
    assert fattest != 0 or torch.cuda.device_count() == 1  # skip trivial case
    device = _engine_device(tiny_engine_parts, 0, 1, monkeypatch, free_entries)
    assert device == torch.device(f"cuda:{fattest}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_engine_falls_back_to_cpu_when_all_gpus_below_floor(tiny_engine_parts, monkeypatch):
    """No usable GPU -> the engine degrades to CPU (historical behaviour)."""
    indices = list(range(max(1, torch.cuda.device_count())))
    free_entries = dict.fromkeys(indices, MIN - 1)
    device = _engine_device(tiny_engine_parts, 0, 1, monkeypatch, free_entries)
    assert device == torch.device("cpu")
