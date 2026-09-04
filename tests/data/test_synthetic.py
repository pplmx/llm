"""Tests for SyntheticDataModule determinism across ranks (RIL ISS-134)."""

import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config


def _synthetic_config(num_samples: int = 16) -> Config:
    cfg = Config()
    cfg.model.hidden_size = 8
    cfg.model.vocab_size = 100
    cfg.training.batch_size = 4
    cfg.training.num_samples = num_samples
    return cfg


def _setup_like_rank(num_samples: int = 16, rank: int = 0):
    """Build + setup a SyntheticDataModule AFTER seeding the global RNG the way
    DistributedManager.setup does per rank (``torch.manual_seed(42 + rank)``).

    Mirrors the DDP setup order: the global RNG is intentionally per-rank, so
    the ONLY way two ranks see identical synthetic data is a dedicated
    generator inside the module.
    """
    torch.manual_seed(42 + rank)
    module = SyntheticDataModule(_synthetic_config(num_samples))
    module.setup()
    return module


def test_synthetic_data_identical_across_ranks():
    """RIL ISS-134: per-rank global RNG seeding must NOT diverge the synthetic
    train/val data. DDP regression previously trained each rank on different
    values at the same sampler index (incoherent global batch, meaningless
    aggregate val loss)."""
    a = _setup_like_rank(rank=0)
    b = _setup_like_rank(rank=1)

    assert len(a.train_dataset) == len(b.train_dataset)
    for i in range(len(a.train_dataset)):
        xa, _ = a.train_dataset[i]
        xb, _ = b.train_dataset[i]
        assert torch.equal(xa, xb), f"train sample {i} diverged across ranks"

    assert len(a.val_dataset) == len(b.val_dataset)
    for i in range(len(a.val_dataset)):
        xa, _ = a.val_dataset[i]
        xb, _ = b.val_dataset[i]
        assert torch.equal(xa, xb), f"val sample {i} diverged across ranks"


def test_synthetic_data_deterministic_across_runs():
    """The same module built twice yields identical data (reproducibility even
    on a single rank, where the global RNG could otherwise be mid-stream)."""
    a = _setup_like_rank(rank=0)
    b = _setup_like_rank(rank=0)
    for i in range(len(a.train_dataset)):
        assert torch.equal(a.train_dataset[i][0], b.train_dataset[i][0])
    for i in range(len(a.val_dataset)):
        assert torch.equal(a.val_dataset[i][0], b.val_dataset[i][0])


def test_synthetic_data_module_rejects_zero_samples():
    """A ``num_samples=0`` config would build a 0-length train set → an empty
    epoch → the engine's per-epoch average divides by zero (RIL ISS-336)."""
    import pytest

    module = SyntheticDataModule(_synthetic_config(num_samples=0))
    with pytest.raises(ValueError, match="num_samples"):
        module.setup()
