"""Tests for TokenizedMapDataModule.setup_tokenized_file_dataset()."""

import pytest
import torch

from llm.data.modules.text import TextDataModule
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.training.core.config import Config

# 100 chars / max_seq_len 32 => 4 sequences; 90% split => 3 train, 1 val
_FIXED_CORPUS = "0123456789" * 10
_EXPECTED_SEQUENCE_COUNT = 4
_EXPECTED_TRAIN_COUNT = 3
_EXPECTED_VAL_COUNT = 1


def _config_with_printable_tokenizer(tmp_path, *, train_path: str, val_path: str | None = None) -> Config:
    """Build config with a saved printable tokenizer independent of dataset text."""
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(TokenizerFactory.from_printable_corpus(), tokenizer_path)

    config = Config()
    config.data.dataset_path = train_path
    config.data.val_dataset_path = val_path
    config.data.tokenizer_path = str(tokenizer_path)
    config.data.max_seq_len = 32
    return config


def test_setup_tokenized_file_dataset_splits_train_val_at_ninety_percent(tmp_path):
    """Requirement: map-style modules split file-backed data 90/10 when no val path is set."""
    data_file = tmp_path / "train.txt"
    data_file.write_text(_FIXED_CORPUS, encoding="utf-8")

    module = TextDataModule(_config_with_printable_tokenizer(tmp_path, train_path=str(data_file)))
    module.setup()

    assert len(module.train_dataset) == _EXPECTED_TRAIN_COUNT
    assert len(module.val_dataset) == _EXPECTED_VAL_COUNT
    assert _EXPECTED_TRAIN_COUNT + _EXPECTED_VAL_COUNT == _EXPECTED_SEQUENCE_COUNT


def test_setup_tokenized_file_dataset_uses_explicit_val_path(tmp_path):
    """Requirement: val_dataset_path loads a separate validation file without random split."""
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("train corpus line\n", encoding="utf-8")
    val_file.write_text("validation corpus line\n", encoding="utf-8")

    module = TextDataModule(
        _config_with_printable_tokenizer(
            tmp_path,
            train_path=str(train_file),
            val_path=str(val_file),
        )
    )
    module.setup()

    assert len(module.train_dataset) == 1
    assert len(module.val_dataset) == 1


def test_setup_tokenized_file_dataset_skips_when_dataset_path_empty():
    """Requirement: missing dataset_path leaves train/val unset instead of failing."""
    config = Config()
    config.data.dataset_path = ""

    module = TextDataModule(config)
    module.setup()

    assert module.train_dataset is None
    assert module.val_dataset is None


def test_setup_tokenized_file_dataset_raises_when_val_path_missing(tmp_path):
    """Requirement: explicit val_dataset_path must refer to an existing file."""
    train_file = tmp_path / "train.txt"
    train_file.write_text("train corpus line\n", encoding="utf-8")
    missing_val = tmp_path / "missing_val.txt"

    module = TextDataModule(
        _config_with_printable_tokenizer(
            tmp_path,
            train_path=str(train_file),
            val_path=str(missing_val),
        )
    )

    with pytest.raises(FileNotFoundError, match="missing_val"):
        module.setup()


def test_split_train_val_identical_across_ranks():
    """All DDP ranks must derive the same 90/10 partition (RIL ISS-087).

    ``DistributedManager.setup`` seeds the global torch RNG per rank
    (``manual_seed(42 + rank)``); if ``split_train_val`` consumed it, each
    GPU rank would split the full dataset differently — each ranks trains on
    a partly-disjoint 90% and the engine's ``reduce_mean``d val_loss averages
    over *different* examples. The partition must be rank-independent.
    """
    import torch
    from torch.utils.data import TensorDataset

    from llm.data.modules.map_base import TokenizedMapDataModule

    ds = TensorDataset(torch.arange(100))

    def _split():
        train, val = TokenizedMapDataModule.split_train_val(ds, 0.9)
        return sorted(train.indices), sorted(val.indices)

    torch.manual_seed(42)  # rank 0's seeded RNG (DistributedManager.setup)
    t0, v0 = _split()
    torch.manual_seed(43)  # rank 1's seeded RNG (42 + 1)
    t1, v1 = _split()

    assert t0 == t1, "train partition must be identical on every DDP rank"
    assert v0 == v1, "val partition must be identical on every DDP rank"
    assert len(t0) + len(v0) == 100


def test_split_train_val_single_sample_uses_whole_dataset_as_train():
    """RIL ISS-200: a 1-sample corpus must not yield an empty train set.

    ``int(0.9 * 1) == 0`` and ``val_size = 1``, so the old ``val_size <= 0``
    guard let ``random_split(dataset, [0, 1])`` produce an EMPTY train subset
    — the epoch then trained on zero batches and save_best/EarlyStopping read
    a meaningless 0.0 epoch_loss. A tiny corpus should fall back to
    whole-dataset train with no val split.
    """
    import torch
    from torch.utils.data import TensorDataset

    from llm.data.modules.map_base import TokenizedMapDataModule

    ds = TensorDataset(torch.tensor([1.0]))
    train, val = TokenizedMapDataModule.split_train_val(ds, 0.9)

    assert val is None, "no val split for a single-sample corpus"
    assert len(train) == 1, "train must keep the single sample"
    # Two-sample corpus still splits 1/1 (both sides non-empty).
    ds2 = TensorDataset(torch.tensor([1.0, 2.0]))
    train2, val2 = TokenizedMapDataModule.split_train_val(ds2, 0.9)
    assert val2 is not None
    assert len(train2) == 1
    assert len(val2) == 1
