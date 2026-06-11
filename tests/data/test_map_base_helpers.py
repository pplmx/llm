"""Tests for TokenizedMapDataModule shared setup helpers."""

import torch

from llm.data.datasets.text import TextDataset
from llm.data.modules.text import TextDataModule
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.training.core.config import Config


def _config_with_printable_tokenizer(tmp_path, *, train_path: str, val_path: str | None = None) -> Config:
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(TokenizerFactory.from_printable_corpus(), tokenizer_path)

    config = Config()
    config.data.dataset_path = train_path
    config.data.val_dataset_path = val_path
    config.data.tokenizer_path = str(tokenizer_path)
    config.data.max_seq_len = 32
    return config


def test_setup_tokenized_file_dataset_splits_train_val(tmp_path):
    data_file = tmp_path / "train.txt"
    data_file.write_text("alpha beta gamma delta\n" * 5, encoding="utf-8")

    config = _config_with_printable_tokenizer(tmp_path, train_path=str(data_file))

    module = TextDataModule(config)
    module.setup()

    assert module.tokenizer is not None
    assert module.train_dataset is not None
    assert module.val_dataset is not None
    assert len(module.train_dataset) + len(module.val_dataset) == len(
        TextDataset(str(data_file), module.tokenizer, max_seq_len=32)
    )


def test_setup_tokenized_file_dataset_explicit_val_path(tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("train corpus line\n", encoding="utf-8")
    val_file.write_text("validation corpus line\n", encoding="utf-8")

    config = _config_with_printable_tokenizer(
        tmp_path,
        train_path=str(train_file),
        val_path=str(val_file),
    )

    module = TextDataModule(config)
    module.setup()

    assert len(module.train_dataset) == 1
    assert len(module.val_dataset) == 1
