"""Tests for the train_bpe CLI entry point (0% baseline coverage)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from llm.tokenization.train_bpe import main


def _write_text_file(path: str, lines: list[str]) -> str:
    """Write a simple text file for training."""
    with Path(path).open("w") as f:
        f.writelines(lines)
    return path


@pytest.fixture
def mock_argv():
    """Fixture to patch sys.argv for CLI testing."""

    def _patch(args: list[str]):
        return patch("sys.argv", args)

    return _patch


def test_train_bpe_basic(tmp_path):
    """Train a BPE tokenizer with default settings."""
    text_file = _write_text_file(str(tmp_path / "corpus.txt"), ["hello world\n", "test data\n"])
    output_path = str(tmp_path / "tokenizer.json")

    test_args = [
        "train_bpe",
        "--files",
        text_file,
        "--output",
        output_path,
        "--vocab_size",
        "50",
    ]
    with patch("sys.argv", test_args):
        main()

    assert (tmp_path / "tokenizer.json").exists()


def test_train_bpe_minimal_vocab(tmp_path):
    """Train with a very small vocabulary."""
    text_file = _write_text_file(str(tmp_path / "small.txt"), ["a b c d e\n", "f g h i j\n"])
    output_path = str(tmp_path / "small.json")

    test_args = [
        "train_bpe",
        "--files",
        text_file,
        "--output",
        output_path,
        "--vocab_size",
        "10",
        "--min_frequency",
        "1",
    ]
    with patch("sys.argv", test_args):
        main()

    assert (tmp_path / "small.json").exists()


def test_train_bpe_custom_special_tokens(tmp_path):
    """Train with custom special tokens."""
    text_file = _write_text_file(str(tmp_path / "custom.txt"), ["apple banana cherry\n"])
    output_path = str(tmp_path / "custom.json")

    test_args = [
        "train_bpe",
        "--files",
        text_file,
        "--output",
        output_path,
        "--vocab_size",
        "30",
        "--special_tokens",
        "[UNK]",
        "[BOS]",
        "[EOS]",
    ]
    with patch("sys.argv", test_args):
        main()

    assert (tmp_path / "custom.json").exists()


def test_train_bpe_default_special_tokens_include_unk(tmp_path):
    """The CLI default special-token list must match the library default and
    include the BPE model's declared ``<unk>`` token (eval deep-dive F4). It
    used to default to ``["[UNK]", ...]`` — the ``<unk>`` UNK token was
    declared but never added to the vocab, so a default CLI-trained tokenizer
    reported ``token_to_id("<unk>") is None`` and produced a different
    UNK/id layout than ``BPETokenizer.train`` on the same input."""
    text_file = _write_text_file(str(tmp_path / "unk.txt"), ["the quick brown fox\n", "jumps over the lazy dog\n"])
    output_path = str(tmp_path / "unk.json")
    test_args = [
        "train_bpe",
        "--files",
        text_file,
        "--output",
        output_path,
        "--vocab_size",
        "40",
        "--min_frequency",
        "1",
    ]
    with patch("sys.argv", test_args):
        main()

    from llm.tokenization.bpe_tokenizer import BPETokenizer

    loaded = BPETokenizer.load(str(output_path))
    assert loaded.get_vocab().get("<unk>") is not None  # the declared UNK must be in the vocab


def test_train_bpe_skips_missing_files(tmp_path):
    """Warning is printed for missing files but training continues if at least one valid file exists."""
    valid_file = _write_text_file(str(tmp_path / "valid.txt"), ["only this one\n"])
    output_path = str(tmp_path / "partial.json")

    test_args = [
        "train_bpe",
        "--files",
        str(tmp_path / "nonexistent.txt"),
        valid_file,
        "--output",
        output_path,
        "--vocab_size",
        "20",
    ]
    with patch("sys.argv", test_args):
        main()

    assert (tmp_path / "partial.json").exists()


def test_train_bpe_no_valid_files_exits(tmp_path):
    """When no training files exist, the CLI exits with code 1."""
    output_path = str(tmp_path / "empty.json")

    test_args = [
        "train_bpe",
        "--files",
        str(tmp_path / "missing1.txt"),
        str(tmp_path / "missing2.txt"),
        "--output",
        output_path,
        "--vocab_size",
        "20",
    ]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
