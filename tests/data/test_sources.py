"""Tests for pluggable text sources."""

import pytest

from llm.data.sources import LocalLineTextSource, build_text_source
from llm.training.core.config import Config


def test_local_line_text_source_skip(tmp_path):
    text_file = tmp_path / "data.txt"
    text_file.write_text("a\nb\nc\nd\n", encoding="utf-8")

    texts = list(LocalLineTextSource(text_file).iter_texts(skip=2))
    assert texts == ["c", "d"]


def test_local_line_source_fingerprint(tmp_path):
    text_file = tmp_path / "data.txt"
    text_file.write_text("x\n", encoding="utf-8")
    fp = LocalLineTextSource(text_file).source_fingerprint()
    assert fp["type"] == "local"
    assert fp["dataset_path"] == str(text_file.resolve())


def test_hf_source_fingerprint():
    from llm.data.sources import HFStreamTextSource

    source = HFStreamTextSource("wikitext", dataset_config="wikitext-2-raw-v1", text_column="text")
    fp = source.source_fingerprint()
    assert fp["type"] == "hf"
    assert fp["dataset_name"] == "wikitext"


def test_hf_iter_texts_skip_counts_survivors_not_raw_rows(monkeypatch):
    """Regression (deep-dive agent finding): ``HFStreamTextSource.iter_texts(skip)``
    used to delegate ``skip`` to ``datasets.StreamingIterableDataset.skip()``,
    which fast-forwards RAW rows *before* the empty/missing-row filter. But
    ``StreamingTextDataset``'s resume cursor (``state.line_index``) counts
    **survivors** (rows that actually yielded text). With any blank or missing
    ``text_column`` row, resuming at ``skip=survivors_consumed`` started too
    late in raw space and re-emitted the already-consumed tail — silent
    training-data duplication on resume. ``skip`` must count survivors,
    matching :class:`LocalLineTextSource`."""
    import sys
    from types import ModuleType

    from llm.data.sources import HFStreamTextSource

    class FakeDataset:
        # Emulates HF streaming: skip() fast-forwards RAW rows (the historical
        # buggy path); the contract under test must never call it.
        def skip(self, n: int):  # pragma: no cover - unused after the fix
            return self

        def __iter__(self):
            yield {"text": "zero"}
            yield {"text": ""}  # blank -> filtered out
            yield {"text": "two"}
            yield {"text": None}  # missing text -> filtered out
            yield {"text": "four"}
            yield {"text": "five"}

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: FakeDataset()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    source = HFStreamTextSource("demo", text_column="text")
    assert list(source.iter_texts()) == ["zero", "two", "four", "five"]  # 4 survivors

    # Resumed after consuming 3 survivors -> must continue from "five", NOT
    # re-emit the already-consumed "four" (old raw-skip path did exactly that).
    assert list(source.iter_texts(skip=3)) == ["five"]


def test_validate_source_fingerprint_mismatch():
    from llm.data.sources import validate_source_fingerprint

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_source_fingerprint({"type": "local", "dataset_path": "/a"}, {"type": "local", "dataset_path": "/b"})


def test_local_line_text_source(tmp_path):
    text_file = tmp_path / "data.txt"
    text_file.write_text("alpha\n\nbeta\n", encoding="utf-8")

    texts = list(LocalLineTextSource(text_file).iter_texts())
    assert texts == ["alpha", "beta"]


def test_build_text_source_local():
    config = Config()
    config.data.data_source = "local"
    config.data.dataset_path = __file__

    source = build_text_source(config.data)
    assert type(source) is LocalLineTextSource


def test_build_text_source_hf_requires_dataset_name():
    config = Config()
    config.data.data_source = "hf"

    with pytest.raises(ValueError, match="dataset_name"):
        build_text_source(config.data)


def test_hf_stream_text_source_class():
    from llm.data.sources import HFStreamTextSource

    source = HFStreamTextSource("wikitext", dataset_config="wikitext-2-raw-v1", text_column="text")
    assert source.dataset_name == "wikitext"
    assert source.text_column == "text"


# --- An empty local source must be reported, not silently produce nothing ---


def test_empty_local_source_warns(tmp_path, caplog):
    """A corpus with no non-blank lines must WARN on the read that found nothing.

    Otherwise the whole streaming pipeline is a silent no-op: the dataset
    iterates zero rows, and the only signal is the engine's "streaming corpus is
    empty" RuntimeError — raised minutes later, after the run has started, and
    only if ``steps_per_epoch`` fails to fill.
    """
    import logging

    from llm.data.sources import LocalLineTextSource

    corpus = tmp_path / "empty.txt"
    corpus.write_text("", encoding="utf-8")
    source = LocalLineTextSource(corpus)

    with caplog.at_level(logging.WARNING, logger="llm.data.sources"):
        rows = list(source.iter_texts())

    assert rows == []
    assert any("yielded 0 non-blank lines" in record.getMessage() for record in caplog.records), (
        "an empty source must say so"
    )
    assert any(str(corpus) in record.getMessage() for record in caplog.records), (
        "the warning should name the file so the user can check the path"
    )


def test_whitespace_only_local_source_warns(tmp_path, caplog):
    """Blank lines are skipped, so a whitespace-only file is empty too."""
    import logging

    from llm.data.sources import LocalLineTextSource

    corpus = tmp_path / "blank.txt"
    corpus.write_text("   \n\n \t \n", encoding="utf-8")
    source = LocalLineTextSource(corpus)

    with caplog.at_level(logging.WARNING, logger="llm.data.sources"):
        assert list(source.iter_texts()) == []

    assert any("yielded 0 non-blank lines" in record.getMessage() for record in caplog.records)


def test_non_empty_local_source_does_not_warn(tmp_path, caplog):
    """No false positive for a real corpus."""
    import logging

    from llm.data.sources import LocalLineTextSource

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello\nworld\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="llm.data.sources"):
        rows = list(LocalLineTextSource(corpus).iter_texts())

    assert rows == ["hello", "world"]
    assert not any("yielded 0 non-blank lines" in record.getMessage() for record in caplog.records)


def test_empty_source_can_be_appended_to_and_read_later(tmp_path):
    """The guard WARNS rather than rejecting: a corpus may be filled later.

    The source re-reads the file on every ``iter_texts`` call, so rejecting an
    empty file at construction would break the legitimate
    append-then-stream workflow. This pins that the read still works.
    """
    from llm.data.sources import LocalLineTextSource

    corpus = tmp_path / "growing.txt"
    corpus.write_text("", encoding="utf-8")
    source = LocalLineTextSource(corpus)

    assert list(source.iter_texts()) == []

    corpus.write_text("content appears later\n", encoding="utf-8")
    assert list(source.iter_texts()) == ["content appears later"]
