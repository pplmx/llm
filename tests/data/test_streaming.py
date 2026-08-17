"""Tests for streaming text datasets."""

import pytest
import torch

from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.sources import DedupTextSource, LocalLineTextSource


def test_streaming_dataset_yields_fixed_length_chunks(tmp_path, line_tokenizer):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("abcdefghij\nklmnopqrst\n", encoding="utf-8")

    source = LocalLineTextSource(text_file)
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=line_tokenizer,
        max_seq_len=8,
        rank=0,
        world_size=1,
    )

    samples = list(dataset)
    assert len(samples) >= 2
    assert samples[0]["input_ids"].shape == (8,)
    assert torch.equal(samples[0]["input_ids"], samples[0]["labels"])


def test_streaming_dataset_reset_restarts_corpus(tmp_path, line_tokenizer):
    """``reset()`` clears the resume cursor so the next pass repeats the corpus.

    The training engine calls this when a streaming source is exhausted
    before ``steps_per_epoch``: pretraining cycles the corpus until the
    step budget is met instead of crashing on the exhausted iterator.
    """
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 5, encoding="utf-8")

    source = LocalLineTextSource(text_file)
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=line_tokenizer,
        max_seq_len=8,
        rank=0,
        world_size=1,
    )

    first_pass = list(dataset)
    assert len(first_pass) > 0
    # Exhausted: a second pass without reset yields nothing (cursor at end).
    assert list(dataset) == []

    dataset.reset()
    second_pass = list(dataset)
    assert len(second_pass) == len(first_pass)
    assert torch.equal(first_pass[0]["input_ids"], second_pass[0]["input_ids"])


def test_streaming_dataset_reset_recycles_persisted_dedup_corpus(tmp_path, line_tokenizer):
    """Regression (RIL ISS-064): ``reset()`` on a persistent-dedup source
    must clear the cross-run seen-set so a corpus fully consumed+hashed by a
    prior run can be recycled.

    Without the fix the engine's corpus-cycle path (exhaust -> reset ->
    retry) hit a dedup seen-file that already covered every record and
    raised ``"streaming corpus is empty; nothing to train on"``.
    """
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 5, encoding="utf-8")
    seen_path = tmp_path / "seen.txt"

    source = DedupTextSource(
        LocalLineTextSource(text_file),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=line_tokenizer,
        max_seq_len=8,
        rank=0,
        world_size=1,
    )

    first_pass = list(dataset)
    assert len(first_pass) > 0
    # A fresh source reading the same corpus + seen-file yields nothing.
    fresh = DedupTextSource(
        LocalLineTextSource(text_file),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )
    assert list(fresh.iter_texts()) == []

    # Engine recycle path: reset must restore consumability.
    dataset.reset()
    second_pass = list(dataset)
    assert len(second_pass) == len(first_pass)
    assert torch.equal(first_pass[0]["input_ids"], second_pass[0]["input_ids"])


def test_streaming_dataset_shards_across_ranks(tmp_path, line_tokenizer):
    text_file = tmp_path / "corpus.txt"
    lines = [f"line-{idx}\n" for idx in range(8)]
    text_file.write_text("".join(lines), encoding="utf-8")

    source = LocalLineTextSource(text_file)

    rank0 = list(StreamingTextDataset(source, line_tokenizer, max_seq_len=4, rank=0, world_size=2))
    rank1 = list(StreamingTextDataset(source, line_tokenizer, max_seq_len=4, rank=1, world_size=2))

    assert len(rank0) > 0
    assert len(rank1) > 0


def test_streaming_data_module_dataloader(tmp_path, monkeypatch, line_tokenizer):
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 20, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 3
    config.training.batch_size = 2
    config.optimization.num_workers = 0

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    loader, sampler = data_module.train_dataloader(rank=0, world_size=1)
    assert sampler is None
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 8)


def test_streaming_train_dataloader_forces_single_worker(tmp_path, monkeypatch, line_tokenizer, caplog):
    """Regression: DataLoader workers fork the dataset and lose the resume
    cursor, so streaming training must run single-process for checkpoint
    resume to capture the real position."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 20, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 3
    config.training.batch_size = 2
    config.optimization.num_workers = 4  # default; must be forced to 0

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    loader, sampler = data_module.train_dataloader(rank=0, world_size=1)
    assert sampler is None
    assert loader.num_workers == 0
    assert "num_workers=0" in caplog.text


def test_streaming_data_module_warns_on_inmemory_dedup(tmp_path, monkeypatch, line_tokenizer, caplog):
    """Streaming setup must warn when dedup runs without persisted state:
    checkpoint resume re-creates the source and would re-process records
    that were already deduplicated."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 20, encoding="utf-8")

    config = Config()
    config.data.data_source = "dedup_local"
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 3
    config.training.batch_size = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    assert "write_seen_hashes" in caplog.text


def test_streaming_dataset_rejects_negative_overlap(tmp_path, line_tokenizer):
    """RIL ISS-202: a negative ``overlap`` must fail fast instead of being
    silently treated as "no overlap" (TextDataset already rejects it).

    The old ``__init__`` only checked ``overlap >= max_seq_len``, so a
    negative value sailed through and ``__iter__`` silently treated it as
    zero overlap — hiding the config bug."""
    import pytest

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("abcdefghij\n", encoding="utf-8")
    source = LocalLineTextSource(text_file)

    with pytest.raises(ValueError, match="non-negative"):
        StreamingTextDataset(
            text_source=source,
            tokenizer=line_tokenizer,
            max_seq_len=8,
            rank=0,
            world_size=1,
            overlap=-1,
        )


def test_streaming_skips_undecodable_rows(tmp_path, sample_text_tokenizer):
    """A row the tokenizer cannot encode is skipped (with a warning), not fatal.

    The default character tokenizer is ASCII-only, so a real corpus always
    contains un-encodable rows; one such row must not abort multi-hour
    pretraining (round-76 TASK-189).
    """
    text_file = tmp_path / "corpus.txt"
    text_file.write_text(
        "apple banana cherry\né très café gênt\nfig grape\n",
        encoding="utf-8",
    )
    source = LocalLineTextSource(text_file)
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=sample_text_tokenizer,  # lowercase ASCII + ' .,' only
        max_seq_len=8,
        rank=0,
        world_size=1,
    )
    samples = list(dataset)  # the é-laden line is skipped, not raised
    assert samples
    assert all(torch.equal(s["input_ids"], s["labels"]) for s in samples)


def test_streaming_undecodable_rows_fail_loud_when_not_skipping(tmp_path, sample_text_tokenizer):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("apple banana cherry\nétrès café\nfig grape\n", encoding="utf-8")
    source = LocalLineTextSource(text_file)
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=sample_text_tokenizer,
        max_seq_len=8,
        skip_undecodable=False,
    )
    with pytest.raises(KeyError, match="not found in tokenizer vocabulary"):
        list(dataset)
