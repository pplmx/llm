"""Tests for streaming dataset checkpoint resume."""

import pytest

from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.sources import LocalLineTextSource
from llm.data.stream_state import StreamDataState


class _LineTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) % 50 for char in text]

    @property
    def pad_token_id(self) -> int:
        return 0


def test_streaming_dataset_resumes_from_saved_state(tmp_path):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("abcdefghij\nklmnopqrst\nuvwxyzab\n", encoding="utf-8")

    source = LocalLineTextSource(text_file)
    state = StreamDataState()

    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=_LineTokenizer(),
        max_seq_len=8,
        rank=0,
        world_size=1,
        stream_data_state=state,
    )

    first_run = list(dataset)
    assert len(first_run) > 0
    saved = state.to_dict()
    assert saved["0"]["line_index"] > 0

    resumed_state = StreamDataState.from_dict(saved)
    resumed_dataset = StreamingTextDataset(
        text_source=LocalLineTextSource(text_file),
        tokenizer=_LineTokenizer(),
        max_seq_len=8,
        rank=0,
        world_size=1,
        stream_data_state=resumed_state,
    )

    second_run = list(resumed_dataset)
    assert second_run == []


def test_streaming_module_checkpoint_roundtrip(tmp_path, monkeypatch):
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 30, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2
    config.optimization.num_workers = 0

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: _LineTokenizer())
    data_module.setup()

    loader, _ = data_module.train_dataloader(rank=0, world_size=1)
    _ = next(iter(loader))

    checkpoint_state = data_module.get_checkpoint_state()
    assert "stream_data" in checkpoint_state
    assert checkpoint_state["stream_source"]["type"] == "local"

    data_module.load_checkpoint_state(checkpoint_state)
    assert data_module.stream_data_state.shards["0"].line_index > 0


def test_streaming_module_rejects_source_fingerprint_mismatch(tmp_path, monkeypatch):
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello\n" * 10, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: _LineTokenizer())
    data_module.setup()

    bad_state = {
        "stream_data": {"0": {"line_index": 3, "token_buffer": []}},
        "stream_source": {"type": "local", "dataset_path": "/other/path.txt"},
    }

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        data_module.load_checkpoint_state(bad_state)
