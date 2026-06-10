"""Tests for streaming text datasets."""

import torch

from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.sources import LocalLineTextSource


class _LineTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) % 50 for char in text]

    @property
    def pad_token_id(self) -> int:
        return 0


def test_streaming_dataset_yields_fixed_length_chunks(tmp_path):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("abcdefghij\nklmnopqrst\n", encoding="utf-8")

    source = LocalLineTextSource(text_file)
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=_LineTokenizer(),
        max_seq_len=8,
        rank=0,
        world_size=1,
    )

    samples = list(dataset)
    assert len(samples) >= 2
    assert samples[0]["input_ids"].shape == (8,)
    assert torch.equal(samples[0]["input_ids"], samples[0]["labels"])


def test_streaming_dataset_shards_across_ranks(tmp_path):
    text_file = tmp_path / "corpus.txt"
    lines = [f"line-{idx}\n" for idx in range(8)]
    text_file.write_text("".join(lines), encoding="utf-8")

    source = LocalLineTextSource(text_file)
    tokenizer = _LineTokenizer()

    rank0 = list(StreamingTextDataset(source, tokenizer, max_seq_len=4, rank=0, world_size=2))
    rank1 = list(StreamingTextDataset(source, tokenizer, max_seq_len=4, rank=1, world_size=2))

    assert len(rank0) > 0
    assert len(rank1) > 0


def test_streaming_data_module_dataloader(tmp_path, monkeypatch):
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
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: _LineTokenizer())
    data_module.setup()

    loader, sampler = data_module.train_dataloader(rank=0, world_size=1)
    assert sampler is None
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 8)
