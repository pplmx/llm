"""Tests for streaming dataset checkpoint resume."""

import sys
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.sources import DedupTextSource, LocalLineTextSource
from llm.data.stream_state import StreamDataState

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _enc(line_tokenizer, text: str) -> list[int]:
    return line_tokenizer.encode(text)


def test_streaming_dataset_resumes_from_saved_state(tmp_path, line_tokenizer):
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("abcdefghij\nklmnopqrst\nuvwxyzab\n", encoding="utf-8")

    source = LocalLineTextSource(text_file)
    state = StreamDataState()

    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=line_tokenizer,
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
        tokenizer=line_tokenizer,
        max_seq_len=8,
        rank=0,
        world_size=1,
        stream_data_state=resumed_state,
    )

    second_run = list(resumed_dataset)
    assert second_run == []


def test_streaming_dataset_inmemory_dedup_resume_does_not_repeat_consumed_tail(tmp_path, line_tokenizer):
    """Regression (RIL ISS-088): with in-memory dedup (the default — no
    seen-hashes file), the streaming resume cursor counts *survivors* of the
    dedup source. The ``iter_texts(skip=...)`` of a fresh dedup source must
    therefore skip survivors too — otherwise a resumed pass re-processes the
    tail of the already-consumed window as fresh data."""
    text_file = tmp_path / "corpus.txt"
    # survivors: xxxx, aaaa, bbbb, cccc  (2nd 'aaaa' is a duplicate)
    text_file.write_text("xxxx\naaaa\nbbbb\naaaa\ncccc\n", encoding="utf-8")

    source = DedupTextSource(LocalLineTextSource(text_file))
    state = StreamDataState()
    dataset = StreamingTextDataset(
        text_source=source,
        tokenizer=line_tokenizer,
        max_seq_len=4,
        rank=0,
        world_size=1,
        stream_data_state=state,
    )

    it = iter(dataset)
    assert next(it)["input_ids"].tolist() == _enc(line_tokenizer, "xxxx")
    assert next(it)["input_ids"].tolist() == _enc(line_tokenizer, "aaaa")
    saved = state.to_dict()

    # Resume on a fresh source instance (as after a process restart).
    resumed = list(
        StreamingTextDataset(
            text_source=DedupTextSource(LocalLineTextSource(text_file)),
            tokenizer=line_tokenizer,
            max_seq_len=4,
            rank=0,
            world_size=1,
            stream_data_state=StreamDataState.from_dict(saved),
        )
    )
    chunks = [ch["input_ids"].tolist() for ch in resumed]

    # The resumed continuation must be exactly bbbb, cccc — the 'aaaa'
    # duplicate (a window survivor seen again after the resume point) must
    # NOT be re-emitted; the old raw-record skip re-yielded it as fresh.
    assert chunks == [_enc(line_tokenizer, "bbbb"), _enc(line_tokenizer, "cccc")]


def test_streaming_module_checkpoint_roundtrip(tmp_path, monkeypatch, line_tokenizer):
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
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    loader, _ = data_module.train_dataloader(rank=0, world_size=1)
    _ = next(iter(loader))

    checkpoint_state = data_module.get_checkpoint_state()
    assert "stream_data" in checkpoint_state
    assert checkpoint_state["stream_source"]["type"] == "local"
    assert checkpoint_state["stream_world_size"] == 1

    data_module.load_checkpoint_state(checkpoint_state)
    assert data_module.stream_data_state.shards["0"].line_index > 0


def test_streaming_module_rejects_source_fingerprint_mismatch(tmp_path, monkeypatch, line_tokenizer):
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello\n" * 10, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    bad_state = {
        "stream_data": {"0": {"line_index": 3, "token_buffer": []}},
        "stream_source": {"type": "local", "dataset_path": "/other/path.txt"},
    }

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        data_module.load_checkpoint_state(bad_state)


def test_streaming_module_rejects_world_size_mismatch(tmp_path, monkeypatch, line_tokenizer):
    """Resuming a checkpoint saved under a different rank layout would silently
    duplicate or drop data (shard cursors depend on the ``line_idx % num_shards``
    arithmetic), so the module must refuse it instead of guessing."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 10, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()
    data_module.train_dataloader(rank=0, world_size=1)

    foreign_state = {
        "stream_data": {"0": {"line_index": 3, "token_buffer": []}},
        "stream_world_size": 4,
        "stream_source": data_module.get_checkpoint_state()["stream_source"],
    }

    with pytest.raises(ValueError, match="world_size"):
        data_module.load_checkpoint_state(foreign_state)


def test_streaming_module_world_size_guard_order_independent(tmp_path, monkeypatch, line_tokenizer):
    """Regression (RIL ISS-204): the world-size resume guard must not depend
    on call order. The old guard compared ``saved_world_size`` only when
    ``self._world_size`` had ALREADY been set — i.e. only when
    ``train_dataloader`` happened to run before ``load_checkpoint_state``.
    ``load_extra_state`` calls the module's ``load_checkpoint_state`` without
    any ordering guarantee, so a foreign-world-size checkpoint loaded before
    ``train_dataloader`` silently bypassed the mismatch check and every rank
    would re-train the corpus with wrong shard arithmetic."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 10, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()

    foreign_state = {
        "stream_data": {"0": {"line_index": 3, "token_buffer": []}},
        "stream_world_size": 4,
        "stream_source": data_module.get_checkpoint_state()["stream_source"],
    }

    # Loaded BEFORE train_dataloader sets _world_size, so the world size is
    # not yet known here — the module must defer the mismatch check, not
    # silently skip it.
    data_module.load_checkpoint_state(foreign_state)
    assert data_module._pending_world_size == 4

    # The guard must fire at the first dataloader call — never silently pass.
    with pytest.raises(ValueError, match="world_size"):
        data_module.train_dataloader(rank=0, world_size=1)


def test_streaming_module_accepts_legacy_checkpoint_without_world_size(tmp_path, monkeypatch, line_tokenizer):
    """Checkpoints written before the world-size stamp must keep loading."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world\n" * 10, encoding="utf-8")

    config = Config()
    config.data.dataset_path = str(text_file)
    config.data.max_seq_len = 8
    config.data.steps_per_epoch = 2

    data_module = StreamingTextDataModule(config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.setup()
    data_module.train_dataloader(rank=0, world_size=1)

    legacy_state = {
        "stream_data": {"0": {"line_index": 3, "token_buffer": []}},
        "stream_source": data_module.get_checkpoint_state()["stream_source"],
    }

    data_module.load_checkpoint_state(legacy_state)
    assert data_module.stream_data_state.shards["0"].line_index == 3


def test_streaming_resume_restores_every_rank_shard(tmp_path, monkeypatch, line_tokenizer):
    """A checkpoint holding ALL rank shards lets every rank continue exactly
    where it stopped — the regression guarded against rank>=1 silently
    restarting from line 0 (only rank 0 persists the checkpoint file)."""
    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config

    lines = [f"line{i}" for i in range(16)]
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_module() -> StreamingTextDataModule:
        config = Config()
        config.data.dataset_path = str(text_file)
        config.data.max_seq_len = 4
        config.data.steps_per_epoch = 2
        config.optimization.num_workers = 0
        config.training.batch_size = 2
        module = StreamingTextDataModule(config)
        monkeypatch.setattr(module, "_load_tokenizer", lambda: line_tokenizer)
        module.setup()
        return module

    def reference_tokens(rank: int, world_size: int) -> list[int]:
        module = make_module()
        loader, _ = module.train_dataloader(rank=rank, world_size=world_size)
        out: list[int] = []
        for batch in loader:
            out.extend(batch["input_ids"].flatten().tolist())
        return out

    # Phase 1: two ranks consume part of the corpus.
    consumed: dict[int, list[int]] = {}
    phase1_states = {}
    for rank in (0, 1):
        module = make_module()
        loader, _ = module.train_dataloader(rank=rank, world_size=2)
        iterator = iter(loader)
        for _ in range(2):
            consumed[rank] = consumed.get(rank, []) + next(iterator)["input_ids"].flatten().tolist()
        phase1_states[rank] = module.get_checkpoint_state()

    # Rank 0 persists a single merged checkpoint (all_gather equivalent).
    merged_stream_data = {**phase1_states[0]["stream_data"], **phase1_states[1]["stream_data"]}
    checkpoint = {**phase1_states[0], "stream_data": merged_stream_data}
    assert set(checkpoint["stream_data"]) == {"0", "1"}
    assert checkpoint["stream_world_size"] == 2

    # Phase 2: resume both ranks from the same checkpoint file.
    for rank in (0, 1):
        module = make_module()
        loader, _ = module.train_dataloader(rank=rank, world_size=2)
        module.load_checkpoint_state(checkpoint)
        resumed: list[int] = []
        for batch in loader:
            resumed.extend(batch["input_ids"].flatten().tolist())

        full = reference_tokens(rank, 2)
        assert resumed == full[len(consumed[rank]) :], (
            f"rank {rank} resumed stream diverges from its pre-checkpoint continuation"
        )
        if consumed[rank]:
            assert resumed, f"rank {rank} lost its remaining corpus after resume"


class _FixedTokenizerModule:
    """Mixin marker so the spawn worker can build a module offline."""


def _gather_worker(rank: int, world_size: int, corpus_path: str, init_file: str) -> None:
    """Worker for the two-process gloo checkpoint-gather test.

    Runs in a spawned child process, so everything it needs must be
    importable (no fixtures / monkeypatch). Each rank consumes one batch,
    then calls ``get_checkpoint_state`` — the collective under test — and
    rank 0 asserts the merged result carries every rank's shard.
    """
    import os

    import torch.distributed as dist

    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from llm.data.modules.streaming import StreamingTextDataModule
    from llm.training.core.config import Config
    from tests.support.tokenizers import LineTokenizer

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    try:
        config = Config()
        config.data.dataset_path = corpus_path
        config.data.max_seq_len = 4
        config.data.steps_per_epoch = 2
        config.optimization.num_workers = 0
        config.training.batch_size = 2

        module = StreamingTextDataModule(config)
        module._load_tokenizer = lambda: LineTokenizer()
        module.setup()
        loader, _ = module.train_dataloader(rank=rank, world_size=world_size)
        _ = next(iter(loader))

        local_keys = set(module.stream_data_state.to_dict())
        assert local_keys == {str(rank)}, "each rank tracks only its own shard locally"

        state = module.get_checkpoint_state()
        assert state is not None
        assert state["stream_world_size"] == world_size
        assert set(state["stream_data"]) == {"0", "1"}, (
            f"rank {rank} checkpoint must merge every rank's shard, got {sorted(state['stream_data'])}"
        )
        own = state["stream_data"][str(rank)]
        assert own["line_index"] > 0, f"rank {rank} lost its own cursor in the merged state"

        # Silence the unused-import lint for os (kept for env debugging).
        _ = os
    finally:
        dist.destroy_process_group()


def _gloo_loopback_usable() -> bool:
    """Probe whether gloo can initialise a 1-rank group on this machine.

    Sandboxed/offline environments may block the socket setup gloo needs;
    the two-process test then skips instead of failing spuriously.
    """
    import tempfile

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        return False
    with tempfile.NamedTemporaryFile(delete=False) as probe:
        init_file = probe.name
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method=f"file://{init_file}")
    except RuntimeError, ValueError, OSError:
        return False
    else:
        dist.destroy_process_group()
        return True


@pytest.mark.slow
def test_streaming_module_gathers_all_shards_across_ranks(tmp_path):
    """``get_checkpoint_state`` must all-gather shard cursors so the single
    checkpoint file persisted by rank 0 contains every rank's resume cursor
    (regression: non-zero ranks silently restarted from line 0 on resume)."""
    if not _gloo_loopback_usable():
        pytest.skip("gloo cannot initialise a loopback process group in this environment")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(f"line{i}" for i in range(16)) + "\n", encoding="utf-8")
    init_file = tmp_path / "gloo_init"

    mp.spawn(
        _gather_worker,
        args=(2, str(corpus), str(init_file)),
        nprocs=2,
        join=True,
    )
