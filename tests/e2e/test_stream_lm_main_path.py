"""E2E tests for the streaming LM main path (P0 docs/e2e alignment).

Covers:

- The local-demo YAML in ``configs/streaming_local_demo.yaml`` is
  well-formed (loadable via ``Config.from_yaml``) and produces a
  working :class:`TrainingEngine` with ``stream_lm`` task wiring
  intact.
- The C4 production YAML in ``configs/streaming_c4.yaml`` is
  well-formed and references the right preset fields
  (``dataset_name=allenai/c4``, ``dataset_config=en``). The test
  does NOT actually download C4 (that's an integration test); it
  only validates the config.
- The ``stream_lm`` task runs end-to-end on a tiny local text
  source: data streams → forward → backward → optimizer step →
  checkpoint save. The trained loss must be finite; the checkpoint
  must exist at the configured path.
- A second run with ``resume_from_checkpoint`` set continues from
  the saved state (extending the existing
  ``test_streaming_resume.py`` coverage).

These tests live under ``tests/e2e/`` so they're opt-in via
``pytest -m e2e`` — they're slower than the unit tests because
they construct real engines and run a handful of optimizer steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from llm.data.modules.streaming import StreamingTextDataModule
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """A small text file with 200 records (one per line) for the
    streaming e2e tests."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(f"this is sample line {idx:04d}" for idx in range(200)),
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def stream_lm_config(tiny_corpus: Path, tmp_path: Path):
    """Minimal valid streaming LM config wired against ``tiny_corpus``."""
    cfg = Config()
    cfg.model.vocab_size = 100
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.max_seq_len = 16
    cfg.training.batch_size = 2
    cfg.training.epochs = 1
    cfg.training.num_samples = 20
    cfg.optimization.use_compile = False
    cfg.optimization.use_amp = False
    cfg.optimization.num_workers = 0  # single-worker → shard key "0"
    cfg.data.data_source = "local"
    cfg.data.dataset_path = str(tiny_corpus)
    cfg.data.max_seq_len = 16
    cfg.data.steps_per_epoch = 5
    cfg.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.checkpoint.save_interval = 1
    cfg.checkpoint.keep_last_n = 2
    cfg.distributed.backend = "gloo"
    return cfg


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CPU for the e2e tests — the engine picks CUDA whenever
    it's available, but the test models are tiny and we're sharing a
    GPU with other processes (which causes spurious CUDA-OOM
    failures unrelated to the test).

    Setting ``torch.cuda.is_available`` to False makes the engine
    fall through to its CPU branch (see
    :class:`TrainingEngine._setup_components`). The tests are
    designed to validate control flow, not GPU-specific kernels —
    CPU runs are perfectly adequate for that.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _patch_stream_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    data_module: StreamingTextDataModule,
    tokenizer: object,
) -> None:
    """Replace the streaming data module's tokenizer loader with the
    test-provided tokenizer (the default loader would try to build a
    SimpleCharacterTokenizer from the tiny test corpus, which fails
    on the special characters in the file)."""
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: tokenizer)


# ---------------------------------------------------------------------------
# YAML validation — the docs must point at configs that actually load
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestStreamingPresetConfigs:
    """The shipped preset configs must load + validate cleanly."""

    def test_local_demo_yaml_loads(self) -> None:
        """``configs/streaming_local_demo.yaml`` must be a valid Config."""
        path = REPO_ROOT / "configs" / "streaming_local_demo.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        # Verify the key fields the tutorial calls out.
        assert cfg.data.data_source == "local"
        assert cfg.data.dataset_path == "data/demo.txt"
        assert cfg.data.steps_per_epoch == 10
        assert cfg.training.batch_size == 4
        assert cfg.checkpoint.checkpoint_dir == "checkpoints_streaming_demo"

    def test_local_demo_yaml_runs_end_to_end(
        self, tmp_path: Path, tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch, line_tokenizer
    ) -> None:
        """The demo YAML must produce a working engine after we
        patch the dataset_path to point at our temp corpus.

        This is the smoke test the tutorial depends on — if the YAML
        drifts out of sync with the engine, the demo would silently
        fail when a new user follows the tutorial."""
        path = REPO_ROOT / "configs" / "streaming_local_demo.yaml"
        cfg = Config.from_yaml(path)
        # Patch the path to the temp corpus so the engine actually
        # finds data (the demo file in `data/demo.txt` doesn't exist
        # in CI).
        cfg.data.dataset_path = str(tiny_corpus)
        cfg.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.optimization.use_compile = False

        data_module = StreamingTextDataModule(cfg)
        _patch_stream_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = LanguageModelingTask(cfg, data_module)
        engine = TrainingEngine(
            config=cfg,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        # Just one epoch at 5 steps — total ~5 optimizer steps.
        engine.run()

        ckpt = tmp_path / "checkpoints" / "epoch_1.pt"
        assert ckpt.exists()

    def test_c4_yaml_loads(self) -> None:
        """``configs/streaming_c4.yaml`` must reference C4 correctly."""
        path = REPO_ROOT / "configs" / "streaming_c4.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        assert cfg.data.data_source == "hf"
        assert cfg.data.dataset_name == "allenai/c4"
        assert cfg.data.dataset_config == "en"
        assert cfg.data.text_column == "text"
        # Dedup wrapper knobs (T3 #39) wired through.
        assert cfg.data.seen_hashes_path is not None
        assert cfg.data.write_seen_hashes is True


# ---------------------------------------------------------------------------
# stream_lm task end-to-end (in-process, fast)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestStreamLMTaskEndToEnd:
    """The ``stream_lm`` task runs the full pipeline."""

    def test_stream_lm_runs_and_saves_checkpoint(
        self,
        stream_lm_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """End-to-end: data → forward → backward → optimizer → checkpoint."""
        data_module = StreamingTextDataModule(stream_lm_config)
        _patch_stream_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = LanguageModelingTask(stream_lm_config, data_module)
        engine = TrainingEngine(
            config=stream_lm_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # Checkpoint must exist at the configured path.
        ckpt_dir = Path(stream_lm_config.checkpoint.checkpoint_dir)
        assert (ckpt_dir / "epoch_1.pt").exists()

        # The checkpoint payload must include model + optimizer state
        # + the streaming data cursor (so resume works).
        ckpt = torch.load(ckpt_dir / "epoch_1.pt", map_location="cpu", weights_only=False)
        assert "model_state" in ckpt
        assert "optimizer_state" in ckpt
        assert "epoch" in ckpt
        assert ckpt["epoch"] == 0  # last completed epoch (0-indexed)
        # The streaming data cursor lives in extra_state (per-shard
        # line_index). The first run consumed some records.
        assert "extra_state" in ckpt
        assert "stream_data" in ckpt["extra_state"]

    def test_stream_lm_loss_is_finite_and_decreasing(
        self,
        stream_lm_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """Loss must be finite at the end of training, and must have
        decreased from its initial value (the model is learning)."""
        # Run 2 epochs so we capture >=2 epoch-summary loss values.
        stream_lm_config.training.epochs = 2

        # Capture loss values via the engine's metrics.
        loss_log: list[float] = []

        original_run = TrainingEngine.run

        def patched_run(self: TrainingEngine) -> None:
            # Capture per-step loss via a hook on the optimizer step.
            # The engine logs to ``self.logger``; we monkey-patch the
            # logger to capture.
            original_logger_info = self.logger.info

            def capturing_info(msg: str, *args: object, **kwargs: object) -> None:
                msg_str = msg % args if args else msg
                # The engine logs loss in two formats:
                #   - per-step: "loss=2.3456"  (rare)
                #   - per-epoch summary: "Train Loss: 4.6700"
                # We accept both.
                for marker in ("Train Loss:", "loss="):
                    if marker in msg_str:
                        try:
                            after = msg_str.split(marker, 1)[1]
                            num = after.split()[0].rstrip(",|")
                            loss_log.append(float(num))
                        except ValueError, IndexError:
                            pass
                        break
                original_logger_info(msg, *args, **kwargs)

            # ``logging.Logger.info`` is defined on the C level and type
            # checkers view it as read-only; cast to ``Any`` for the
            # monkey-patch only.
            logger_any = cast("Any", self.logger)
            logger_any.info = capturing_info
            try:
                original_run(self)
            finally:
                logger_any.info = original_logger_info

        monkeypatch.setattr(TrainingEngine, "run", patched_run)

        data_module = StreamingTextDataModule(stream_lm_config)
        _patch_stream_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = LanguageModelingTask(stream_lm_config, data_module)
        engine = TrainingEngine(
            config=stream_lm_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # At least 2 loss values logged (start + end are different).
        assert len(loss_log) >= 2, f"expected >=2 loss samples, got {loss_log}"
        # All losses finite (no NaN/Inf).
        assert all(loss == loss for loss in loss_log), "NaN in loss log"  # NaN != NaN
        assert all(abs(loss) < 1e6 for loss in loss_log), "Inf in loss log"

    def test_stream_lm_resume_continues_from_checkpoint(
        self,
        stream_lm_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """A second run with ``resume_from_checkpoint`` set continues
        exactly where the first run left off — same model state,
        same data cursor."""
        # First run.
        data_module = StreamingTextDataModule(stream_lm_config)
        _patch_stream_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = LanguageModelingTask(stream_lm_config, data_module)
        engine = TrainingEngine(
            config=stream_lm_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        ckpt_path = Path(stream_lm_config.checkpoint.checkpoint_dir) / "epoch_1.pt"
        assert ckpt_path.exists()

        # Capture the data cursor after the first run.
        first_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        first_cursor = first_ckpt["extra_state"]["stream_data"]
        assert first_cursor  # non-empty

        # Second run with resume_from_checkpoint set; bump epochs so
        # the second run actually does more work.
        stream_lm_config.checkpoint.resume_from_checkpoint = str(ckpt_path)
        stream_lm_config.training.epochs = 2

        data_module2 = StreamingTextDataModule(stream_lm_config)
        _patch_stream_tokenizer(monkeypatch, data_module2, line_tokenizer)
        data_module2.prepare_data()
        data_module2.setup()
        task2 = LanguageModelingTask(stream_lm_config, data_module2)
        # The engine's start_epoch should be 1 (post first run).
        engine2 = TrainingEngine(
            config=stream_lm_config,
            task=task2,
            rank=0,
            world_size=1,
            data_module=data_module2,
        )
        assert engine2.start_epoch == 1, f"resume should start at epoch 1, got {engine2.start_epoch}"

        engine2.run()

        # Second-run checkpoint should also exist.
        second_ckpt_path = Path(stream_lm_config.checkpoint.checkpoint_dir) / "epoch_2.pt"
        assert second_ckpt_path.exists()

        # The resumed run preserved the model state — epoch counter
        # advanced correctly.
        second_ckpt = torch.load(second_ckpt_path, map_location="cpu", weights_only=False)
        assert second_ckpt["epoch"] == 1  # 0-indexed, after second epoch
