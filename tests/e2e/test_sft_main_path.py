"""E2E tests for the SFT main path (Main Path #2 docs/e2e alignment).

Mirrors :mod:`tests.e2e.test_stream_lm_main_path` for the
``llm-train sft`` main path — the same tutorial-alignment pattern
applied to Supervised Fine-Tuning.

Covers:

- ``configs/sft_local_demo.yaml`` is well-formed (loadable via
  :meth:`Config.from_yaml`) and references the right Alpaca-style
  JSONL path / tokenizer / batch size.
- The ``sft`` task runs end-to-end on a tiny JSONL: data loads →
  Alpaca template → tokenize → forward → backward → optimizer step →
  checkpoint save. The trained loss must be finite; the checkpoint
  must exist at the configured path.
- The ``sft`` task with ``peft_method="lora"`` saves an adapter
  sidecar at the configured ``peft_save_path`` (the
  PEFT-save-load bridge wired in T2 PEFT #48).

These tests live under ``tests/e2e/`` so they're opt-in via
``pytest -m e2e`` — slower than unit tests because they construct
real engines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from llm.data.modules.sft import SFTDataModule
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.sft_task import SFTTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_alpaca_jsonl(tmp_path: Path) -> Path:
    """A small Alpaca-style JSONL with 30 records (10 per template x 3)."""
    data = [
        {"instruction": "Greet me.", "input": "", "output": "Hello!"},
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Translate to French.", "input": "hi", "output": "salut"},
    ] * 10
    path = tmp_path / "sft_data.jsonl"
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture
def sft_config(tiny_alpaca_jsonl: Path, tmp_path: Path):
    """Minimal valid SFT config wired against ``tiny_alpaca_jsonl``."""
    cfg = Config()
    cfg.model.vocab_size = 100
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.max_seq_len = 32
    cfg.training.batch_size = 2
    cfg.training.epochs = 1
    cfg.training.num_samples = 30
    cfg.training.max_steps = 20
    cfg.training.warmup_epochs = 0
    cfg.optimization.use_compile = False
    cfg.optimization.use_amp = False
    cfg.optimization.num_workers = 0
    cfg.data.dataset_path = str(tiny_alpaca_jsonl)
    cfg.data.max_seq_len = 32
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
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _patch_sft_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    data_module: SFTDataModule,
    tokenizer: object,
) -> None:
    """Replace the SFT data module's tokenizer loader with the
    test-provided tokenizer (the default loader would try to build a
    SimpleCharacterTokenizer from the Alpaca template text, which has
    characters not in a tiny test vocab).

    SFT/DPO data modules use :meth:`setup_tokenizer` from
    :class:`TokenizedMapDataModule` — we patch that.
    """
    monkeypatch.setattr(data_module, "setup_tokenizer", lambda: setattr(data_module, "tokenizer", tokenizer))


# ---------------------------------------------------------------------------
# YAML validation — the docs must point at configs that actually load
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSFTPresetConfigs:
    """The shipped SFT preset configs must load + validate cleanly."""

    def test_local_demo_yaml_loads(self) -> None:
        """``configs/sft_local_demo.yaml`` must be a valid Config."""
        path = REPO_ROOT / "configs" / "sft_local_demo.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        # Verify the key fields the tutorial calls out.
        assert cfg.data.dataset_path == "data/sft_demo.jsonl"
        assert cfg.data.tokenizer_type == "simple"
        assert cfg.training.batch_size == 4
        assert cfg.checkpoint.checkpoint_dir == "checkpoints_sft_demo"

    def test_alpaca_yaml_loads(self) -> None:
        """``configs/sft_alpaca.yaml`` must reference Alpaca + GPT-2."""
        path = REPO_ROOT / "configs" / "sft_alpaca.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        # GPT-2 BPE for tokenizer (Alpaca is normally tokenized with BPE).
        assert cfg.data.tokenizer_type == "hf"
        assert cfg.data.tokenizer_path == "gpt2"
        # Alpaca JSONL path matches the SFTDataset expectation.
        assert cfg.data.dataset_path == "data/alpaca.jsonl"
        # Production-shape model.
        assert cfg.model.hidden_size == 256
        assert cfg.model.num_layers == 6


# ---------------------------------------------------------------------------
# sft task end-to-end (in-process, fast)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSFTTaskEndToEnd:
    """The ``sft`` task runs the full pipeline on a tiny Alpaca JSONL."""

    def test_sft_runs_and_saves_checkpoint(
        self,
        sft_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """End-to-end: JSONL → Alpaca template → tokenize → forward →
        backward → optimizer → checkpoint."""
        data_module = SFTDataModule(sft_config)
        _patch_sft_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(sft_config, data_module)
        engine = TrainingEngine(
            config=sft_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # Checkpoint must exist at the configured path (v2 split layout).
        ckpt_dir = Path(sft_config.checkpoint.checkpoint_dir)
        assert (ckpt_dir / "epoch_1.safetensors").exists()
        assert (ckpt_dir / "epoch_1.meta.json").exists()
        assert (ckpt_dir / "epoch_1.extra_state.pt").exists()

        # The training-state sidecar must include optimizer state + epoch.
        extra_state_blob = torch.load(
            ckpt_dir / "epoch_1.extra_state.pt", map_location="cpu", weights_only=False
        )
        assert "optimizer_state" in extra_state_blob
        # ``epoch`` lives in the .meta.json sidecar.
        meta = json.loads((ckpt_dir / "epoch_1.meta.json").read_text())
        assert "epoch" in meta

    def test_sft_loss_is_finite(
        self,
        sft_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """Loss must be finite at the end of training (no NaN/Inf)."""
        # Capture loss via the engine's logger.
        loss_log: list[float] = []
        original_run = TrainingEngine.run

        def patched_run(self: TrainingEngine) -> None:
            original_logger_info = self.logger.info

            def capturing_info(msg: str, *args: object, **kwargs: object) -> None:
                msg_str = msg % args if args else msg
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

        data_module = SFTDataModule(sft_config)
        _patch_sft_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(sft_config, data_module)
        engine = TrainingEngine(
            config=sft_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # At least one loss value logged.
        assert len(loss_log) >= 1, f"expected >=1 loss sample, got {loss_log}"
        # All losses finite (no NaN/Inf).
        assert all(loss == loss for loss in loss_log), "NaN in loss log"
        assert all(abs(loss) < 1e6 for loss in loss_log), "Inf in loss log"

    def test_sft_with_peft_saves_adapter_sidecar(
        self,
        sft_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
        tmp_path: Path,
    ) -> None:
        """SFT + LoRA: ``PEFTAdapterCheckpointCallback`` writes the
        sidecar at the configured ``peft_save_path``. This is the
        main-path integration with T2 PEFT #48."""
        from llm.core.lora import apply_lora

        # Apply LoRA to a fresh model — the engine's build_model is
        # mocked below so we don't go through PEFT_REGISTRY here.
        # Simulate "peft_method=lora" by writing the sidecar manually.
        sft_config.training.peft_method = "lora"
        sft_config.training.peft_kwargs = {"rank": 4, "alpha": 8.0}
        adapter_path = tmp_path / "peft_adapter_lora.bin"
        sft_config.training.peft_save_path = str(adapter_path)

        # Run the SFT engine (without registry-based LoRA application —
        # the engine only auto-applies when `build_model` is on the
        # PEFT path, which the SFTTask inherits).
        data_module = SFTDataModule(sft_config)
        _patch_sft_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()

        # Use SFTTask.build_model so the peft_method branch fires.
        task = SFTTask(sft_config, data_module)
        # Force-apply LoRA so the adapter is present in the model
        # (the engine's build_model path uses apply_peft via the
        # registry; we just call the module-level helper directly to
        # ensure the adapter tensors exist for the save callback).
        engine = TrainingEngine(
            config=sft_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        # Apply LoRA to the engine's model so the callback has
        # adapter tensors to save.
        apply_lora(engine.model, rank=4, alpha=8.0)
        engine.run()

        # The PEFT callback fires on_train_end and writes the sidecar.
        assert adapter_path.exists(), "PEFT adapter sidecar not written by PEFTAdapterCheckpointCallback"
        # The sidecar must be a valid save_peft envelope.
        payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        from llm.core.peft.checkpoint import PEFT_CHECKPOINT_FORMAT_VERSION

        assert payload["format_version"] == PEFT_CHECKPOINT_FORMAT_VERSION
        assert payload["method_name"] == "lora"
