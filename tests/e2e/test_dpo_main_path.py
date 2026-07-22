"""E2E tests for the DPO main path (Main Path #2 docs/e2e alignment).

Mirrors :mod:`tests.e2e.test_stream_lm_main_path` for the
``llm-train dpo`` main path — the same tutorial-alignment pattern
applied to Direct Preference Optimization.

Covers:

- ``configs/dpo_local_demo.yaml`` is well-formed (loadable via
  :meth:`Config.from_yaml`) and references the right chosen /
  rejected JSONL path / tokenizer / DPO beta / batch size.
- ``configs/dpo_ultrafeedback.yaml`` is well-formed and points at
  the production UltraFeedback preset.
- The ``dpo`` task runs end-to-end on a tiny preference JSONL:
  data loads → tokenize → policy forward + reference forward →
  DPO loss → backward → optimizer step → checkpoint save. The
  trained loss must be finite.
- The two-model build (policy + reference) works: both models
  start from the same weights (verified by comparing state_dicts
  before any training).

These tests live under ``tests/e2e/`` so they're opt-in via
``pytest -m e2e`` — slower than unit tests because they construct
real engines with two models in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from llm.data.modules.dpo import DPODataModule
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.dpo_task import DPOTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_dpo_jsonl(tmp_path: Path) -> Path:
    """A small preference JSONL with 30 records (10 pairs x 3 patterns)."""
    data = [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
        {"prompt": "Largest planet?", "chosen": "Jupiter", "rejected": "Earth"},
    ] * 10
    path = tmp_path / "dpo_data.jsonl"
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture
def dpo_config(tiny_dpo_jsonl: Path, tmp_path: Path):
    """Minimal valid DPO config wired against ``tiny_dpo_jsonl``.

    DPO holds policy + reference in memory, so batch_size stays tiny
    and ``num_samples`` is small — the focus is control flow, not
    DPO math.
    """
    cfg = Config()
    cfg.model.vocab_size = 100
    cfg.model.hidden_size = 16
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.max_seq_len = 32
    cfg.training.batch_size = 2
    cfg.training.epochs = 1
    cfg.training.num_samples = 30
    cfg.training.max_steps = 10
    cfg.training.warmup_epochs = 0
    cfg.training.dpo_beta = 0.1
    cfg.optimization.use_compile = False
    cfg.optimization.use_amp = False
    cfg.optimization.num_workers = 0
    cfg.data.dataset_path = str(tiny_dpo_jsonl)
    cfg.data.max_seq_len = 32
    cfg.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.checkpoint.save_interval = 1
    cfg.checkpoint.keep_last_n = 2
    cfg.distributed.backend = "gloo"
    return cfg


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CPU — the engine picks CUDA whenever it's available,
    but the test models are tiny and we're sharing a GPU with
    other processes (which causes spurious CUDA-OOM failures
    unrelated to the test).
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _patch_dpo_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    data_module: DPODataModule,
    tokenizer: object,
) -> None:
    """Replace the DPO data module's tokenizer loader with the
    test-provided tokenizer.

    SFT/DPO data modules use :meth:`setup_tokenizer` from
    :class:`TokenizedMapDataModule` — we patch that.
    """
    monkeypatch.setattr(data_module, "setup_tokenizer", lambda: setattr(data_module, "tokenizer", tokenizer))


# ---------------------------------------------------------------------------
# YAML validation — the docs must point at configs that actually load
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDPOPresetConfigs:
    """The shipped DPO preset configs must load + validate cleanly."""

    def test_local_demo_yaml_loads(self) -> None:
        """``configs/dpo_local_demo.yaml`` must be a valid Config."""
        path = REPO_ROOT / "configs" / "dpo_local_demo.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        # Verify the key fields the tutorial calls out.
        assert cfg.data.dataset_path == "data/dpo_demo.jsonl"
        assert cfg.data.tokenizer_type == "simple"
        assert cfg.training.batch_size == 2  # smaller — DPO doubles memory
        assert cfg.training.dpo_beta == 0.1
        assert cfg.checkpoint.checkpoint_dir == "checkpoints_dpo_demo"

    def test_ultrafeedback_yaml_loads(self) -> None:
        """``configs/dpo_ultrafeedback.yaml`` must reference
        UltraFeedback + GPT-2 + DPO beta."""
        path = REPO_ROOT / "configs" / "dpo_ultrafeedback.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = Config.from_yaml(path)
        # GPT-2 BPE for tokenizer (UltraFeedback is normally tokenized with BPE).
        assert cfg.data.tokenizer_type == "hf"
        assert cfg.data.tokenizer_path == "gpt2"
        # UltraFeedback JSONL path.
        assert cfg.data.dataset_path == "data/ultrafeedback.jsonl"
        # DPO beta + production-shape model.
        assert cfg.training.dpo_beta == 0.1
        assert cfg.model.hidden_size == 256


# ---------------------------------------------------------------------------
# dpo task end-to-end (in-process, fast)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDPOTaskEndToEnd:
    """The ``dpo`` task runs the full pipeline on a tiny preference JSONL."""

    def test_dpo_runs_and_saves_checkpoint(
        self,
        dpo_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """End-to-end: JSONL → tokenize → policy + ref forward →
        DPO loss → backward → optimizer → checkpoint."""
        data_module = DPODataModule(dpo_config)
        _patch_dpo_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = DPOTask(dpo_config, data_module)
        engine = TrainingEngine(
            config=dpo_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # Checkpoint must exist at the configured path (v2 split layout).
        ckpt_dir = Path(dpo_config.checkpoint.checkpoint_dir)
        assert (ckpt_dir / "epoch_1.safetensors").exists()
        assert (ckpt_dir / "epoch_1.meta.json").exists()
        assert (ckpt_dir / "epoch_1.extra_state.pt").exists()

        # The training-state sidecar must include optimizer state + epoch.
        extra_state_blob = torch.load(ckpt_dir / "epoch_1.extra_state.pt", map_location="cpu", weights_only=False)
        assert "optimizer_state" in extra_state_blob
        # ``epoch`` lives in the .meta.json sidecar.
        meta = json.loads((ckpt_dir / "epoch_1.meta.json").read_text())
        assert "epoch" in meta

    def test_dpo_loss_is_finite(
        self,
        dpo_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """Loss must be finite at the end of training (no NaN/Inf)."""
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

        data_module = DPODataModule(dpo_config)
        _patch_dpo_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = DPOTask(dpo_config, data_module)
        engine = TrainingEngine(
            config=dpo_config,
            task=task,
            rank=0,
            world_size=1,
            data_module=data_module,
        )
        engine.run()

        # At least one loss value logged.
        assert len(loss_log) >= 1, f"expected >=1 loss sample, got {loss_log}"
        # All losses finite (no NaN/Inf). DPO loss can start near
        # log(2) ≈ 0.693 and decrease toward 0; we just require
        # finiteness here, not convergence.
        assert all(loss == loss for loss in loss_log), "NaN in loss log"
        assert all(abs(loss) < 1e6 for loss in loss_log), "Inf in loss log"

    def test_dpo_two_model_build_initializes_from_same_weights(
        self,
        dpo_config: Config,
        monkeypatch: pytest.MonkeyPatch,
        line_tokenizer,
    ) -> None:
        """``DPOTask.build_model`` builds policy + reference from the
        SAME initial weights — the reference is a frozen copy. This
        is the structural property that makes DPO loss meaningful
        (the log-ratio of policy vs ref).

        We verify it pre-training by comparing state_dicts immediately
        after build_model — they should match.
        """
        data_module = DPODataModule(dpo_config)
        _patch_dpo_tokenizer(monkeypatch, data_module, line_tokenizer)
        data_module.prepare_data()
        data_module.setup()
        task = DPOTask(dpo_config, data_module)

        policy = task.build_model()

        # Policy and reference must start from the same weights.
        assert task.ref_model is not None, "ref_model not built"
        # Compare all parameters by name (id() doesn't survive the
        # build_model calls).
        ref_state = dict(task.ref_model.named_parameters())
        for name, param in policy.named_parameters():
            assert name in ref_state, f"policy has param {name!r} missing from ref"
            assert torch.equal(param.data, ref_state[name].data), f"policy/ref weight mismatch at {name!r}"

        # Reference must be in eval mode + frozen.
        assert not task.ref_model.training, "ref_model should be in eval mode"
        for p in task.ref_model.parameters():
            assert not p.requires_grad, "ref_model params must have requires_grad=False"
