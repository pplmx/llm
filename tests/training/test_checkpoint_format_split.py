"""Tests for the v2 split checkpoint format (ADR-006).

The v2 layout splits a training checkpoint into three sidecars:

- ``<name>.safetensors`` — model weights (state dict only)
- ``<name>.meta.json`` — JSON-encoded training metadata
- ``<name>.extra_state.pt`` — torch.save'd optimizer / scheduler /
  scaler + extra_state

The :class:`CheckpointManager` writes this layout by default and
:meth:`CheckpointManager.load_checkpoint` accepts both the new layout
AND the legacy single-file ``.pt`` layout from v0.0.5 — auto-detected
on read. These tests pin both halves:

- **Write side** — ``save_checkpoint`` writes the expected three
  sidecars with the right content shape (format_version, model_state
  round-trips byte-identical, metadata is JSON-parseable).
- **Read side (new layout)** — ``load_checkpoint_payload`` returns the
  unified dict, and ``CheckpointManager.load_checkpoint`` populates
  the model / optimizer / scheduler / scaler state and the
  ``loaded_extra_state``.
- **Read side (legacy compatibility)** — a v0.0.5-era single-file
  ``.pt`` written by hand loads cleanly via the same entry point,
  with a one-shot ``DeprecationWarning`` fired.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from llm.training.core.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    EXTRA_STATE_SUFFIX,
    META_SUFFIX,
    SAFETENSORS_SUFFIX,
    load_checkpoint_payload,
)
from llm.training.core.config import CheckpointConfig
from llm.training.core.utils import CheckpointManager, Logger, LoggingConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TensorState:
    """Minimal stand-in for an nn.Module with a real tensor state dict.

    The legacy ``DummyState`` in :file:`tests/training/test_checkpoint_manager.py`
    only carries an ``int`` in its state dict — fine for legacy
    ``torch.save`` (which serializes anything), but useless for the
    safetensors sidecar (which requires tensors). This fixture has
    BOTH a tensor (for the safetensors sidecar) and an int (for the
    optimizer / scheduler state sidecar), so we can exercise the full
    v2 layout.
    """

    def __init__(self) -> None:
        self._tensor = torch.zeros(2, 3)
        self._int_field = 1  # mimics optimizer step count

    def state_dict(self) -> dict:
        return {"weight": self._tensor, "step": self._int_field}

    def load_state_dict(self, state) -> None:
        if "weight" in state:
            self._tensor = state["weight"].clone()
        if "step" in state:
            self._int_field = state["step"]


@pytest.fixture
def checkpoint_manager(tmp_path: Path):
    config = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_interval=10,  # only `latest` / `best` written; no epoch_*.safetensors
        keep_last_n=2,
        save_best=True,
    )
    logging_config = LoggingConfig(log_level="DEBUG")
    logger = Logger(rank=0, config=logging_config)
    return CheckpointManager(config, rank=0, logger=logger)


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


class TestSaveSplitFormat:
    """`save_checkpoint` writes three sidecars at the expected paths."""

    def test_latest_split_writes_three_sidecars(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(epoch=0, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=1.0)
        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / f"latest{SAFETENSORS_SUFFIX}").exists()
        assert (ckpt_dir / f"latest{META_SUFFIX}").exists()
        assert (ckpt_dir / f"latest{EXTRA_STATE_SUFFIX}").exists()

    def test_best_written_when_loss_improves(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(epoch=0, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=1.0)
        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / f"best{SAFETENSORS_SUFFIX}").exists()
        assert (ckpt_dir / f"best{META_SUFFIX}").exists()
        assert (ckpt_dir / f"best{EXTRA_STATE_SUFFIX}").exists()

    def test_epoch_sidecars_only_when_interval_triggers(self, checkpoint_manager, tmp_path: Path):
        # save_interval=10 → save_checkpoint for epoch 0 does NOT
        # write an epoch_<N> set; only `latest` and `best` go to disk.
        ts = TensorState()
        checkpoint_manager.save_checkpoint(epoch=0, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=1.0)
        ckpt_dir = tmp_path / "checkpoints"
        assert not (ckpt_dir / "epoch_1.safetensors").exists()
        # Force the interval boundary: epoch 9 (the 10th call) triggers epoch_10.
        checkpoint_manager.config.save_interval = 1
        checkpoint_manager.save_checkpoint(epoch=9, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=0.5)
        assert (ckpt_dir / "epoch_10.safetensors").exists()

    def test_meta_json_carries_format_version_and_metadata(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(
            epoch=2,
            model=ts,
            optimizer=ts,
            scheduler=ts,
            scaler=ts,
            loss=0.42,
            model_config={"vocab_size": 100, "hidden_size": 16},
        )
        meta_path = tmp_path / "checkpoints" / f"latest{META_SUFFIX}"
        meta = json.loads(meta_path.read_text())
        assert meta["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert meta["epoch"] == 2
        assert meta["loss"] == 0.42
        assert meta["model_config"]["vocab_size"] == 100

    def test_extra_state_pt_carries_optimizer_and_extra_state(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=ts,
            optimizer=ts,
            scheduler=ts,
            scaler=ts,
            loss=1.0,
            extra_state={"stream_data": {"0": {"line_index": 7}}},
        )
        blob = torch.load(
            tmp_path / "checkpoints" / f"latest{EXTRA_STATE_SUFFIX}",
            map_location="cpu",
            weights_only=False,
        )
        assert "optimizer_state" in blob
        assert "scheduler_state" in blob
        assert "scaler_state" in blob
        assert blob["extra_state"]["stream_data"]["0"]["line_index"] == 7


# ---------------------------------------------------------------------------
# Read side — new layout
# ---------------------------------------------------------------------------


class TestLoadSplitFormat:
    """`load_checkpoint_payload` returns the unified dict from the v2 layout."""

    def test_roundtrip_through_payload_helper(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(epoch=0, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=1.0)
        # Use the public helper. Pass the legacy ``.pt`` stem — the
        # helper auto-resolves to the split layout at the same stem.
        payload = load_checkpoint_payload(tmp_path / "checkpoints" / "latest.pt")
        assert payload is not None
        assert payload["epoch"] == 0
        assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
        # ``model_state`` carries the tensor; ``step`` is in the
        # extra_state sidecar (filtered out by safetensors because
        # it's an int).
        assert "weight" in payload["model_state"]
        assert torch.equal(payload["model_state"]["weight"], torch.zeros(2, 3))

    def test_checkpoints_manager_loads_split_layout(self, checkpoint_manager, tmp_path: Path):
        ts = TensorState()
        checkpoint_manager.save_checkpoint(
            epoch=3,
            model=ts,
            optimizer=ts,
            scheduler=ts,
            scaler=ts,
            loss=0.5,
            extra_state={"stream_data": {"0": {"line_index": 99}}},
        )
        # Pass the legacy ``.pt`` stem; the manager auto-resolves to
        # the split layout at the same stem.
        checkpoint_manager.config.resume_from_checkpoint = str(tmp_path / "checkpoints" / "latest.pt")
        start_epoch, best_loss = checkpoint_manager.load_checkpoint(ts, ts, ts, ts, device=torch.device("cpu"))
        assert start_epoch == 4
        assert best_loss == 0.5
        assert checkpoint_manager.loaded_extra_state["stream_data"]["0"]["line_index"] == 99


# ---------------------------------------------------------------------------
# Read side — legacy .pt compatibility
# ---------------------------------------------------------------------------


class TestLegacyCheckpointLoad:
    """A v0.0.5-era single-file ``.pt`` still loads via the same entry point."""

    def test_legacy_pt_loads_with_deprecation_warning(self, tmp_path: Path):
        legacy_path = tmp_path / "old_checkpoint.pt"
        legacy_blob = {
            "epoch": 7,
            "loss": 0.123,
            "best_loss": 0.05,
            "model_state": {"weight": torch.ones(2, 3)},
            "optimizer_state": {"step": 100},
            "scheduler_state": None,
            "scaler_state": None,
            "extra_state": {"stream_data": {"0": {"line_index": 13}}},
        }
        torch.save(legacy_blob, legacy_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = load_checkpoint_payload(legacy_path)

        # The DeprecationWarning fires once.
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "llm-migrate-ckpt" in str(deprecations[0].message)

        # Payload is the unified dict, with format_version absent (legacy).
        assert payload is not None
        assert payload["epoch"] == 7
        assert payload["loss"] == 0.123
        assert payload["model_state"]["weight"].shape == (2, 3)
        assert payload["extra_state"]["stream_data"]["0"]["line_index"] == 13
        assert payload.get("format_version") is None

    def test_legacy_loader_returns_none_when_path_missing(self, tmp_path: Path):
        # Non-existent path → None (caller decides whether to warn).
        assert load_checkpoint_payload(tmp_path / "nope.pt") is None


# ---------------------------------------------------------------------------
# Sidecar cleanup
# ---------------------------------------------------------------------------


class TestCheckpointRotationSplitLayout:
    """`_cleanup_old_checkpoints` removes all three sidecars at the stem."""

    def test_rotation_removes_all_three_sidecars(self, tmp_path: Path):
        config = CheckpointConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            save_interval=1,
            keep_last_n=1,  # keep only the newest
            save_best=False,
        )
        logger = Logger(rank=0, config=LoggingConfig(log_level="DEBUG"))
        mgr = CheckpointManager(config, rank=0, logger=logger)

        ts = TensorState()
        for i in range(3):
            mgr.save_checkpoint(epoch=i, model=ts, optimizer=ts, scheduler=ts, scaler=ts, loss=1.0)
            mgr.config.save_interval = 1  # re-arm in case default slipped

        ckpt_dir = tmp_path / "checkpoints"
        # With keep_last_n=1 and 3 saves (epoch_1, epoch_2, epoch_3),
        # only the newest trio should remain. The first one's sidecars
        # must all be gone.
        existing = sorted(p.name for p in ckpt_dir.iterdir())
        assert "epoch_1.safetensors" not in existing
        assert "epoch_1.meta.json" not in existing
        assert "epoch_1.extra_state.pt" not in existing
        assert "epoch_3.safetensors" in existing
        assert "epoch_3.meta.json" in existing
        assert "epoch_3.extra_state.pt" in existing
