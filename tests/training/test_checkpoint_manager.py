from pathlib import Path

import pytest
import torch

from llm.training.core.config import CheckpointConfig
from llm.training.core.utils import CheckpointManager, Logger, LoggingConfig
from tests.support.devices import DEFAULT_DEVICE

# Use real components where easy, mocks where interface is all that matters for file I/O
# We can use TinyModel from conftest if we import it, or just a dummy state dict holder.


class DummyState:
    def state_dict(self):
        return {"a": 1}

    def load_state_dict(self, state):
        pass


@pytest.fixture
def checkpoint_manager(tmp_path):
    config = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"), save_interval=1, keep_last_n=2, save_best=True
    )
    logging_config = LoggingConfig(log_level="DEBUG")
    logger = Logger(rank=0, config=logging_config)

    return CheckpointManager(config, rank=0, logger=logger)


def test_checkpoint_manager_init(checkpoint_manager):
    assert Path(checkpoint_manager.config.checkpoint_dir).exists()


def test_save_checkpoint(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        epoch=0, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, loss=1.0
    )

    # v2 split layout: each checkpoint name writes three sidecars.
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)
    for stem in ("latest", "epoch_1", "best"):
        assert (ckpt_dir / f"{stem}.safetensors").exists()
        assert (ckpt_dir / f"{stem}.meta.json").exists()
        assert (ckpt_dir / f"{stem}.extra_state.pt").exists()


def test_checkpoint_rotation(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Save 3 checkpoints (keep_last_n=2)
    for i in range(3):
        checkpoint_manager.save_checkpoint(i, model, optimizer, scheduler, scaler, loss=1.0)

    # Expect: epoch_2 and epoch_3 sidecars. epoch_1 sidecars should be deleted.
    assert (ckpt_dir / "epoch_3.safetensors").exists()
    assert (ckpt_dir / "epoch_3.meta.json").exists()
    assert (ckpt_dir / "epoch_3.extra_state.pt").exists()
    assert (ckpt_dir / "epoch_2.safetensors").exists()
    assert not (ckpt_dir / "epoch_1.safetensors").exists()
    assert not (ckpt_dir / "epoch_1.meta.json").exists()
    assert not (ckpt_dir / "epoch_1.extra_state.pt").exists()


def test_load_checkpoint_saves_extra_state(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        0,
        model,
        optimizer,
        scheduler,
        scaler,
        loss=0.25,
        extra_state={"stream_data": {"0": {"line_index": 42, "token_buffer": [1, 2]}}},
    )

    # Pass the legacy "latest.pt" path; the loader should resolve to
    # the split layout (latest.safetensors + latest.meta.json +
    # latest.extra_state.pt) when no legacy .pt exists at the path.
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")
    _, best_loss = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, scaler, device=DEFAULT_DEVICE)

    assert best_loss == 0.25
    assert checkpoint_manager.loaded_extra_state["stream_data"]["0"]["line_index"] == 42


def test_save_checkpoint_includes_model_config(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        0,
        model,
        optimizer,
        scheduler,
        scaler,
        loss=0.5,
        model_config={"vocab_size": 100, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 16},
    )

    # model_config lives in meta.json under the v2 layout.
    import json

    with (Path(checkpoint_manager.config.checkpoint_dir) / "latest.meta.json").open() as f:
        meta = json.load(f)
    assert meta["model_config"]["vocab_size"] == 100
    assert meta["format_version"] == "2.0"


def test_load_checkpoint(checkpoint_manager):
    # Setup: Save one
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(0, model, optimizer, scheduler, scaler, loss=0.5)

    # Enable resume — pass the legacy .pt stem; the manager finds
    # the split layout at the same stem.
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    start_epoch, best_loss = checkpoint_manager.load_checkpoint(
        model, optimizer, scheduler, scaler, device=DEFAULT_DEVICE
    )

    assert start_epoch == 1
    assert best_loss == 0.5


def test_load_checkpoint_mismatched_architecture_raises(checkpoint_manager):
    """Resume from a checkpoint whose model architecture differs from the
    current one must RAISE with a clear error, not silently restart from
    scratch (RIL ISS-108).

    A shape/key mismatch means the user changed config (e.g. hidden_size) or
    pointed at the wrong checkpoint; silently discarding the resume trains a
    full run the user believes continued the old one.
    """
    import torch.nn as nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, None, None, None, loss=0.5)
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    # Different architecture (5x5 vs saved 4x4) → load_state_dict size mismatch.
    different_arch = nn.Linear(5, 5)
    with pytest.raises(RuntimeError, match=r"checkpoint|architecture|config"):
        checkpoint_manager.load_checkpoint(different_arch, None, None, None, device=DEFAULT_DEVICE)


def test_save_stamps_matching_save_id_into_meta_and_extra(checkpoint_manager):
    """RIL ISS-127: every save stamps a shared ``save_id`` (and the epoch)
    into BOTH the meta.json and the extra_state.pt sidecar. The loader uses
    the pair to prove the three sidecars came from one atomic save."""
    import json

    checkpoint_manager.save_checkpoint(0, DummyState(), DummyState(), DummyState(), DummyState(), loss=1.0)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    meta = json.loads((ckpt_dir / "latest.meta.json").read_text())
    extra = torch.load(ckpt_dir / "latest.extra_state.pt", map_location="cpu", weights_only=False)

    assert "save_id" in meta, "meta.json must carry the per-save save_id"
    assert meta["save_id"] == extra["save_id"], "meta + extra_state must share the save_id"
    assert meta["epoch"] == extra["epoch"], "meta + extra_state must share the epoch"


def test_load_rejects_inconsistent_trio_stale_extra(checkpoint_manager):
    """RIL ISS-127: a crash interrupted between writing the (new) weights
    sidecar and the (old) extra_state/optimizer must NOT resume silently with
    a mismatched trio — new weights paired with stale optimizer/scheduler
    state would silently re-train/skip. Loading must raise instead."""
    import torch as _torch  # noqa: F401
    from torch import nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, DummyState(), DummyState(), DummyState(), loss=0.5)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Simulate a crash-mid-save: rewrite the extra_state.pt from a DIFFERENT
    # save (a stale generation) — the loader must detect the save_id/epoch
    # mismatch and refuse to silently resume.
    stale_extra = ckpt_dir / "stale.extra_state.pt"
    torch.save(
        {"optimizer_state": None, "scheduler_state": None, "scaler_state": None, "extra_state": None}, stale_extra
    )
    stale = torch.load(stale_extra, map_location="cpu", weights_only=False)
    # Copy the stale blob into the real extra_state path but with a different
    # save_id + epoch than the meta sidecar.
    stale["save_id"] = "STALE-SAVE"
    stale["epoch"] = 999
    torch.save(stale, str(ckpt_dir / "latest.extra_state.pt"))

    checkpoint_manager.config.resume_from_checkpoint = str(ckpt_dir / "latest.pt")
    with pytest.raises(ValueError, match="inconsistent"):
        checkpoint_manager.load_checkpoint(saved, None, None, None, device=DEFAULT_DEVICE)
