from pathlib import Path

import pytest
import torch

from llm.training.core.config import CheckpointConfig
from llm.training.core.utils import CheckpointManager, Logger, LoggingConfig

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
    _, best_loss = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, scaler, device=torch.device("cpu"))

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
        model, optimizer, scheduler, scaler, device=torch.device("cpu")
    )

    assert start_epoch == 1
    assert best_loss == 0.5
