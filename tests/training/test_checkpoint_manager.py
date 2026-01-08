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

    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "epoch_1.pt").exists()
    assert (ckpt_dir / "best.pt").exists()  # loss 1.0 < inf


def test_checkpoint_rotation(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Save 3 checkpoints (keep_last_n=2)
    for i in range(3):
        checkpoint_manager.save_checkpoint(i, model, optimizer, scheduler, scaler, loss=1.0)

    # Expect: epoch_2.pt, epoch_3.pt. epoch_1.pt should be deleted.
    assert (ckpt_dir / "epoch_3.pt").exists()
    assert (ckpt_dir / "epoch_2.pt").exists()
    assert not (ckpt_dir / "epoch_1.pt").exists()


def test_load_checkpoint(checkpoint_manager):
    # Setup: Save one
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(0, model, optimizer, scheduler, scaler, loss=0.5)

    # Enable resume
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    start_epoch, best_loss = checkpoint_manager.load_checkpoint(
        model, optimizer, scheduler, scaler, device=torch.device("cpu")
    )

    assert start_epoch == 1
    assert best_loss == 0.5
