from unittest.mock import MagicMock

import pytest

from llm.training.core.callbacks import EarlyStopping, LRSchedulerCallback, TensorBoardLogger


def test_early_stopping_min_mode():
    # Min mode: lower is better
    callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    engine = MagicMock()
    engine.rank = 0
    engine.should_stop_training = False
    callback.set_engine(engine)
    callback.on_train_start()

    # Epoch 0: 1.0 (Best)
    callback.on_epoch_end(0, {"val_loss": 1.0})
    assert not engine.should_stop_training
    assert callback.best_value == 1.0
    assert callback.wait == 0

    # Epoch 1: 1.1 (Worse) -> Wait=1
    callback.on_epoch_end(1, {"val_loss": 1.1})
    assert not engine.should_stop_training
    assert callback.wait == 1

    # Epoch 2: 1.2 (Worse) -> Wait=2 -> Stop
    callback.on_epoch_end(2, {"val_loss": 1.2})
    assert engine.should_stop_training
    assert callback.stopped_epoch == 2


def test_early_stopping_max_mode():
    # Max mode: higher is better (e.g. accuracy)
    callback = EarlyStopping(monitor="acc", patience=1, mode="max")
    engine = MagicMock()
    engine.rank = 0
    engine.should_stop_training = False
    callback.set_engine(engine)
    callback.on_train_start()

    # Epoch 0: 0.8 (Best)
    callback.on_epoch_end(0, {"acc": 0.8})
    assert callback.best_value == 0.8

    # Epoch 1: 0.7 (Worse) -> Wait=1 -> Stop (patience=1)
    callback.on_epoch_end(1, {"acc": 0.7})
    assert engine.should_stop_training


def test_early_stopping_auto_mode():
    # Auto mode checking
    # 'loss' in name -> min
    cb_loss = EarlyStopping(monitor="val_loss", mode="auto")
    assert cb_loss.monitor_op.__name__ == "lt"  # less than

    # other -> max
    cb_acc = EarlyStopping(monitor="accuracy", mode="auto")
    assert cb_acc.monitor_op.__name__ == "gt"  # greater than


def test_early_stopping_restores_patience_on_resume():
    """Regression (RIL ISS-089): EarlyStopping must persist wait/best across
    a resume, and on_train_start must not clobber the restored state.

    Previously the patience counter restarted on every resume, so a run that
    had almost exhausted its patience kept training full epochs.
    """
    callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    engine = MagicMock()
    engine.rank = 0
    engine.should_stop_training = False
    callback.set_engine(engine)
    callback.on_train_start()

    callback.on_epoch_end(0, {"val_loss": 1.0})  # best = 1.0, wait 0
    callback.on_epoch_end(1, {"val_loss": 1.1})  # wait 1

    state = callback.get_checkpoint_state()
    assert state is not None
    assert state["early_stopping"]["wait"] == 1
    assert state["early_stopping"]["best_value"] == 1.0

    # A fresh engine resumes from the checkpoint then starts training.
    fresh = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    fresh.set_engine(engine)
    fresh.load_checkpoint_state(state)
    fresh.on_train_start()  # must NOT reset wait/best on a resumed run

    assert fresh.wait == 1, "patience counter must survive resume"
    assert fresh.best_value == 1.0, "best value must survive resume"

    # Epoch 2 still worse -> wait reaches 2 -> stop (not restart at 0).
    fresh.on_epoch_end(2, {"val_loss": 1.2})
    assert engine.should_stop_training
    assert fresh.stopped_epoch == 2


def test_tensorboard_logger_log_dir_not_doubled(tmp_path):
    """Regression (RIL ISS-410): ``on_train_start`` must log to EXACTLY the
    configured ``log_dir``. It used to append ``config.logging.log_dir`` a
    second time, so the writer landed at ``logs/logs`` (train.py already
    passes ``config.logging.log_dir`` as the constructor arg)."""
    pytest.importorskip("torch.utils.tensorboard")
    from pathlib import Path

    log_dir = tmp_path / "my-logs"
    cb = TensorBoardLogger(log_dir=str(log_dir))
    engine = MagicMock()
    engine.rank = 0
    # The engine's OWN logging.log_dir (e.g. the "logs" default) must NOT be
    # re-appended under the configured destination.
    engine.config.logging.log_dir = "logs"
    cb.set_engine(engine)
    cb.on_train_start()
    try:
        assert Path(cb.writer.log_dir) == log_dir, (
            f"writer must land exactly on {log_dir}, doubled to {cb.writer.log_dir}"
        )
    finally:
        cb.writer.close()


def test_lr_callback_without_engine_optimizer_noops():
    """Custom-loop tasks (PPO/RLHF) expose no engine-owned optimizer; the LR
    observers must skip quietly instead of raising (RIL ISS-402)."""
    engine = MagicMock()
    engine.rank = 0
    engine.optimizer = None
    engine.scheduler = None
    engine.config.logging.log_interval = 1
    engine.global_step = 1
    engine.logger = MagicMock()

    cb = LRSchedulerCallback()
    cb.set_engine(engine)
    # Neither raises despite engine.optimizer being None.
    cb.on_train_step_end(0, 0, MagicMock(), {})
    cb.on_epoch_end(0)

    tb_cb = TensorBoardLogger(log_dir="unused")
    tb_cb.set_engine(engine)
    # No writer was ever opened and no engine optimizer exists — must no-op.
    tb_cb.on_epoch_end(0, {"avg_loss": 1.0})
