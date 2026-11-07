import operator
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
    """Regression: EarlyStopping must persist wait/best across
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
    """Regression: ``on_train_start`` must log to EXACTLY the
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
    observers must skip quietly instead of raising."""
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


def test_tensorboard_logger_degrades_when_tensorboard_missing(monkeypatch):
    """A missing optional ``tensorboard`` must not abort the training run.

    ``tensorboard`` lives in the optional ``[logging]`` extra, but ``train.py``
    wires :class:`TensorBoardLogger` unconditionally — so a plain
    ``llm train`` on a default install raised ``ImportError`` from
    ``on_train_start`` before the first batch, killing an otherwise valid run.
    The callback now degrades to no-TensorBoard with a warning (the same
    no-op-with-a-warning contract ``llm.data.dvc`` uses for its optional
    backend) and training continues.
    """
    cb = TensorBoardLogger(log_dir="unused")
    engine = MagicMock()
    engine.rank = 0
    engine.logger = MagicMock()
    cb.set_engine(engine)

    # Simulate the import failing exactly as it does on a default install.
    def _boom(_log_path):
        raise ImportError("tensorboard is an optional training dependency.")

    monkeypatch.setattr(TensorBoardLogger, "_make_writer", staticmethod(_boom))
    cb.on_train_start()  # must NOT raise

    assert cb.writer is None, "writer must stay unset when tensorboard is unavailable"
    assert engine.logger.warning.called, "the degradation must be announced, not silent"
    # Subsequent hooks must tolerate the absent writer.
    cb.on_epoch_end(0, {"avg_loss": 1.0})
    cb.on_train_end()


# --- EarlyStopping: mode resolution must be idempotent ---


@pytest.mark.parametrize(
    ("mode", "monitor", "expected_op", "delta_negative"),
    [
        ("min", "val_loss", operator.lt, True),
        ("max", "val_acc", operator.gt, False),
        ("auto", "val_loss", operator.lt, True),
        ("auto", "val_accuracy", operator.gt, False),
    ],
)
def test_early_stopping_mode_resolution_is_idempotent(mode, monitor, expected_op, delta_negative):
    """Re-resolving the mode must not flip ``min_delta`` back.

    ``_check_mode`` used to do ``self.min_delta *= -1``, so calling it a second
    time restored the ORIGINAL sign (-0.1 -> +0.1 -> -0.1). Only ``__init__``
    calls it today, but the in-place mutation means any future re-resolve
    silently compares against the wrong threshold. It now derives the sign from
    the recorded ``min_delta`` every time.
    """
    stopping = EarlyStopping(monitor=monitor, patience=2, mode=mode, min_delta=0.1)
    after_init = (stopping.monitor_op, stopping.min_delta)

    # ONE extra call is enough to expose the bug: the old implementation did
    # `min_delta *= -1`, so a single re-resolve flipped the sign. Asserting only
    # after TWO calls would miss it entirely, because two flips return to the
    # original value.
    stopping._check_mode()
    assert (stopping.monitor_op, stopping.min_delta) == after_init, (
        f"mode={mode!r} drifted after ONE extra check: {after_init} -> {(stopping.monitor_op, stopping.min_delta)}"
    )

    stopping._check_mode()
    assert (stopping.monitor_op, stopping.min_delta) == after_init, (
        f"mode={mode!r} drifted after repeated checks: {after_init} -> {(stopping.monitor_op, stopping.min_delta)}"
    )
    assert stopping.monitor_op is expected_op
    assert (stopping.min_delta < 0) is delta_negative


def test_early_stopping_resume_keeps_restored_patience():
    """A resumed run must not reset the patience counter it just restored."""
    from unittest.mock import MagicMock

    stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    engine = MagicMock()
    engine.rank = 0
    stopping.set_engine(engine)

    stopping.load_checkpoint_state({"early_stopping": {"wait": 2, "stopped_epoch": 0, "best_value": 0.5}})
    stopping.on_train_start()

    assert stopping.wait == 2, "on_train_start clobbered the restored patience counter"
    assert stopping.best_value == 0.5


def test_early_stopping_fresh_run_starts_from_zero():
    """Without restored state, a new run must begin with a clean counter."""
    from unittest.mock import MagicMock

    stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    engine = MagicMock()
    engine.rank = 0
    stopping.set_engine(engine)

    stopping.wait = 9
    stopping.best_value = 0.1
    stopping.on_train_start()

    assert stopping.wait == 0
    assert stopping.best_value is None
