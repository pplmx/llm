from unittest.mock import MagicMock

from llm.training.core.callbacks import EarlyStopping


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
