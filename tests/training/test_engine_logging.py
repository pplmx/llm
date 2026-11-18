import pytest
import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, LoggingConfig, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.regression_task import RegressionTask


def test_log_batch_stats_handles_empty_gradients(tmp_path):
    """
    Verify _log_batch_stats does not raise IndexError when gradient_norms is empty
    (which happens during validation or first batch before backward).
    Uses real objects instead of mocks.
    """
    # 1. Real Config
    config = Config(
        training=TrainingConfig(
            task="regression",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            output_dir=str(tmp_path / "output"),
            num_samples=20,  # Required by SyntheticDataModule
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        logging=LoggingConfig(log_interval=1, log_level="INFO"),
        optimization=OptimizationConfig(num_workers=0),  # Avoid multiprocessing in tests
    )

    # 2. Real Task
    # SyntheticDataModule takes the config object
    data_module = SyntheticDataModule(config)
    data_module.setup()  # Initialize datasets

    task = RegressionTask(config, data_module=data_module)
    task.build_model()  # Ensure model is built if strictly needed, though engine calls it.

    # 3. Real Engine
    # We need rank/world_size for DistributedManager, but we can pass 0/1 for single process.
    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
    )

    # 4. Setup State
    # Force gradient_norms to be empty (it initializes empty, but just to be explicit)
    engine.performance_monitor.gradient_norms = []

    # We need the optimizer to have param groups for the logging call to access 'lr'
    engine.optimizer = task.build_optimizer(engine.model)

    # 5. Call method
    try:
        engine._log_batch_stats(epoch=0, batch_idx=0, num_batches=10, metrics={"loss": 0.5})
    except IndexError:
        pytest.fail("_log_batch_stats raised IndexError with empty gradient_norms")
    except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
        pytest.fail(f"_log_batch_stats raised unexpected exception: {e}")
    finally:
        # Cleanup log handlers to avoid polluting other tests
        if hasattr(engine, "logger") and hasattr(engine.logger, "logger"):
            handlers = engine.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                engine.logger.logger.removeHandler(handler)


def _lm_engine_off_cuda(monkeypatch, tmp_path, *, log_interval: int, num_batches: int = 2):
    """A real engine on CPU with a hand-built LM loader of ``num_batches`` batches.

    ``torch.cuda.is_available`` is patched off so the engine does not spend
    ~25 s probing free VRAM on this shared host; nothing under test depends on
    the device.
    """
    from torch.utils.data import DataLoader, TensorDataset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = Config(
        model=ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16),
        training=TrainingConfig(batch_size=2, num_samples=4, epochs=1, run_validation=False),
        optimization=OptimizationConfig(num_workers=0, use_compile=False, use_amp=False),
        logging=LoggingConfig(log_interval=log_interval, log_level="ERROR", save_logs=False, log_dir=str(tmp_path)),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    task = LanguageModelingTask(config, data_module)
    engine = TrainingEngine(config, task, rank=0, world_size=1, data_module=data_module)
    ids = torch.randint(0, config.model.vocab_size, (2 * num_batches, 16), dtype=torch.long)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=2)
    engine.val_dataloader = None
    return engine


def test_log_interval_zero_divides_by_zero_inside_the_training_loop(monkeypatch, tmp_path):
    """Records the failure ``LoggingConfig(log_interval=..., gt=0)`` prevents.

    The config now rejects a zero, so this test bypasses validation the only way
    a caller still can — assigning the attribute afterwards. It pins WHY the
    constraint exists: the modulus is evaluated per batch, so the failure is a
    bare ``ZeroDivisionError`` raised from the training loop rather than from
    anywhere that names the field. If the call site ever grows a real "0 = off"
    branch, this test has to change, which is the intended signal.
    """
    engine = _lm_engine_off_cuda(monkeypatch, tmp_path, log_interval=1)
    engine.config.logging.log_interval = 0  # pydantic does not validate assignment

    with pytest.raises(ZeroDivisionError):
        engine._run_epoch(0)


def test_log_interval_controls_which_batches_are_logged(monkeypatch, tmp_path):
    """Control for the guard: a legal interval still logs on that cadence.

    Without this, a change that skipped logging entirely would satisfy the
    division test above while silently removing the feature.
    """
    engine = _lm_engine_off_cuda(monkeypatch, tmp_path, log_interval=2)

    logged: list[int] = []
    engine._log_batch_stats = lambda epoch, batch_idx, num_batches, metrics, **kwargs: logged.append(batch_idx)

    engine._run_epoch(0)

    # 2 batches at interval 2 -> exactly the second one (index 1) is logged.
    assert logged == [1], f"interval=2 over 2 batches must log batch index 1 only, got {logged}"


# --- the logged LR must be the one the epoch TRAINED at --------------------
#
# The summary reads the scheduler AFTER advancing it, so it reported the NEXT
# epoch's LR. With warmup the gap is severe: measured over 4 epochs with
# lr=0.1 and warmup_epochs=1, epoch 0 trained at 1e-07 while the summary
# printed 3.3e-02, and the final epoch printed 0.0. TensorBoard's scalar had the
# same off-by-one because callbacks fire after the scheduler step too.


def _lr_config(tmp_path, *, epochs: int, warmup_epochs: int) -> Config:
    return Config(
        training=TrainingConfig(
            task="regression",
            epochs=epochs,
            batch_size=2,
            num_samples=8,
            lr=0.1,
            warmup_epochs=warmup_epochs,
            output_dir=str(tmp_path / "output"),
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", use_compile=False),
    )


def test_the_logged_lr_matches_the_lr_training_used(tmp_path):
    """Every epoch's summary must print the LR that epoch actually used."""
    config = _lr_config(tmp_path, epochs=4, warmup_epochs=1)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    used: list[float] = []
    real_run_epoch = engine._run_epoch

    def spy(epoch: int) -> float:
        used.append(engine.optimizer.param_groups[0]["lr"])
        return real_run_epoch(epoch)

    engine._run_epoch = spy  # type: ignore[method-assign]

    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    engine.logger.addHandler(handler)
    try:
        engine.run()
    finally:
        engine.logger.removeHandler(handler)

    printed = [line.split("LR: ")[1].split(" |")[0] for line in stream.getvalue().splitlines() if "LR: " in line]
    assert len(printed) == len(used) == 4, f"expected 4 epoch summaries, got {len(printed)}"
    assert printed == [f"{value:.6f}" for value in used], (
        f"the summary printed {printed} but training used {[f'{value:.6f}' for value in used]}"
    )


def test_the_epoch_summary_does_not_print_the_next_epochs_lr(tmp_path):
    """The distinguishing case: with warmup, epoch 1 must not look like epoch 2.

    Before the fix the first two epochs printed 0.033 and 0.067 while training
    used 1e-07 and 0.1 — the summary was one scheduler step ahead.
    """
    config = _lr_config(tmp_path, epochs=4, warmup_epochs=1)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    used: list[float] = []
    real_run_epoch = engine._run_epoch

    def spy(epoch: int) -> float:
        used.append(engine.optimizer.param_groups[0]["lr"])
        return real_run_epoch(epoch)

    engine._run_epoch = spy  # type: ignore[method-assign]

    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    engine.logger.addHandler(handler)
    try:
        engine.run()
    finally:
        engine.logger.removeHandler(handler)

    printed = [float(line.split("LR: ")[1].split(" |")[0]) for line in stream.getvalue().splitlines() if "LR: " in line]
    # Warmup epoch trains at ~0, then the post-warmup rates: 0.1, 0.075, 0.025.
    assert printed[0] < 1e-5, (
        f"epoch 1 printed LR {printed[0]}, but the warmup epoch trained at ~0; "
        "the summary is reading the scheduler one step ahead"
    )
    assert abs(printed[1] - 0.1) < 1e-6, f"epoch 2 should train at the peak 0.1, printed {printed[1]}"


# --- per-batch telemetry needs a UNIQUE x value -----------------------------
#
# Batch scalars were keyed on ``global_step`` (optimizer steps). Under gradient
# accumulation one step covers several batches, so those batches share an x
# value and TensorBoard keeps only the last write. Measured with accum=2 over 5
# batches: (batch_idx, global_step) = (0,0) (1,1) (2,1) (3,2) (4,3) — 4 distinct
# x values for 5 points.


@pytest.mark.parametrize("accum", [1, 2, 4])
def test_every_batch_gets_a_distinct_telemetry_x_value(tmp_path, accum):
    """One micro-batch, one x value — regardless of the accumulation factor."""
    config = Config(
        training=TrainingConfig(
            task="regression",
            epochs=1,
            batch_size=2,
            num_samples=10,
            lr=1e-3,
            output_dir=str(tmp_path / "output"),
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", gradient_accumulation_steps=accum, use_compile=False),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    seen: list[int] = []
    real_run_callbacks = engine._run_callbacks

    def spy(hook: str, **kwargs):
        if hook == "on_train_step_end":
            seen.append(engine.batches_seen)
        return real_run_callbacks(hook, **kwargs)

    engine._run_callbacks = spy  # type: ignore[method-assign]
    engine._run_epoch(0)

    assert len(seen) == 5, f"expected 5 micro-batches, saw {len(seen)}"
    assert len(set(seen)) == len(seen), (
        f"accum={accum}: {len(seen)} batches produced only {len(set(seen))} distinct telemetry "
        f"x values ({seen}); later points overwrite earlier ones in TensorBoard"
    )
    assert seen == sorted(seen), f"telemetry x values must be monotonic, got {seen}"


def test_the_batch_counter_is_independent_of_the_optimizer_step_counter(tmp_path):
    """The two must diverge by the accumulation factor, not track each other."""
    config = Config(
        training=TrainingConfig(
            task="regression", epochs=1, batch_size=2, num_samples=10, lr=1e-3, output_dir=str(tmp_path / "output")
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", gradient_accumulation_steps=2, use_compile=False),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    engine._run_epoch(0)

    assert engine.batches_seen == 5, f"expected 5 micro-batches, got {engine.batches_seen}"
    assert engine.global_step == 3, f"expected ceil(5/2)=3 optimizer steps, got {engine.global_step}"
    assert engine.batches_seen != engine.global_step, (
        "the micro-batch counter and the optimizer-step counter are the same value; "
        "per-batch telemetry would still collide"
    )


# --- interior accumulation micro-steps have NO gradient norm ---------------
#
# ``grad_norm`` is computed only where clipping runs, i.e. at optimizer-step
# boundaries. Interior micro-steps used to record a ``0.0`` placeholder, which
# the batch log then printed as a genuine "Grad Norm: 0.00" — indistinguishable
# from a vanishing gradient. Measured with accum=2 over 5 batches, the collected
# history was [0.0, 1.1013, 0.0, 0.7844, 1.1946]: two of five were the injected
# placeholder, so 40% of the log lines claimed zero gradients.


def _accum_config(tmp_path, accum: int) -> Config:
    return Config(
        training=TrainingConfig(
            task="regression",
            epochs=1,
            batch_size=2,
            num_samples=10,
            lr=1e-3,
            output_dir=str(tmp_path / "output"),
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", gradient_accumulation_steps=accum, use_compile=False),
        logging=LoggingConfig(log_interval=1),
    )


def test_interior_micro_steps_do_not_record_a_zero_gradient_norm(tmp_path):
    """Only real measurements may enter the gradient-norm history."""
    config = _accum_config(tmp_path, accum=2)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    engine._run_epoch(0)

    norms = engine.performance_monitor.gradient_norms
    # 5 batches at accum=2 -> 3 step boundaries, so 3 real measurements.
    assert len(norms) == 3, f"expected 3 step-boundary norms, got {len(norms)}: {norms}"
    assert all(value > 0.0 for value in norms), (
        f"a placeholder 0.0 was recorded as a gradient norm: {norms}; "
        "0.0 is a real possible norm and reads as a vanished gradient"
    )


def test_without_accumulation_every_batch_records_a_norm(tmp_path):
    """Control: accum=1 has a step boundary per batch, so no n/a and no zeros."""
    config = _accum_config(tmp_path, accum=1)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    engine._run_epoch(0)

    norms = engine.performance_monitor.gradient_norms
    assert len(norms) == 5, f"expected one norm per batch, got {len(norms)}: {norms}"
    assert all(value > 0.0 for value in norms)


def test_the_batch_log_says_na_when_this_batch_measured_no_gradient(tmp_path, caplog):
    """An interior micro-step must print 'n/a', not a borrowed or zero value."""
    import logging as logging_module

    config = _accum_config(tmp_path, accum=2)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    with caplog.at_level(logging_module.INFO):
        engine._run_epoch(0)

    lines = [record.getMessage() for record in caplog.records if "Grad Norm:" in record.getMessage()]
    assert len(lines) == 5, f"expected 5 batch log lines, got {len(lines)}"
    values = [line.split("Grad Norm: ")[1].split(" |")[0] for line in lines]
    assert values.count("n/a") == 2, (
        f"expected exactly 2 interior micro-steps to report n/a, got {values}; "
        "5 batches at accum=2 have 3 step boundaries"
    )
    assert "0.00" not in values, f"a placeholder zero was printed as a gradient norm: {values}"


# --- B-Time must be THIS batch's duration, not a running average -----------
#
# It was read from ``get_avg_batch_time()`` — a mean over the whole epoch — under
# a label that reads as a per-batch figure. Measured over 5 batches whose real
# durations were 319.1 / 2.6 / 1.5 / 1.5 / 1.5 ms, the log printed
# 319.1 / 160.9 / 107.7 / 81.2 / 65.2 ms: a steadily "improving" curve that was
# really one warm-up batch being averaged away.


def _capture_engine_log(engine) -> str:
    """Capture the engine logger's own output.

    ``caplog`` only sees records that propagate to the root logger; the engine
    configures its own logger with ``propagate=False``, so its INFO lines never
    reach caplog. Attaching a handler directly is what actually captures them.
    """
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    engine.logger.addHandler(handler)
    return stream


def _time_config(tmp_path) -> Config:
    return Config(
        training=TrainingConfig(
            task="regression", epochs=1, batch_size=2, num_samples=10, lr=1e-3, output_dir=str(tmp_path / "output")
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", use_compile=False),
        logging=LoggingConfig(log_interval=1),
    )


def test_b_time_reports_the_current_batch_not_the_running_mean(tmp_path):
    """Each batch's printed time must match that batch's own duration."""
    config = _time_config(tmp_path)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    stream = _capture_engine_log(engine)
    try:
        engine._run_epoch(0)
    finally:
        engine.logger.removeHandler(stream)

    printed = [
        float(line.split("B-Time: ")[1].split("ms")[0])
        for line in stream.getvalue().splitlines()
        if "B-Time:" in line and "n/a" not in line
    ]
    measured = [round(value * 1000, 1) for value in engine.performance_monitor.batch_times]

    assert len(printed) == len(measured) == 5, (
        f"expected 5 timed batches, got {len(printed)} printed / {len(measured)} measured"
    )
    # Tolerance, not equality: the log line re-reads the clock a few instructions
    # after the duration was recorded, which shows up as a ~0.1 ms difference on
    # short batches. A running MEAN would be off by whole milliseconds on the
    # tail (measured: 160.9 / 107.7 / 81.2 / 65.2 for real 2.6 / 1.5 / 1.5 / 1.5).
    for printed_value, measured_value in zip(printed, measured, strict=True):
        assert abs(printed_value - measured_value) < 1.0, (
            f"the log printed {printed} but the batch durations were {measured}; "
            "B-Time is showing a running average rather than the current batch"
        )


def test_the_validation_log_does_not_report_a_training_average(tmp_path):
    """Validation never times its batches, so it must print n/a, not the training mean."""
    config = _time_config(tmp_path)
    data_module = SyntheticDataModule(config)
    data_module.setup()
    engine = TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)

    engine._run_epoch(0)
    stream = _capture_engine_log(engine)
    try:
        engine._run_validation_epoch(0)
    finally:
        engine.logger.removeHandler(stream)

    lines = [line for line in stream.getvalue().splitlines() if "B-Time:" in line]
    assert lines, "the validation loop printed no batch stats"
    assert all("B-Time: n/a" in line for line in lines), (
        f"validation printed a training-phase B-Time: {[line.split('B-Time: ')[1].split(' |')[0] for line in lines]}"
    )
