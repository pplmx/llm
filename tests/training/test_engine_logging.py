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
    engine._log_batch_stats = lambda epoch, batch_idx, num_batches, metrics: logged.append(batch_idx)

    engine._run_epoch(0)

    # 2 batches at interval 2 -> exactly the second one (index 1) is logged.
    assert logged == [1], f"interval=2 over 2 batches must log batch index 1 only, got {logged}"
