import pytest

from llm.data.synthetic_data_module import SyntheticDataModule
from llm.training.core.config import Config, LoggingConfig, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
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
    except Exception as e:
        pytest.fail(f"_log_batch_stats raised unexpected exception: {e}")
    finally:
        # Cleanup log handlers to avoid polluting other tests
        if hasattr(engine, "logger") and hasattr(engine.logger, "logger"):
            handlers = engine.logger.logger.handlers[:]
            for handler in handlers:
                handler.close()
                engine.logger.logger.removeHandler(handler)
