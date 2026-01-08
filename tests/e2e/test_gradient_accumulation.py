import pytest

from llm.training.core.callbacks import MetricsLogger
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask
from tests.dummies import DummyLMDataModule


@pytest.mark.e2e
def test_gradient_accumulation(tiny_config):
    # Setup for grad accumulation
    # Batch size 2, Acc steps 2 -> Global Batch Size 4
    tiny_config.training.batch_size = 2
    tiny_config.optimization.gradient_accumulation_steps = 2
    tiny_config.training.num_samples = 16
    tiny_config.training.epochs = 1
    tiny_config.logging.log_interval = 1

    # Init modules
    data_module = DummyLMDataModule(tiny_config)

    task = LanguageModelingTask(tiny_config, data_module)

    engine = TrainingEngine(
        config=tiny_config, task=task, rank=0, world_size=1, data_module=data_module, callbacks=[MetricsLogger()]
    )

    engine.run()

    # Expected steps: 16 samples / 2 batch = 8 batches.
    # 8 batches / 2 acc = 4 steps.
    assert engine.global_step == 4
