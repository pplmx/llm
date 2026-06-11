import pytest
import torch.optim as optim
from torch.nn import Linear

from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.regression_task import RegressionTask


@pytest.mark.heavy
def test_regression_task_scheduler_avoids_zero_tmax():
    """Requirement: cosine scheduler uses T_max>=1 when warmup consumes all epochs."""
    config = Config(
        training=TrainingConfig(
            task="regression",
            epochs=1,
            warmup_epochs=1,
            scheduler_type="cosine",
            lr=1e-3,
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
    )

    task = RegressionTask(config, data_module=None)
    optimizer = optim.Adam(Linear(16, 1).parameters(), lr=1e-3)
    scheduler = task.build_scheduler(optimizer)

    cosine_scheduler = scheduler._schedulers[1]
    assert cosine_scheduler.T_max >= 1

    optimizer.step()
    scheduler.step()


@pytest.mark.heavy
def test_lm_task_scheduler_avoids_zero_tmax():
    """Requirement: LMTask scheduler survives epochs equal to warmup_epochs."""
    config = Config(
        training=TrainingConfig(
            task="lm",
            epochs=5,
            warmup_epochs=5,
            scheduler_type="cosine",
            lr=1e-3,
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
    )
    task = LanguageModelingTask(config, data_module=None)
    optimizer = optim.Adam(Linear(16, 1).parameters(), lr=1e-3)

    scheduler = task.build_scheduler(optimizer)
    cosine_scheduler = scheduler._schedulers[1]
    assert cosine_scheduler.T_max >= 1

    optimizer.step()
    scheduler.step()
