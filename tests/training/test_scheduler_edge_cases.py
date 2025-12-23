import pytest
import torch.optim as optim
from torch.nn import Linear

from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.regression_task import RegressionTask


def test_regression_task_scheduler_zero_division_fix():
    """Test that T_max avoids 0 even if epochs == warmup_epochs."""
    config = Config(
        training=TrainingConfig(
            task="regression",
            epochs=1,
            warmup_epochs=1,  # This would cause T_max=0 if not fixed
            scheduler_type="cosine",
            lr=1e-3,
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
    )

    task = RegressionTask(config, data_module=None)
    model = Linear(16, 1)  # Dummy model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Should not raise ZeroDivisionError
    scheduler = task.build_scheduler(optimizer)
    assert scheduler is not None

    # Verify we got a SequentialLR (Warmup + Cosine)
    assert isinstance(scheduler, optim.lr_scheduler.SequentialLR)

    # Check the underlying Cosine annealing T_max
    # SequentialLR schedulers list: [0] is LinearLR (warmup), [1] is Cosine
    # SequentialLR stores schedulers in _schedulers (private)
    schedulers = getattr(scheduler, "schedulers", getattr(scheduler, "_schedulers", None))
    assert schedulers is not None
    cosine_scheduler = schedulers[1]
    assert isinstance(cosine_scheduler, optim.lr_scheduler.CosineAnnealingLR)
    assert cosine_scheduler.T_max >= 1

    # Also verify that stepping doesn't crash (which is where ZeroDivisionError actually happens)
    try:
        optimizer.step()  # Suppress UserWarning about step order
        scheduler.step()
    except ZeroDivisionError:
        pytest.fail("scheduler.step() raised ZeroDivisionError")


def test_lm_task_scheduler_zero_division_fix():
    """Test LMTask specific scheduler construction too."""
    config = Config(
        training=TrainingConfig(
            task="lm",
            epochs=5,
            warmup_epochs=5,  # Equal
            scheduler_type="cosine",
            lr=1e-3,
        ),
        model=ModelConfig(hidden_size=16, num_layers=1),
    )
    task = LanguageModelingTask(config, data_module=None)
    model = Linear(16, 1)  # Dummy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = task.build_scheduler(optimizer)
    assert scheduler is not None
    schedulers = getattr(scheduler, "schedulers", getattr(scheduler, "_schedulers", None))
    assert schedulers is not None
    cosine_scheduler = schedulers[1]
    assert cosine_scheduler.T_max >= 1

    try:
        optimizer.step()
        scheduler.step()
    except ZeroDivisionError:
        pytest.fail("scheduler.step() raised ZeroDivisionError")
