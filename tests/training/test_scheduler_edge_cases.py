import torch.optim as optim
from torch.nn import Linear

from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.regression_task import RegressionTask


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


# --- the LR trajectory itself, not just its shape -------------------------
#
# The two tests above guard ``T_max >= 1``. Nothing checked the VALUES the
# schedule produces, so a change to ``eta_min``, the warmup split, or the cosine
# argument would keep them green while training silently ran at the wrong rate.
#
# Verified against the closed form:
#   warmup (LinearLR, start_factor=1e-6, end 1.0 over ``warmup`` epochs)
#   then   CosineAnnealingLR(eta_min = lr * 0.1, T_max = epochs - warmup)


def _trajectory(config, epochs: int) -> list[float]:
    """Run the scheduler for ``epochs`` and return the LR at each epoch start."""
    import torch.nn as nn
    import torch.optim as optim

    from llm.training.tasks.lm_task import LanguageModelingTask

    task = LanguageModelingTask.__new__(LanguageModelingTask)
    task.config = type("Cfg", (), {"training": config})()
    optimizer = optim.SGD(nn.Linear(2, 2).parameters(), lr=config.lr)
    scheduler = task.build_scheduler(optimizer)
    assert scheduler is not None, "the scheduler must exist for these configs"

    values = []
    for _ in range(epochs):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return values


def test_the_cosine_trajectory_matches_the_closed_form():
    """Warmup ramps to the base LR, then cosine decays toward ``lr * 0.1``."""
    import math

    from llm.training.core.config import TrainingConfig

    base_lr, epochs, warmup = 0.01, 6, 2
    config = TrainingConfig(scheduler_type="cosine", epochs=epochs, warmup_epochs=warmup, lr=base_lr)
    trajectory = _trajectory(config, epochs)

    # Warmup: LinearLR from start_factor 1e-6 to 1.0, so epoch 0 is ~0 and the
    # last warmup epoch reaches the base rate exactly.
    assert trajectory[0] < 1e-5, f"the first epoch must start near zero, got {trajectory[0]}"
    assert abs(trajectory[warmup] - base_lr) < 1e-9, (
        f"the LR must reach the base rate at the end of warmup, got {trajectory[warmup]}"
    )

    # Cosine: eta_min + (base - eta_min) * (1 + cos(pi*t/T_max)) / 2
    t_max = epochs - warmup
    eta_min = base_lr * 0.1
    for index in range(1, t_max):
        expected = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * index / t_max)) / 2
        actual = trajectory[warmup + index]
        assert abs(actual - expected) < 1e-6, f"epoch {warmup + index}: {actual} != closed form {expected}"

    # And the floor is respected at the end.
    assert trajectory[-1] >= eta_min - 1e-9


def test_the_step_schedule_halves_the_rate_at_the_midpoint():
    """``StepLR(step_size=epochs//2, gamma=0.5)`` — the documented shape."""
    from llm.training.core.config import TrainingConfig

    epochs = 4
    config = TrainingConfig(scheduler_type="step", epochs=epochs, warmup_epochs=0, lr=0.01)
    trajectory = _trajectory(config, epochs)

    assert abs(trajectory[0] - 0.01) < 1e-9
    # step_size = epochs // 2 = 2, gamma 0.5.
    assert abs(trajectory[2] - 0.005) < 1e-9, f"the LR must halve at the midpoint, got {trajectory}"
    assert abs(trajectory[3] - 0.005) < 1e-9


def test_warmup_is_skipped_when_zero():
    """``warmup_epochs=0`` must return the bare main scheduler, starting AT the base rate."""
    from llm.training.core.config import TrainingConfig

    config = TrainingConfig(scheduler_type="cosine", epochs=4, warmup_epochs=0, lr=0.01)
    trajectory = _trajectory(config, 4)

    assert abs(trajectory[0] - 0.01) < 1e-9, f"without warmup epoch 0 must train at the base rate, got {trajectory[0]}"
