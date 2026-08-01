"""Tests for RegressionTask — synthetic regression demo task."""

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, StepLR

from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.regression_task import RegressionTask


def _make_config(**overrides) -> Config:
    """Build a minimal RegressionTask config, overriding TrainingConfig fields."""
    training_defaults = {
        "task": "regression",
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "scheduler_type": "cosine",
        "warmup_epochs": 0,
    }
    training_defaults.update(overrides)
    return Config(
        training=TrainingConfig(**training_defaults),
        model=ModelConfig(hidden_size=16, num_layers=1),
    )


# --- build_model ---


def test_regression_build_model():
    task = RegressionTask(_make_config(), data_module=None)
    model = task.build_model()
    assert isinstance(model, torch.nn.Module)

    # Forward pass with random input
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 16)


# --- build_optimizer ---


def test_regression_build_optimizer():
    task = RegressionTask(_make_config(), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


# --- build_scheduler ---


def test_regression_scheduler_cosine_no_warmup():
    task = RegressionTask(_make_config(scheduler_type="cosine", warmup_epochs=0), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    assert isinstance(scheduler, CosineAnnealingLR)


def test_regression_scheduler_step_no_warmup():
    task = RegressionTask(_make_config(scheduler_type="step", warmup_epochs=0), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    assert isinstance(scheduler, StepLR)


def test_regression_scheduler_unknown_type_returns_none():
    """plateau is a valid TrainingConfig value but not in RegressionTask's scheduler_map."""
    task = RegressionTask(_make_config(scheduler_type="plateau", warmup_epochs=0), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    assert scheduler is None


def test_regression_scheduler_cosine_with_warmup():
    task = RegressionTask(_make_config(scheduler_type="cosine", warmup_epochs=2), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    assert isinstance(scheduler, SequentialLR)
    assert len(scheduler._schedulers) == 2
    assert isinstance(scheduler._schedulers[1], CosineAnnealingLR)


def test_regression_scheduler_step_with_warmup():
    task = RegressionTask(_make_config(scheduler_type="step", warmup_epochs=3), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    assert isinstance(scheduler, SequentialLR)
    assert len(scheduler._schedulers) == 2
    assert isinstance(scheduler._schedulers[1], StepLR)


def test_regression_scheduler_step():
    """Scheduler actually updates learning rate."""
    task = RegressionTask(_make_config(scheduler_type="step", warmup_epochs=0), data_module=None)
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)
    lr_before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    # StepLR decays by gamma=0.1 each step_size epochs
    assert optimizer.param_groups[0]["lr"] <= lr_before


# --- build_criterion ---


def test_regression_build_criterion():
    task = RegressionTask(_make_config(), data_module=None)
    criterion = task.build_criterion()
    assert isinstance(criterion, torch.nn.MSELoss)


# --- train_step & validation_step ---


def test_regression_train_step():
    task = RegressionTask(_make_config(), data_module=None)
    model = task.build_model()
    criterion = task.build_criterion()

    data = torch.randn(4, 16)
    target = data + 0.1 * torch.randn_like(data)
    batch = (data, target)

    loss, metrics = task.train_step(batch, model, criterion)
    assert loss.item() > 0
    assert "loss" in metrics
    assert metrics["loss"] > 0


def test_regression_validation_step():
    task = RegressionTask(_make_config(), data_module=None)
    model = task.build_model()
    criterion = task.build_criterion()

    data = torch.randn(4, 16)
    target = data + 0.1 * torch.randn_like(data)
    batch = (data, target)

    loss, metrics = task.validation_step(batch, model, criterion)
    assert loss.item() > 0
    assert "val_loss" in metrics
    assert metrics["val_loss"] > 0
