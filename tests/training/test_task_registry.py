"""Tests for training task registry."""

import pytest
import torch

from llm.data.modules.streaming import StreamingTextDataModule
from llm.data.modules.synthetic import SyntheticDataModule
from llm.data.modules.text import TextDataModule
from llm.training.core.config import Config
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks import builtin as _task_registry  # noqa: F401
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.reward_task import RewardTask


def test_registry_resolves_lm_task_and_text_data_module():
    lm_spec = TASK_REGISTRY.get("lm")
    assert lm_spec.task_cls is LanguageModelingTask

    data_module = lm_spec.data_module_factory(Config())
    assert type(data_module) is TextDataModule


def test_stream_lm_uses_streaming_data_module():
    spec = TASK_REGISTRY.get("stream_lm")
    data_module = spec.data_module_factory(Config())
    assert type(data_module) is StreamingTextDataModule
    assert data_module.is_streaming is True


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        TASK_REGISTRY.register("lm", LanguageModelingTask, TextDataModule)


def test_register_with_custom_data_module_factory():
    TASK_REGISTRY.register(
        "_test_custom_factory",
        LanguageModelingTask,
        data_module_factory=lambda config: SyntheticDataModule(config),
        description="custom factory test task",
    )
    spec = TASK_REGISTRY.get("_test_custom_factory")
    data_module = spec.data_module_factory(Config())
    assert type(data_module) is SyntheticDataModule


def test_reward_task_builds_trainable_stack():
    reward_spec = TASK_REGISTRY.get("reward")
    assert reward_spec.task_cls is RewardTask

    config = Config()
    task = reward_spec.task_cls(config, SyntheticDataModule(config))
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)

    assert type(optimizer) is torch.optim.AdamW
    assert any(p.requires_grad for p in model.parameters())
    assert type(scheduler) is torch.optim.lr_scheduler.SequentialLR
    assert scheduler._milestones == [config.training.warmup_epochs]
