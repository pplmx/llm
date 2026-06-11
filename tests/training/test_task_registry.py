"""Tests for training task registry."""

import pytest

from llm.data.modules.streaming import StreamingTextDataModule
from llm.data.modules.text import TextDataModule
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks import builtin as _task_registry  # noqa: F401
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.reward_task import RewardTask


def test_builtin_tasks_are_registered():
    names = TASK_REGISTRY.names()
    assert "lm" in names
    assert "stream_lm" in names
    assert "reward" in names
    assert "ppo" in names


def test_registry_resolves_task_and_data_module():
    lm_spec = TASK_REGISTRY.get("lm")
    assert lm_spec.task_cls is LanguageModelingTask

    from llm.training.core.config import Config

    config = Config()
    data_module = lm_spec.data_module_factory(config)
    assert isinstance(data_module, TextDataModule)


def test_stream_lm_uses_streaming_data_module():
    spec = TASK_REGISTRY.get("stream_lm")
    from llm.training.core.config import Config

    config = Config()
    data_module = spec.data_module_factory(config)
    assert isinstance(data_module, StreamingTextDataModule)
    assert data_module.is_streaming is True


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        TASK_REGISTRY.register("lm", LanguageModelingTask, TextDataModule)


def test_register_with_custom_data_module_factory():
    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.config import Config

    TASK_REGISTRY.register(
        "_test_custom_factory",
        LanguageModelingTask,
        data_module_factory=lambda config: SyntheticDataModule(config),
        description="custom factory test task",
    )
    spec = TASK_REGISTRY.get("_test_custom_factory")
    data_module = spec.data_module_factory(Config())
    assert isinstance(data_module, SyntheticDataModule)


def test_reward_task_has_optimizer_and_scheduler():
    reward_spec = TASK_REGISTRY.get("reward")
    assert reward_spec.task_cls is RewardTask

    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.config import Config

    config = Config()
    task = reward_spec.task_cls(config, SyntheticDataModule(config))
    model = task.build_model()
    optimizer = task.build_optimizer(model)
    scheduler = task.build_scheduler(optimizer)

    assert optimizer is not None
    assert scheduler is not None
