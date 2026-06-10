"""Task registry for coupling training tasks with their DataModules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from llm.data.base import BaseDataModule
from llm.training.tasks.base_task import TrainingTask

TaskT = TypeVar("TaskT", bound=TrainingTask)
DataModuleT = TypeVar("DataModuleT", bound=BaseDataModule)


@dataclass(frozen=True)
class TaskSpec:
    """Binds a training task to its DataModule factory."""

    task_cls: type[TrainingTask]
    data_module_factory: Callable[[Any], BaseDataModule]
    description: str = ""


class TaskRegistry:
    """Registry mapping CLI task names to TaskSpec entries."""

    def __init__(self) -> None:
        self._registry: dict[str, TaskSpec] = {}

    def register(
        self,
        name: str,
        task_cls: type[TaskT],
        data_module_cls: type[DataModuleT],
        *,
        description: str = "",
    ) -> None:
        if name in self._registry:
            raise ValueError(f"Task '{name}' is already registered.")

        def factory(config: Any) -> BaseDataModule:
            return data_module_cls(config)

        self._registry[name] = TaskSpec(
            task_cls=task_cls,
            data_module_factory=factory,
            description=description,
        )

    def get(self, name: str) -> TaskSpec:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(f"Task '{name}' not found. Available: {available}")
        return self._registry[name]

    def names(self) -> list[str]:
        return sorted(self._registry)


TASK_REGISTRY = TaskRegistry()
