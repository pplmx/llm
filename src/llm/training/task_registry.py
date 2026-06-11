"""Task registry for coupling training tasks with their DataModules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from llm.data.base import BaseDataModule
from llm.training.tasks.base_task import TrainingTask

TaskT = TypeVar("TaskT", bound=TrainingTask)
DataModuleT = TypeVar("DataModuleT", bound=BaseDataModule)

DataModuleFactory = Callable[[Any], BaseDataModule]


@dataclass(frozen=True)
class TaskSpec:
    """Binds a training task to its DataModule factory."""

    task_cls: type[TrainingTask]
    data_module_factory: DataModuleFactory
    description: str = ""


class TaskRegistry:
    """Registry mapping CLI task names to TaskSpec entries."""

    def __init__(self) -> None:
        self._registry: dict[str, TaskSpec] = {}

    def register(
        self,
        name: str,
        task_cls: type[TaskT],
        data_module_cls: type[DataModuleT] | None = None,
        *,
        data_module_factory: DataModuleFactory | None = None,
        description: str = "",
    ) -> None:
        if name in self._registry:
            raise ValueError(f"Task '{name}' is already registered.")

        if data_module_factory is None:
            if data_module_cls is None:
                raise ValueError("Either data_module_cls or data_module_factory must be provided.")

            def resolved_factory(config: Any, cls: type[DataModuleT] = data_module_cls) -> BaseDataModule:
                return cls(config)
        elif data_module_cls is not None:
            raise ValueError("Pass either data_module_cls or data_module_factory, not both.")
        else:
            resolved_factory = data_module_factory

        self._registry[name] = TaskSpec(
            task_cls=task_cls,
            data_module_factory=resolved_factory,
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
