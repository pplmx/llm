from typing import Any

from llm.core.registry import ComponentRegistry

# Use ComponentRegistry for consistency
_model_registry = ComponentRegistry("Model")

# Backward-compatible API
MODEL_REGISTRY: dict[str, type[Any]] = _model_registry._registry


def register_model(name: str) -> Any:
    """
    A decorator to register a model class with a given name.

    Args:
        name (str): The name to register the model under.
    """
    return _model_registry.register(name)


def get_model(name: str) -> type[Any]:
    """
    Retrieves a registered model class by its name.

    Args:
        name (str): The name of the model to retrieve.

    Returns:
        Type[Any]: The registered model class.

    Raises:
        ValueError: If no model with the given name is registered.
    """
    return _model_registry.get(name)
