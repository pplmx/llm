from collections.abc import Callable
from typing import Any

# A simple dictionary to store registered models
MODEL_REGISTRY: dict[str, type[Any]] = {}


def register_model(name: str) -> Callable[[type[Any]], type[Any]]:
    """
    A decorator to register a model class with a given name.

    Args:
        name (str): The name to register the model under.
    """

    def decorator(model_class: type[Any]) -> type[Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model with name '{name}' already registered.")
        MODEL_REGISTRY[name] = model_class
        return model_class

    return decorator


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
    if name not in MODEL_REGISTRY:
        raise ValueError(f"No model with name '{name}' registered.")
    return MODEL_REGISTRY[name]
