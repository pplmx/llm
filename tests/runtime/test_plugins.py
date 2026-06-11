"""Tests for setuptools entry point plugin discovery."""

from unittest.mock import MagicMock, patch

from llm.runtime.plugins import load_entry_point_hooks, load_entry_point_registry
from llm.runtime.registry import Registry


def test_load_entry_point_registry_skips_existing_names():
    registry: Registry[str] = Registry("test")
    registry.register("builtin", "factory-a")

    ep = MagicMock()
    ep.name = "builtin"
    ep.load.return_value = "factory-b"

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[ep]):
        loaded = load_entry_point_registry("llm.test_group", registry)

    assert loaded == []
    assert registry.get("builtin") == "factory-a"


def test_load_entry_point_registry_registers_new_plugins():
    registry: Registry[str] = Registry("test")

    ep = MagicMock()
    ep.name = "custom"
    ep.load.return_value = "custom-factory"

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[ep]):
        loaded = load_entry_point_registry("llm.test_group", registry)

    assert loaded == ["custom"]
    assert registry.get("custom") == "custom-factory"


def test_load_entry_point_hooks_invokes_callables():
    hook = MagicMock()

    ep = MagicMock()
    ep.name = "register_tasks"
    ep.load.return_value = hook

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[ep]):
        invoked = load_entry_point_hooks("llm.tasks")

    hook.assert_called_once_with()
    assert invoked == ["register_tasks"]
