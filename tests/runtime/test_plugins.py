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


def test_load_entry_point_registry_overwrite_replaces_existing_names():
    """Regression (RIL ISS-061): ``overwrite=True`` must REPLACE a
    pre-registered name instead of crashing.

    The old loader skipped the existence guard when ``overwrite=True`` but
    still called ``Registry.register``, which raises
    ``ValueError: 'X' is already registered`` — so the documented override
    mechanism could never overwrite anything."""
    registry: Registry[str] = Registry("test")
    registry.register("builtin", "factory-a")

    ep = MagicMock()
    ep.name = "builtin"
    ep.load.return_value = "factory-b"

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[ep]):
        loaded = load_entry_point_registry("llm.test_group", registry, overwrite=True)

    assert loaded == ["builtin"]
    assert registry.get("builtin") == "factory-b"


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


def test_load_entry_point_registry_isolates_broken_plugin():
    """RIL ISS-131: one plugin failing to import must not abort the loop — the
    healthy plugins still register, and the registry is usable."""
    registry: Registry[str] = Registry("test")
    registry.register("builtin", "factory-builtin")

    broken = MagicMock()
    broken.name = "zzz_broken"
    broken.load.side_effect = ImportError("broken module")

    healthy = MagicMock()
    healthy.name = "healthy"
    healthy.load.return_value = "factory-healthy"

    eps = [broken, healthy]
    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=eps):
        loaded = load_entry_point_registry("llm.test_group", registry)

    # The healthy plugin registered; the builtin is untouched; the broken one
    # was skipped (not fatal).
    assert loaded == ["healthy"]
    assert registry.get("healthy") == "factory-healthy"
    assert registry.get("builtin") == "factory-builtin"
    assert "zzz_broken" not in registry


def test_load_entry_point_hooks_isolates_broken_hook():
    """RIL ISS-131: a hook raising must not prevent later hooks from running."""
    good = MagicMock()
    good.name = "good"
    good.load.return_value = good  # any callable

    bad = MagicMock()
    bad.name = "bad"
    bad.load.return_value = type("_R", (), {"__call__": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))})()

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[bad, good]):
        invoked = load_entry_point_hooks("llm.tasks")

    assert invoked == ["good"]
