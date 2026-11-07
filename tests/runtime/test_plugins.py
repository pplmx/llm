"""Tests for setuptools entry point plugin discovery."""

from unittest.mock import MagicMock, patch

import pytest

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


def test_load_entry_point_registry_warns_on_skipped_collision(caplog):
    """Regression: a plugin claiming a built-in's name is
    silently dropped by the default ``overwrite=False`` path. The
    ``export/registry.py`` docstring claims this "raises loudly", but the
    loader used to skip with no log line — a packaging conflict that went
    completely unnoticed. The skip must at least emit a warning."""
    registry: Registry[str] = Registry("test")
    registry.register("onnx", "builtin-onnx")

    ep = MagicMock()
    ep.name = "onnx"
    ep.load.return_value = "plugin-onnx"

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[ep]):
        loaded = load_entry_point_registry("llm.export_backends", registry)

    assert loaded == []
    assert registry.get("onnx") == "builtin-onnx"
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("onnx" in r.getMessage() for r in warning_records), "collision must be logged"
    assert any("already registered" in r.getMessage() for r in warning_records)


def test_load_entry_point_registry_overwrite_replaces_existing_names():
    """Regression: ``overwrite=True`` must REPLACE a
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
    """one plugin failing to import must not abort the loop — the
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
    """a hook raising must not prevent later hooks from running."""
    good = MagicMock()
    good.name = "good"
    good.load.return_value = good  # any callable

    bad = MagicMock()
    bad.name = "bad"
    bad.load.return_value = type("_R", (), {"__call__": lambda self: (_ for _ in ()).throw(RuntimeError("boom"))})()

    with patch("llm.runtime.plugins._iter_group_entry_points", return_value=[bad, good]):
        invoked = load_entry_point_hooks("llm.tasks")

    assert invoked == ["good"]


# --- Lazy registries must answer public reads, without deadlocking ---


def test_lazy_registry_names_trigger_the_bootstrap():
    """``names()`` on a lazy registry must list the built-ins, not [].

    The docs tell users to run ``PEFT_REGISTRY.names()`` to discover methods,
    but the registry is populated lazily by an ``ensure_*`` function. Following
    the docs in a fresh process used to return an EMPTY list, so the documented
    workflow reported that no methods existed.
    """
    from llm.core.peft import registry as peft_registry

    # Force the cold state a fresh process starts in.
    peft_registry._methods_registered = False
    peft_registry.PEFT_REGISTRY._entries.clear()
    peft_registry.PEFT_REGISTRY._warming = False

    names = peft_registry.PEFT_REGISTRY.names()
    assert names, "names() returned an empty list for a lazily-populated registry"
    assert "lora" in names


def test_lazy_registry_unknown_get_lists_the_builtins():
    """An unknown lookup must name the real alternatives, not an empty string."""
    from llm.core.peft import registry as peft_registry

    peft_registry._methods_registered = False
    peft_registry.PEFT_REGISTRY._entries.clear()
    peft_registry.PEFT_REGISTRY._warming = False

    with pytest.raises(ValueError, match="not found") as excinfo:
        peft_registry.PEFT_REGISTRY.get("definitely_not_a_method")
    message = str(excinfo.value)
    assert "lora" in message, f"the error should list the available methods, got: {message}"


def test_bootstrap_does_not_deadlock_with_the_warmup_hook():
    """Calling ``ensure_*`` directly must not re-enter through a read.

    The ``ensure_*`` functions hold a non-reentrant lock and then check
    membership on the registry they are filling. Once reads trigger the
    bootstrap, that inner check re-enters the lock and deadlocks — observed as
    ``import llm.serving`` (and every test in ``test_peft_registry.py``)
    hanging forever.
    """
    from llm.core.peft import registry as peft_registry
    from llm.generation.registry import BACKEND_REGISTRY, ensure_backends_registered

    peft_registry._methods_registered = False
    peft_registry.PEFT_REGISTRY._entries.clear()
    peft_registry.PEFT_REGISTRY._warming = False

    # Must return promptly; the surrounding test would hang otherwise.
    peft_registry.ensure_methods_registered()
    ensure_backends_registered()
    assert "lora" in peft_registry.PEFT_REGISTRY.names()
    assert "eager" in BACKEND_REGISTRY.names()


def test_infrastructure_reads_bypass_the_bootstrap():
    """``has_raw`` must never trigger the warm-up (it runs during the bootstrap)."""
    from llm.runtime.registry import Registry

    calls: list[str] = []
    registry: Registry[int] = Registry("RawProbe")

    def warmup() -> None:
        calls.append("warmup")
        registry.register("from_warmup", 1)

    registry.set_warmup(warmup)

    assert registry.has_raw("from_warmup") is False, "has_raw triggered the bootstrap"
    assert calls == []
    # A normal read does populate it.
    assert registry.names() == ["from_warmup"]
    assert calls == ["warmup"]


def test_a_plugin_cannot_take_a_builtin_exporter_name(caplog):
    """Built-ins win: the docstring's "reference implementation" guarantee, for real.

    ``ensure_exporters_registered`` claims the built-ins are registered before
    the entry-point load precisely so a plugin claiming ``onnx`` cannot replace
    the reference exporter. That is load-order logic with no direct test on the
    real registry — the generic ``load_entry_point_registry`` cases above use a
    throwaway one.
    """
    from unittest.mock import MagicMock, patch

    from llm.export.registry import EXPORT_REGISTRY, ensure_exporters_registered

    ensure_exporters_registered()
    builtin = EXPORT_REGISTRY.get("onnx")

    hijack = MagicMock()
    hijack.name = "onnx"
    hijack.load.return_value = "hijacked-exporter"

    with (
        patch("llm.runtime.plugins._iter_group_entry_points", return_value=[hijack]),
        caplog.at_level("WARNING"),
    ):
        loaded = load_entry_point_registry("llm.export_backends", EXPORT_REGISTRY)

    assert loaded == [], "the plugin should not have been registered"
    assert EXPORT_REGISTRY.get("onnx") is builtin, "the built-in exporter was replaced by a plugin"
    assert any("onnx" in record.getMessage() for record in caplog.records), "the collision must be logged"


def test_registering_a_third_party_backend_makes_it_config_valid():
    """A plugin's backend name must pass ``ServingConfig`` validation — and only it.

    ``generation_backend`` is validated against ``BACKEND_REGISTRY`` at config
    load. If that check read a stale/empty registry, a freshly registered plugin
    would be rejected as unknown; if it passed everything through, a typo would
    reach the server and 500 on the first request. Both directions are asserted
    so the test cannot pass on a no-op validator.
    """
    from llm.generation.registry import BACKEND_REGISTRY, ensure_backends_registered

    ensure_backends_registered()

    # Baseline: an unregistered name is rejected (the guard is live).
    from llm.serving.config import ServingConfig

    with pytest.raises(ValueError, match="generation_backend"):
        ServingConfig(generation_backend="_test_plugin_backend")

    BACKEND_REGISTRY.replace("_test_plugin_backend", lambda **_kwargs: None)
    try:
        assert ServingConfig(generation_backend="_test_plugin_backend").generation_backend == "_test_plugin_backend"
    finally:
        BACKEND_REGISTRY._entries.pop("_test_plugin_backend", None)

    # Once removed it is rejected again, proving the lookup is not cached per name.
    with pytest.raises(ValueError, match="generation_backend"):
        ServingConfig(generation_backend="_test_plugin_backend")
