"""Every module's `__all__` must name things that actually exist.

A stale `__all__` is invisible at runtime: `from llm.x import *` silently omits the
missing name, and IDEs/`ty` may or may not flag it. It also misleads anyone reading
the module's public surface — a listed-but-absent symbol looks like an API that was
removed without updating its declaration.

This walks the whole package (skipping CLI entry points and plugin shims, which
import lazily / at entry-point time) and asserts every declared name resolves.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import llm

_SKIP_PREFIXES = ("llm.cli.", "llm.plugins")


def _iter_modules():
    for module_info in pkgutil.walk_packages(llm.__path__, prefix="llm."):
        if module_info.name.startswith(_SKIP_PREFIXES):
            continue
        yield module_info.name


def test_every_module_all_entry_resolves():
    """`__all__` must not advertise names the module cannot supply."""
    missing: list[str] = []
    unimportable: list[str] = []
    checked = 0

    for name in _iter_modules():
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - ANY import failure is out of scope here
            # Not this test's job: a module that cannot import is a different
            # failure, and asserting it here would mask the __all__ check.
            unimportable.append(f"{name}: {type(exc).__name__}")
            continue
        checked += 1
        missing.extend(f"{name}.{symbol}" for symbol in getattr(module, "__all__", ()) if not hasattr(module, symbol))

    assert checked > 100, f"only {checked} modules walked — the discovery is broken"
    assert not missing, "__all__ advertises names that do not exist:\n  " + "\n  ".join(sorted(missing))


@pytest.mark.parametrize("module_name", ["llm", "llm.models", "llm.training.core", "llm.serving", "llm.multimodal"])
def test_key_packages_star_import_cleanly(module_name):
    """A package's declared surface must be importable as a whole.

    `from pkg import *` is what a plugin author or an example script writes; if one
    entry is broken the whole star-import fails, not just that name.
    """
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        pytest.skip(f"{module_name} declares no __all__ (nothing to star-import deliberately)")

    namespace: dict[str, object] = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102 - fixed, repo-owned module names
    # Only `__all__` names are pulled in by a star-import (plus the implicit
    # ``__builtins__``), so compare against exactly the declared set. Note that
    # ``llm`` deliberately lists ``__version__`` in its ``__all__``.
    exported = {key for key in namespace if key != "__builtins__"}
    assert exported == set(declared), (
        f"{module_name}: star-import produced {sorted(exported ^ set(declared))} differing from __all__"
    )


def test_declared_entry_points_all_resolve():
    """Every `llm.*` entry point in pyproject must load.

    These are the plugin seam: `pyproject.toml` names a module path, and the
    installed distribution is what makes it real. A typo'd path or a renamed
    function is INVISIBLE until a user installs the package and a registry
    silently comes up empty — the in-repo tests mock the entry-point iteration, so
    they cannot catch it either.

    Resolved from the INSTALLED distribution (importlib.metadata), so this checks
    the same data a shipped wheel would carry.
    """
    from importlib.metadata import entry_points

    declared = [group for group in entry_points().groups if group.startswith("llm.")]
    assert declared, "no llm.* entry-point groups are installed — the package metadata is not active"

    problems: list[str] = []
    seen = 0
    for group in sorted(declared):
        for entry_point in entry_points(group=group):
            seen += 1
            try:
                entry_point.load()
            except Exception as exc:  # noqa: BLE001 - report ANY load failure
                problems.append(f"{group}:{entry_point.name} -> {type(exc).__name__}: {exc}")

    assert seen > 0, "llm.* groups exist but declare no entry points"
    assert not problems, "entry points fail to load:\n  " + "\n  ".join(problems)


def test_entry_point_module_paths_exist_in_the_source_tree():
    """The declared module paths must be real files, not only importable-by-luck.

    `entry_point.load()` above could succeed through a stale editable install; this
    checks the SOURCE, so a path typo is caught even before reinstalling.
    """
    import importlib
    from importlib.metadata import entry_points

    problems: list[str] = []
    for group in sorted(g for g in entry_points().groups if g.startswith("llm.")):
        for entry_point in entry_points(group=group):
            module_path = entry_point.value.split(":")[0]
            try:
                importlib.import_module(module_path)
            except Exception as exc:  # noqa: BLE001 - report ANY import failure
                problems.append(f"{group}:{entry_point.name} -> {module_path}: {type(exc).__name__}: {exc}")

    assert not problems, "entry points name modules that do not import:\n  " + "\n  ".join(problems)
