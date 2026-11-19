"""Every marker declared in ``pytest.toml`` must actually select something.

A declared-but-unused marker is not harmless. It reads as a supported way to
slice the suite, so ``pytest -m integration`` looks like it will run the
external-dependency tests — and instead it collects nothing and exits 0, which
in CI looks exactly like "those tests passed". This repo keeps its suite
offline-only (verified: no test opens a socket or calls ``requests.get``; the
HuggingFace paths are exercised through an injected fake module), so there was
no such slice to select.

``integration`` was declared that way — introduced with the tool-config split
and never used by a single test or ``just`` recipe. It has been removed, and this
test keeps the declaration honest in both directions: a marker with no users is
a lie, and a marker used without being declared produces a PytestUnknownMarkWarning
(which the project turns into noise rather than an error).
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTEST_TOML = REPO_ROOT / "pytest.toml"
TESTS_DIR = REPO_ROOT / "tests"

# ``need_gpu(n)`` is parametrised; the declaration name is the part before "(".
_MARKER_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


def _declared_markers() -> list[str]:
    config = tomllib.loads(PYTEST_TOML.read_text(encoding="utf-8"))
    raw = config["pytest"]["markers"]
    names = []
    for entry in raw:
        match = _MARKER_NAME.match(entry)
        assert match, f"unparseable marker declaration: {entry!r}"
        names.append(match.group(1))
    return names


def _used_markers() -> set[str]:
    used: set[str] = set()
    for path in TESTS_DIR.rglob("*.py"):
        for match in re.finditer(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)", path.read_text(errors="ignore")):
            used.add(match.group(1))
    return used


def test_every_declared_marker_is_used_by_some_test():
    """A declared marker with zero users is a filter that silently matches nothing."""
    unused = sorted(set(_declared_markers()) - _used_markers())
    assert not unused, (
        f"pytest.toml declares markers no test uses: {unused}. "
        "Either use them or delete them — `pytest -m <name>` on an unused marker "
        "collects nothing and exits 0, which reads as a pass."
    )


def test_the_declaration_list_is_not_empty():
    """Anti-vacuity floor: the check above must not pass on an empty list."""
    declared = _declared_markers()
    assert len(declared) >= 8, f"only {declared} declared — did the markers section get truncated?"


@pytest.mark.parametrize("marker", ["quick", "slow", "heavy", "e2e", "gpu", "multi_gpu", "full_cluster", "need_gpu"])
def test_each_marker_still_appears(marker: str):
    """The markers the Justfile recipes depend on must keep existing."""
    assert marker in _declared_markers(), f"{marker} is used by a `just` recipe but no longer declared"


# --- every recipe must run under --frozen --------------------------------
#
# ``uv run`` without ``--frozen`` re-resolves the environment when ``uv.lock`` and
# ``pyproject.toml`` disagree, silently substituting whatever versions resolve
# instead of failing. CI and Docker use ``uv sync --frozen``, so a recipe that
# omits the flag can pass locally on a different dependency set than CI installs.
# ``AGENTS.md`` states the rule ("测试均 `uv run --frozen`"); six static-check
# recipes were missing it while the test recipes had it.


def test_every_uv_run_in_the_justfile_is_frozen():
    """No ``uv run`` without ``--frozen`` anywhere in the Justfile."""
    from pathlib import Path

    justfile = (Path(__file__).resolve().parents[1] / "Justfile").read_text(encoding="utf-8")

    offenders: list[str] = []
    for lineno, line in enumerate(justfile.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("uv run"):
            continue
        if "--frozen" not in stripped:
            offenders.append(f"{lineno}: {stripped}")

    assert not offenders, (
        "these Justfile recipes call `uv run` without `--frozen`, so they can resolve "
        "a different environment than CI installs:\n  " + "\n  ".join(offenders)
    )
