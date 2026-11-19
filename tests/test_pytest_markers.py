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


# --- CI must pin the same interpreter the docs workflow pins ---------------
#
# ``requires-python = ">=3.14"`` is a RANGE, so `uv` may legally pick a newer
# interpreter on a runner that has one (3.15 exists). ``docs.yml`` pins 3.14
# explicitly via ``actions/setup-python``; ``ci.yml`` used to leave every
# ``setup-uv`` block unpinned, so the lint/test jobs and the docs job could run on
# different interpreters — and a future 3.15-only behaviour would surface only in
# whichever job happened to get it.


def test_ci_pins_python_where_docs_pins_python():
    """Every ``setup-uv`` block in ci.yml must carry the same python-version."""
    from pathlib import Path

    import yaml

    workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    ci_text = (workflows / "ci.yml").read_text(encoding="utf-8")
    docs_text = (workflows / "docs.yml").read_text(encoding="utf-8")

    # What does the docs job pin? Follow it rather than hardcoding.
    docs_pin = None
    for step in yaml.safe_load(docs_text)["jobs"].values():
        for action in step.get("steps", []):
            with_ = action.get("with") or {}
            if with_.get("python-version"):
                docs_pin = str(with_["python-version"])
    assert docs_pin, "docs.yml no longer pins python-version; update this test"

    blocks = ci_text.count("uses: astral-sh/setup-uv")
    pinned = ci_text.count(f'python-version: "{docs_pin}"')
    assert pinned >= blocks, (
        f"ci.yml has {blocks} setup-uv blocks but only {pinned} carry "
        f'python-version: "{docs_pin}" (the pin docs.yml uses), so jobs can run on '
        "different interpreters"
    )


# --- CI workflows must declare minimum token permissions -------------------
#
# A workflow without a `permissions:` block inherits the repository default, which
# can be read-write across all scopes. `docs.yml` already narrows itself to
# `contents: read` + `pages: write` (it deploys Pages); `ci.yml` narrows to
# `contents: read` (it only reads the repo and uploads artefacts, which uses a
# separate scoped token). The permissive default is a privilege the jobs never use.


def test_ci_declares_read_only_permissions():
    """`ci.yml` must declare `contents: read` rather than inheriting the default."""
    from pathlib import Path

    import yaml

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))

    permissions = parsed.get("permissions")
    assert permissions is not None, (
        "ci.yml has no top-level `permissions:` block, so it inherits the repository "
        "default (possibly read-write for every scope) instead of the read-only access "
        "its jobs actually need"
    )
    assert permissions.get("contents") == "read", f"ci.yml should narrow contents to read, got {permissions!r}"


def test_ci_does_not_grant_write_scopes_it_never_uses():
    """The CI jobs read the repo and upload artefacts; they never push back."""
    from pathlib import Path

    import yaml

    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    permissions = yaml.safe_load(workflow.read_text(encoding="utf-8"))["permissions"]

    write_scopes = {scope for scope, level in permissions.items() if level == "write"}
    assert not write_scopes, (
        f"ci.yml grants write access to {sorted(write_scopes)}, but no job in it needs "
        "to push commits, releases, pages or packages"
    )


def test_every_workflow_declares_permissions():
    """A workflow without a block inherits the repository default."""
    from pathlib import Path

    import yaml

    workflows_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    missing = [
        path.name
        for path in sorted(workflows_dir.glob("*.yml"))
        if yaml.safe_load(path.read_text(encoding="utf-8")).get("permissions") is None
    ]

    assert not missing, (
        "these workflows declare no `permissions:` block and so inherit the repository "
        f"default (possibly read-write for every scope): {missing}"
    )


def test_the_write_scopes_granted_are_the_expected_minimum_set():
    """Each workflow's write scopes must match what it actually does.

    Pinning the set stops a future edit from quietly widening one — e.g. giving the
    CI workflow ``contents: write`` to make some upload work.
    """
    from pathlib import Path

    import yaml

    workflows_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    expected_writes = {
        "ci.yml": set(),
        "docs.yml": {"pages", "id-token"},  # deploy to Pages
        "docker.yml": {"packages"},  # push image tags to GHCR
        "labeler.yml": {"issues", "pull-requests"},  # apply labels
        "release.yml": {"contents"},  # publish a GitHub release
    }

    actual: dict[str, set[str]] = {}
    for name in expected_writes:
        permissions = yaml.safe_load((workflows_dir / name).read_text(encoding="utf-8"))["permissions"]
        actual[name] = {scope for scope, level in permissions.items() if level == "write"}

    assert actual == expected_writes, (
        "a workflow's granted write scopes changed; update the expectation only if the "
        f"workflow genuinely needs it.\n  expected: {expected_writes}\n  actual:   {actual}"
    )
