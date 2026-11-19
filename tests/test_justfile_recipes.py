"""Every ``just`` recipe must present a usable one-line description.

``just --list`` is the discovery surface for this repo's commands — ``AGENTS.md``
points every contributor at the Justfile, and the recipes are how CI and laptops
run the same thing. ``just`` shows the LAST comment line above a recipe as its
description, so a multi-line comment that wraps mid-sentence produces a fragment:

    test-cov-fast # those are now excluded here and run in test/heavy/e2e/full).
    test-gpu      # auto-skip individually when too few usable GPUs (tests/conftest.py).

Four recipes read that way — the listing was telling readers what a command does
*not* do, with the subject left on an earlier line. The fix is to end each block
with a self-contained summary line. This test pins the outcome (a description
that reads like a sentence, not a continuation) rather than the comment style,
and it also fences the ``dev`` recipe's CUDA requirement, which is the one place
where following the list blindly costs minutes of compilation.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
JUSTFILE = REPO_ROOT / "Justfile"

pytestmark = pytest.mark.skipif(shutil.which("just") is None, reason="`just` is not installed")


def _listed() -> dict[str, str]:
    """Map recipe name -> description as ``just --list`` renders it."""
    just = shutil.which("just")
    assert just is not None  # guarded by pytestmark
    # S603: the argv is fixed (this binary, this repo's Justfile), no user input.
    proc = subprocess.run(  # noqa: S603
        [just, "--list", "--justfile", str(JUSTFILE)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.split("\n"):
        if "#" not in line or not line.startswith("    "):
            continue
        name_part, _, desc = line.partition("#")
        # Some recipes take arguments: ``name arg`` in the listing.
        name = name_part.strip().split()[0] if name_part.strip() else ""
        if name:
            out[name] = desc.strip()
    return out


def test_the_listing_has_descriptions_for_the_core_recipes():
    """Anti-vacuity floor: the parse must find the recipes people actually use."""
    listed = _listed()
    assert len(listed) >= 20, f"only {len(listed)} recipes listed with descriptions"
    for required in ("test-fast", "check", "docs", "dev", "test-cov-fast", "test-gpu"):
        assert required in listed, f"{required} missing from `just --list`"


@pytest.mark.parametrize("name", _listed() or ["test-fast"])
def test_no_description_starts_mid_sentence(name: str):
    """A description opening with a conjunction or a closing bracket is a fragment.

    These are the exact shapes the four broken recipes had: they began with
    ``those``/``the``/``auto``/``Each`` (a continuation) or ended with ``).``
    (the tail of a parenthetical from an earlier line).
    """
    listed = _listed()
    if name not in listed:
        pytest.skip(f"{name} not listed")
    desc = listed[name]
    # The signal that a block wrapped mid-sentence is a description that STARTS
    # with a continuation word, or one whose parentheses do not balance (the
    # opener landed on an earlier line). "Ends with `)`" is NOT a signal — a
    # complete sentence may legitimately close with a parenthetical, e.g.
    # "Heavy resource tests only (slow, memory/CPU-intensive; separate job)."
    lowered = desc.split()[0].lower() if desc.split() else ""
    assert lowered not in {"those", "this", "each", "marked", "auto", "and", "or"}, (
        f"`{name}` description starts mid-sentence: {desc!r}"
    )
    assert desc.count("(") == desc.count(")"), (
        f"`{name}` description has unbalanced parentheses — it is the tail of a comment block that wrapped: {desc!r}"
    )


def test_every_description_is_a_plausible_sentence():
    """Cheap shape check: a summary must be non-empty, long enough, and word-led.

    This is a floor, not a prose linter. The reliable fragment signal is the
    continuation-word rule in the test above; a wrapped sentence whose last line
    happens to start with a capital ("CPU-only hosts (CI / laptops). GPU tests
    belong to ...") is indistinguishable from a deliberate summary by any cheap
    rule, so this test does not pretend to catch it.
    """
    for name, desc in _listed().items():
        assert desc, f"`{name}` has no description"
        assert len(desc) >= 8, f"`{name}` description too short to be useful: {desc!r}"
        # A description may legitimately open with an identifier rather than
        # prose: ``--cov-fail-under`` leads `test-cov`'s line, and `ty` opens
        # `ty`'s line ("ty type check"). Only reject something that is neither
        # capitalised nor a code/identifier token.
        lead = desc[0]
        assert lead.isupper() or lead.isdigit() or lead in "`-" or desc.split()[0].isidentifier(), (
            f"`{name}` description does not start with a word, capital, or code span: {desc!r}"
        )


def test_the_perf_extra_comes_from_the_astral_gpu_index():
    """`perf` (flash-attn) must resolve to a pre-built wheel, not an sdist build.

    PyPI ships flash-attn as an sdist only, and compiling it needs `nvcc`, takes
    many minutes, and fails outright on Python 3.14 + CUDA 13. `pyproject.toml`
    therefore declares the Astral GPU index, which carries a wheel per CUDA +
    torch pair. This test pins the declaration, its version floor (the indexes
    stop at 2.8.3.post1), and the fact that the recipe set exposes a way to sync
    just that extra.
    """
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    indexes = pyproject["tool"]["uv"].get("index", [])
    astral = [i for i in indexes if "wheels.astral.sh" in i.get("url", "")]
    assert astral, (
        "pyproject.toml declares no Astral GPU index, so `uv sync --extra perf` "
        "falls back to building flash-attn from its sdist"
    )

    # Declaring the index is not enough: uv's dependency-confusion protection stops
    # at the first index that carries a package, so PyPI's sdist would still win.
    # `[tool.uv.sources]` is what routes flash-attn to the Astral index, and it is
    # the half that hand-editing drops (`uv add --optional perf flash-attn
    # --index ...` writes both).
    sources = pyproject["tool"]["uv"].get("sources", {})
    assert sources.get("flash-attn", {}).get("index") == astral[0]["name"], (
        "pyproject.toml declares the Astral index but does not route flash-attn to "
        f"it via [tool.uv.sources] (found {sources.get('flash-attn')!r}); PyPI's "
        "sdist would win and the CUDA source build returns"
    )

    perf = pyproject["project"]["optional-dependencies"]["perf"]
    floors = [req for req in perf if req.startswith("flash-attn")]
    assert floors, f"the perf extra no longer lists flash-attn: {perf!r}"
    # The Astral CUDA indexes top out at 2.8.3.post1; a post2 floor cannot be met
    # by any pre-built wheel and silently reintroduces the source build.
    assert "post2" not in floors[0], (
        f"perf requires {floors[0]!r}, but the Astral GPU indexes stop at "
        "2.8.3.post1 — no pre-built wheel satisfies a post2 floor"
    )

    listed = _listed()
    assert "sync-perf" in listed, (
        "no `sync-perf` recipe, so there is no documented way to install flash-attn without `--all-extras`"
    )
