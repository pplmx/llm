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


def test_dev_recipe_warns_about_the_cuda_build():
    """`just dev` pulls `perf` (flash-attn, sdist-only) — the list must say so.

    `--all-extras` includes `perf`, whose `flash-attn` has no prebuilt wheel and
    compiles against CUDA at install time. Without the warning, a newcomer runs
    the recipe on a CPU-only box and waits through a build that cannot finish.
    """
    listed = _listed()
    desc = listed.get("dev", "")
    assert "CUDA" in desc, f"`just dev` does not mention the CUDA requirement: {desc!r}"
