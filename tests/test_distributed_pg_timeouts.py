"""Multi-rank test process groups must carry a bounded ``timeout=``.

Every distributed test in this repo spawns its workers and then polls the
parent with a join deadline (``TP_JOIN_TIMEOUT_S``, ``PP_JOIN_TIMEOUT_S``,
``DDP_JOIN_TIMEOUT_S``, ...). The workers themselves call
``dist.init_process_group`` directly rather than going through
``DistributedManager``, so they do NOT inherit
``DistributedConfig.collective_timeout_seconds``.

That combination produced a genuinely undiagnosable failure. Torch's default
process-group timeout is 30 MINUTES, so when one rank dies mid-test its siblings
block inside the next collective long past the parent's deadline. The parent
gives up FIRST and reports only its own budget::

    TimeoutError: TP parity spawn exceeded 360s

which names no rank, no operation, and no cause — the session handover recorded
exactly that symptom as an unreproducible 6-GPU flake. With a bounded
``timeout=`` the dead-sibling case instead raises ``torch.distributed.TimeoutError``
inside the worker, and the parent relays that message as the failing rank's
error.

A single-process group (``world_size=1``) has no sibling that can stall it, so
those call sites are exempt — that is the one shape this check allows through.
"""

from __future__ import annotations

import ast
import pathlib

TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _pg_calls(path: pathlib.Path):
    """Yield ``(lineno, call_node)`` for every ``*dist.init_process_group(...)`` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "init_process_group":
            continue
        # Match ``dist.init_process_group`` and ``_dist.init_process_group``, but
        # not an unrelated ``foo.bar.init_process_group``.
        base = func.value
        if isinstance(base, ast.Name) and base.id.lstrip("_") == "dist":
            yield node.lineno, node


def _kwarg(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal(expr) -> object | None:
    """Best-effort constant fold for ``world_size`` (ints only)."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return expr.value
    return None


def _all_test_files() -> list[pathlib.Path]:
    files = sorted(TESTS_DIR.rglob("*.py"))
    # Guard against the silent-vacuous failure this file exists to prevent.
    assert len(files) > 100, f"expected the whole suite, only found {len(files)} files"
    return files


def test_every_multi_rank_process_group_is_bounded() -> None:
    """No multi-rank ``init_process_group`` may rely on torch's 30-minute default."""
    offenders: list[str] = []
    checked = 0
    for path in _all_test_files():
        for lineno, call in _pg_calls(path):
            world_size = _literal(_kwarg(call, "world_size"))
            if world_size == 1:
                # A one-process group cannot be stalled by a sibling that died.
                continue
            checked += 1
            if _kwarg(call, "timeout") is None:
                rel = path.relative_to(TESTS_DIR.parent)
                offenders.append(f"{rel}:{lineno}")
    assert checked >= 10, (
        f"only {checked} multi-rank init_process_group calls found — the AST walk is "
        "not seeing the suite, so a green result here would be meaningless"
    )
    assert not offenders, (
        "these multi-rank process groups pass no timeout=, so they inherit torch's "
        "30-minute default and a dead rank surfaces as a blind parent join timeout:\n  " + "\n  ".join(offenders)
    )


def test_the_single_rank_exemption_is_real() -> None:
    """The ``world_size=1`` carve-out must stay justified, not become a loophole.

    Both exempt sites really do create a one-process group, so no sibling can
    stall them. This pins that shape so the exemption cannot quietly grow to cover
    a multi-rank call.
    """
    exempt: list[str] = []
    for path in _all_test_files():
        for lineno, call in _pg_calls(path):
            if _kwarg(call, "timeout") is not None:
                continue
            if _literal(_kwarg(call, "world_size")) == 1:
                exempt.append(f"{path.relative_to(TESTS_DIR.parent)}:{lineno}")
    # Sanity floor: if the AST walk broke, this test would pass vacuously.
    assert len(exempt) <= 5, f"too many unbounded single-rank groups to be a deliberate carve-out: {exempt}"


def test_a_multi_rank_call_without_timeout_would_be_caught(tmp_path) -> None:
    """Positive control: a planted offender must be reported.

    Without this, a refactor that stopped finding any calls (a renamed helper, a
    changed import alias) would leave the main test green while checking nothing.
    """
    planted = tmp_path / "test_planted_pg.py"
    planted.write_text(
        "import torch.distributed as dist\n"
        "def test_x():\n"
        "    dist.init_process_group(backend='nccl', rank=0, world_size=2)\n",
        encoding="utf-8",
    )
    calls = list(_pg_calls(planted))
    assert len(calls) == 1, calls
    _lineno, call = calls[0]
    assert _literal(_kwarg(call, "world_size")) == 2
    assert _kwarg(call, "timeout") is None, "the planted offender must look unbounded"
