"""No test may allocate a rendezvous port by binding and releasing it.

The pattern this forbids is the one that made long cumulative runs die::

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]      # socket closed here

The kernel hands out a free port, the helper returns it, and closing the socket
releases it — so between that close and the worker's own bind, any other process
on the machine can take it. Two suites that get the same port deadlock: the
loser's ``TCPStore`` client cannot reach its own server, ``init_process_group``
blocks for the full socket timeout, and the run fails with
``TimeoutError: <suite> spawn exceeded Ns``. It passes in isolation, so it reads
as machine load and is easy to attribute to the environment instead of the code.

Measured before the fix, with 8 workers racing (the width ``just test-parallel``
uses): 27 of 800 calls were handed a port another caller already held. After
moving every spawn site onto a parent-owned ``TCPStore`` (port 0, held for the
whole spawn), the same contention produced 0 collisions in 120 rounds.

Eleven files were migrated over two rounds. This test exists because the mistake
is invisible in review — the helper LOOKS careful (it binds, it listens, it reads
the assigned port) and the failure only shows up under concurrency. A regression
would cost another long debugging session, so the pattern is now rejected
statically.

The legitimate alternatives, both documented in ``tests/support/distributed.py``:

* ``rendezvous_store(world_size)`` — parent-owned store, workers get ``(host, port)``
  and join as non-master;
* a parent-created ``TCPStore(host, 0, world_size, is_master=True, ...)`` whose
  ``.port`` is published via ``MASTER_PORT``, with workers building a
  ``TCPStore(..., is_master=False)``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tests"

# The helper that explains the pattern is allowed to NAME it in prose; only real
# ``getsockname`` calls are forbidden.
DOC_ALLOWED = {"distributed.py"}


def _python_files() -> list[pathlib.Path]:
    files = [p for p in TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    # Anti-vacuity: a wrong root would make every assertion below scan nothing.
    assert len(files) > 100, f"expected the test tree, found {len(files)} files under {TESTS_ROOT}"
    return files


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _line_of(source: str, node: ast.AST) -> int:
    del source
    return int(getattr(node, "lineno", 0))


@pytest.mark.quick
def test_no_test_allocates_a_port_by_binding_and_releasing() -> None:
    """``getsockname()`` on an ephemeral bind appears nowhere in the test tree."""
    offenders: list[str] = []
    for path in _python_files():
        if path.name in DOC_ALLOWED:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - all test files must parse
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) == "getsockname":
                rel = path.relative_to(TESTS_ROOT)
                offenders.append(f"{rel}:{_line_of('', node)}")
    assert not offenders, (
        "these tests allocate a rendezvous port by binding an ephemeral socket and reading "
        "its assigned number, which RELEASES the port before any worker binds it — two suites "
        "can then be handed the same one and deadlock on connect (see "
        "tests/support/distributed.py and this module's docstring for the measurements). "
        "Use a parent-owned TCPStore instead. Offenders: " + ", ".join(sorted(offenders))
    )


@pytest.mark.quick
def test_the_support_module_is_the_only_place_that_mentions_the_pattern() -> None:
    """Positive control: the scan really reads the tree.

    If the walker silently returned nothing (bad root, bad filter), the test
    above would pass over an empty set of files. The support module documents
    the forbidden pattern, so it must be found.
    """
    found: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "socket.AF_INET, socket.SOCK_STREAM" in text:
            found.append(str(path.relative_to(TESTS_ROOT)))
    assert found, (
        "expected at least one file to contain `socket.AF_INET, socket.SOCK_STREAM` — if the "
        "walker finds none it is not reading the tree, and the getsockname scan above proves "
        "nothing"
    )


@pytest.mark.quick
def test_the_forbidden_shape_is_actually_recognised() -> None:
    """The validator flags a real instance of the pattern (self-test).

    Parses the exact helper text that used to live in eleven files and confirms
    the detector fires on it, so a refactor of the walker cannot quietly stop
    recognising the thing it exists to catch.
    """
    sample = """
import socket

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])
"""
    tree = ast.parse(sample)
    hits = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_name(node) == "getsockname"]
    assert len(hits) == 1, "the detector must recognise the historical helper"


@pytest.mark.quick
def test_a_store_based_helper_is_not_flagged() -> None:
    """The replacement shape must pass, or the gate would block the fix itself."""
    sample = """
import torch.distributed as dist

def make(world_size):
    return dist.TCPStore("127.0.0.1", 0, world_size, is_master=True, wait_for_workers=False)
"""
    tree = ast.parse(sample)
    hits = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_name(node) == "getsockname"]
    assert hits == [], "a TCPStore-based helper has no getsockname call and must not be flagged"
