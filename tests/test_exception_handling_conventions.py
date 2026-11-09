"""Conventions that keep failures debuggable.

Two properties are worth enforcing repo-wide, because both erode silently:

1. A ``except Exception`` that neither re-raises nor logs swallows the failure.
   That is sometimes correct — a capability probe, or a fallback where a later
   layer reports properly — but it is never *self-evident*, so each one must say
   why in an inline comment or a ``noqa`` rationale. Without that, the next
   reader cannot tell a deliberate probe from a bug.
2. Exceptions must not be caught as ``BaseException`` in ordinary request
   handling: it captures ``KeyboardInterrupt`` and ``SystemExit``, so Ctrl-C
   stops interrupting. One place does it deliberately (the batch engine forwards
   it through a result object and re-raises later, after freeing slots), and it
   documents the reason.

The scan uses the stdlib ``ast`` so it sees every handler, including nested and
conditional ones.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "llm"
assert _SRC.is_dir(), f"source tree not found at {_SRC}"

#: Handlers that neither raise nor log but are deliberate. Each entry names the
#: file and why, so a new silent handler has to be justified here as well.
_SILENT_EXCEPTION_ALLOWLIST: dict[str, str] = {}


def _handlers() -> list[tuple[pathlib.Path, ast.ExceptHandler, list[str]]]:
    out: list[tuple[pathlib.Path, ast.ExceptHandler, list[str]]] = []
    for path in sorted(_SRC.rglob("*.py")):
        # UTF-8 is stated explicitly: the default is platform-dependent, and
        # on Windows (cp1252) reading a source file containing a non-ASCII
        # byte raised UnicodeDecodeError, failing the whole scan.
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - the package must parse anyway
            continue
        lines = text.split("\n")
        out.extend((path, node, lines) for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))
    return out


def _handler_body(handler: ast.ExceptHandler) -> str:
    return "\n".join(ast.unparse(statement) for statement in handler.body)


def _reports_failure(handler: ast.ExceptHandler) -> bool:
    body = _handler_body(handler)
    return any(marker in body for marker in ("raise", "logger", "log.", "warning", "error", "exception"))


def _has_rationale(handler: ast.ExceptHandler, lines: list[str]) -> bool:
    """An inline comment on the ``except`` line, or a ``noqa`` with reason."""
    header = lines[handler.lineno - 1]
    return "#" in header


def test_every_broad_silent_handler_documents_why() -> None:
    """``except Exception`` without raise/log needs an inline rationale.

    A probe (``except ImportError: optional = False``) is fine; a swallowed
    ``Exception`` on a user-visible path is not. The distinction lives in the
    comment, so require one.
    """
    undocumented: list[str] = []
    for path, handler, lines in _handlers():
        if handler.type is None:
            continue  # bare ``except:`` is caught by the test below instead
        if _reports_failure(handler):
            continue
        exception_name = ast.unparse(handler.type)
        if "Exception" not in exception_name and "BaseException" not in exception_name:
            continue
        if _has_rationale(handler, lines):
            continue
        relative = path.relative_to(_SRC.parent.parent)
        undocumented.append(f"{relative}:{handler.lineno}  except {exception_name}")

    assert not undocumented, (
        "these handlers swallow every exception without logging or re-raising, and say nothing about why. "
        "Either report the failure, or add an inline comment explaining the probe/fallback:\n  "
        + "\n  ".join(undocumented)
    )


def test_base_exception_is_only_caught_where_documented() -> None:
    """Catching ``BaseException`` stops Ctrl-C from interrupting.

    ``KeyboardInterrupt`` and ``SystemExit`` derive from ``BaseException``, not
    ``Exception``. The batch engine catches it deliberately so a forward failure
    can free slots before re-raising; anything else doing so must say why.
    """
    offenders: list[str] = []
    for path, handler, lines in _handlers():
        if handler.type is None:
            continue
        if "BaseException" not in ast.unparse(handler.type):
            continue
        if _has_rationale(handler, lines):
            continue
        offenders.append(f"{path.relative_to(_SRC.parent.parent)}:{handler.lineno}")
    assert not offenders, "except BaseException with no documented reason (it swallows Ctrl-C):\n  " + "\n  ".join(
        offenders
    )


def test_the_scan_actually_finds_handlers() -> None:
    """Guard against a vacuous pass (a broken scan would find nothing)."""
    handlers = _handlers()
    assert len(handlers) > 100, f"the scan found only {len(handlers)} except handlers — it probably broke"
