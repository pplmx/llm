"""Assertion helpers for CLI output that may contain ANSI escape codes."""

from __future__ import annotations

import os
import re

# Covers CSI sequences (colors, styles) plus OSC/other escapes.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (e.g. rich/typer styling) from output."""
    return _ANSI_RE.sub("", text)


def help_render_diag(output: str = "") -> str:
    """Build a diagnostic string for CLI ``--help`` rendering failures.

    CLI help is rendered by typer/rich at a width baked from the ``TERMINAL_WIDTH``
    env var the first time ``typer.rich_utils`` is imported; a narrow width can make
    every option name vanish from the output (an empty options box) without failing
    the exit code. Embed this in help-test assertion messages so a CI-only render
    failure is self-explanatory: env widths, installed typer/rich versions, the
    ``llm`` package location, the baked ``typer.rich_utils.MAX_WIDTH``, and a snippet
    of the rendered output.
    """
    from importlib import metadata

    def _get_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover
            return "?"

    # A diagnostic must never raise and mask the real assertion failure, so we
    # tolerate any import/metadata error here and report it inline instead.
    try:
        import llm
    except Exception as exc:  # noqa: BLE001 - diagnostic must not raise
        llm_file = f"<import error: {exc}>"
    else:
        llm_file = getattr(llm, "__file__", "<no __file__>")

    try:
        import typer.rich_utils as _rich_utils

        max_width = _rich_utils.MAX_WIDTH
    except Exception as exc:  # noqa: BLE001 - diagnostic must not raise
        max_width = f"<error: {exc}>"

    stripped = strip_ansi(output)
    return (
        "help-render diag: "
        f"env[TERMINAL_WIDTH]={os.environ.get('TERMINAL_WIDTH')!r} "
        f"env[COLUMNS]={os.environ.get('COLUMNS')!r} "
        f"env[LINES]={os.environ.get('LINES')!r}; "
        f"typer={_get_version('typer')} rich={_get_version('rich')}; "
        f"llm={llm_file}; "
        f"typer.rich_utils.MAX_WIDTH={max_width}; "
        f"raw[:90]={output[:90]!r}; "
        f"raw_full={output[:2000]!r}; "
        f"stripped_full={stripped!r}"
    )
