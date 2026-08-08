"""Assertion helpers for CLI output that may contain ANSI escape codes."""

from __future__ import annotations

import re

# Covers CSI sequences (colors, styles) plus OSC/other escapes.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (e.g. rich/typer styling) from output."""
    return _ANSI_RE.sub("", text)
