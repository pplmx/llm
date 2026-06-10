"""Utilities for deprecated import path shims."""

from __future__ import annotations

import warnings


def warn_deprecated(old_path: str, new_path: str) -> None:
    warnings.warn(
        f"Importing from '{old_path}' is deprecated; use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
