"""Shared argument-validation helpers for the compression CLIs.

``llm decompose``, ``llm prune`` and ``llm quantize`` all take a ``--model``
checkpoint and an ``--output`` path, and all three must refuse the same two
destructive misuses:

- ``--output`` naming the source checkpoint (writing would destroy it), and
- ``--output`` naming an existing directory (the atomic-save helper writes a
  sibling ``.tmp`` and renames it, surfacing as an ``IsADirectoryError`` that
  names the temp file rather than the path the user typed).

The three commands carried their own copies of these guards. They had already
drifted: only ``quantize`` rejected an existing directory, so the same typo
against ``llm prune`` produced a confusing error while ``llm quantize``
answered with a usable one. Keeping the guards here makes the contract shared
by construction.
"""

from __future__ import annotations

from pathlib import Path

import typer


def die(message: str) -> typer.Exit:
    """Print a one-line ``Error:`` to stderr and exit 1.

    Raises rather than returns: every caller uses it as the terminal statement
    of a validation branch, so returning would invite a caller to continue with
    an invalid value.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def validate_model_path(model: Path) -> None:
    """Require ``--model`` to be a file, or a checkpoint stem with v2 sidecars.

    A regular file covers the legacy single-file blob and the ``.safetensors``
    sidecar; a bare stem / ``<stem>.pt`` spelling resolves to the v2 split trio
    that :func:`llm.cli._model_loading.load_trained_model` loads.
    """
    from llm.cli._model_loading import checkpoint_exists

    if model.is_dir():
        die(f"--model {model} is a directory, not a regular file.")
    if not checkpoint_exists(model):
        die(f"--model {model} does not exist (no file and no v2 checkpoint sidecars at this stem).")


def reject_clobbering_output(output: Path, model: Path) -> None:
    """Refuse an ``--output`` that would destroy the source or cannot be a file.

    Both checks are usage errors, reported before any expensive work starts:

    - ``--output`` resolving to the same path as ``--model`` would irretrievably
      replace the original weights after an expensive pass. The comparison
      resolves symlinks and relative paths so ``out.pt``, ``./out.pt`` and a
      symlink to ``out.pt`` are all caught.
    - ``--output`` naming an existing directory would surface from the
      atomic-save helper as a raw ``IsADirectoryError`` naming the ``.tmp``
      sibling, not the path the user typed.
    """
    if output.resolve() == model.resolve():
        die(
            f"--output ({output}) must not be the same file as --model ({model}); "
            "writing would destroy the source checkpoint. Choose a different output path."
        )
    if output.exists() and output.is_dir():
        die(f"--output ({output}) is an existing directory; must be a file path.")


__all__ = ["die", "reject_clobbering_output", "validate_model_path"]
