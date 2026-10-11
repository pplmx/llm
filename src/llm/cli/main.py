"""The unified ``llm`` CLI: ``llm <subcommand>``.

Composes the framework's commands as subcommands so the entry point is a
single discoverable ``llm`` binary instead of six flat ``llm train`` /
``llm serve`` / ``llm prune`` / ``llm decompose`` / ``llm quantize`` /
``llm migrate-ckpt`` programs. Each command module exposes a Typer app that is
registered below.

Usage::

    llm train --task lm --epochs 1 --batch-size 32
    llm serve --host 127.0.0.1 --port 8000
    llm quantize gptq --model ckpt.pt --output ckpt-int4.pt
    llm prune --model ckpt.pt --output pruned.pt
    llm decompose --model ckpt.pt --output lowrank.pt
    llm migrate-ckpt path/to/latest.pt
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import typer

from llm.cli import migrate_ckpt
from llm.cli.decompose import app as decompose_app
from llm.cli.prune import app as prune_app
from llm.cli.quantize import app as quantize_app
from llm.serving.cli import app as serve_app
from llm.training.train import app as train_app

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    add_completion=False,
    help="llm — a modular PyTorch framework for training and serving LLMs.",
)


def _version_callback(value: bool) -> None:
    """Eager ``--version``/``-V``: print the version and exit before subcommand
    or "missing command" resolution. (A typer group callback can't hold a
    positional argument, but an eager option short-circuits cleanly and leaves
    subcommand dispatch untouched.)"""
    if value:
        typer.echo(f"llm {importlib.metadata.version('llm')}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """The ``llm`` command group."""


app.add_typer(train_app, name="train", help="Train a model (LM / SFT / DPO / ...).")
app.add_typer(serve_app, name="serve", help="Start the OpenAI-compatible inference server.")
app.add_typer(quantize_app, name="quantize", help="Post-training quantization (GPTQ / FP8).")
app.add_typer(prune_app, name="prune", help="Weight-prune a model checkpoint.")
app.add_typer(decompose_app, name="decompose", help="Low-rank (SVD U-V) factorize a model.")


# ``migrate-ckpt`` takes a POSITIONAL path argument, which cannot live on a
# typer group callback (the positional collides with subcommand dispatch), so
# it is registered as a plain root command that delegates to the module's ``main``.
@app.command("migrate-ckpt", help="Convert legacy v0.0.5 checkpoints to the v2 layout.")
def migrate_ckpt_cmd(
    path: Path = typer.Argument(..., help="Path to the legacy `.pt` checkpoint or its stem."),
    in_place: bool = typer.Option(False, "--in-place", help="Delete the legacy `.pt` after conversion."),
    verify: bool = typer.Option(False, "--verify", help="Round-trip check the new layout; exit 2 on mismatch."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and write nothing."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing split-layout trio."),
) -> None:
    """Convert a legacy v0.0.5 ``.pt`` checkpoint to the v2 split layout."""
    migrate_ckpt.main(path=path, in_place=in_place, verify=verify, dry_run=dry_run, overwrite=overwrite)


if __name__ == "__main__":
    app()
