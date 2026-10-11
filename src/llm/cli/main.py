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

import typer

from llm.cli.decompose import app as decompose_app
from llm.cli.migrate_ckpt import app as migrate_app
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

app.add_typer(train_app, name="train", help="Train a model (LM / SFT / DPO / ...).")
app.add_typer(serve_app, name="serve", help="Start the OpenAI-compatible inference server.")
app.add_typer(quantize_app, name="quantize", help="Post-training quantization (GPTQ / FP8).")
app.add_typer(prune_app, name="prune", help="Weight-prune a model checkpoint.")
app.add_typer(decompose_app, name="decompose", help="Low-rank (SVD U-V) factorize a model.")
app.add_typer(migrate_app, name="migrate-ckpt", help="Convert legacy v0.0.5 checkpoints to the v2 layout.")


if __name__ == "__main__":
    app()
