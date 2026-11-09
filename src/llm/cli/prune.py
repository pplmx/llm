"""``llm prune`` — weight-prune a pretrained model checkpoint.

Thin wrapper over :func:`llm.quantization.prune.prune_model` (the sibling of the
``llm quantize`` CLI). Loads a bare ``torch.save`` model blob (a ``DecoderModel``,
plain or already quantized), zeroes a fraction of each ``nn.Linear``'s weights via
a persistent ``weight_mask``, and writes a new pruned blob atomically.

Exit codes:

    0 — success (sparsity reported)
    1 — argument validation failed (bad ratio / method / clobbered output)
    2 — runtime failure (model load, pruning, save)
    (Note: typer itself exits 2 — the same as a runtime failure — for
    argparse-level usage errors like a missing required --model/--output,
    so a bare mis-typed invocation is not distinguishable from an in-run
    crash by exit code alone; the stderr message differs.)
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from llm.cli._compression_args import die, reject_clobbering_output, validate_model_path

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
    help="Prune (sparsify) the linear weights of a model checkpoint.",
)


def _validate_ratio(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        die(f"--ratio must be in (0, 1); got {ratio}.")
    return ratio


def _validate_method(method: str) -> str:
    if method not in ("magnitude", "random"):
        die(f"--method must be 'magnitude' or 'random'; got {method!r}.")
    return method


def _atomic_save_blob(module, output: Path) -> None:
    """Write ``module`` atomically (temp + fsync + rename)."""
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    torch.save(module, tmp)
    # fsync on a READ-WRITE handle: Windows `os.fsync` rejects read-only handles
    # with EBADF (POSIX tolerates them); "rb+" keeps fsync-before-rename durable
    # on both platforms.
    with tmp.open("rb+") as fh:
        os.fsync(fh.fileno())
    tmp.replace(output)


def _resolve_target_modules(target_modules: str | None) -> list[str] | None:
    """Parse the comma-separated ``--target-modules`` flag into a list.

    Empty / whitespace-only input normalizes to ``None`` ("all") so the CLI
    summary echoes ``all`` only when every nn.Linear is actually targeted. The
    old ``[]`` filtered to ZERO layers and then failed at runtime.
    """
    if target_modules is None:
        return None
    parsed = [tok.strip() for tok in target_modules.split(",") if tok.strip()]
    return parsed or None


@app.callback()
def prune(
    model: Path = typer.Option(
        ...,
        "--model",
        help="Checkpoint to load: v2 split stem (e.g. 'checkpoints/epoch_5'), the '.pt' spelling, a '.safetensors' sidecar, a legacy single-file .pt blob, or a bare torch.save'd module.",
    ),
    output: Path = typer.Option(..., "--output", help="Output path for the pruned model blob."),
    ratio: float = typer.Option(0.5, "--ratio", help="Fraction of each Linear's weights to zero (0 < ratio < 1)."),
    method: str = typer.Option("magnitude", "--method", help="'magnitude' (keep largest |W|) or 'random'."),
    target_modules: str | None = typer.Option(
        None, "--target-modules", help="Comma-separated module-name substrings to prune (default: all Linear)."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Seed for 'random' pruning (reproducibility)."),
) -> None:
    """Prune a pretrained model's linear weights and save a new blob."""
    _validate_ratio(ratio)
    _validate_method(method)
    if method == "magnitude" and seed is not None:
        # Only "random" pruning consumes the seed; passing one
        # with magnitude silently does nothing — surface it so the user isn't
        # misled into thinking the run is reproducible.
        typer.echo(
            f"Warning: --seed {seed} is ignored with --method magnitude (only 'random' pruning uses the seed).",
            err=True,
        )
    validate_model_path(model)
    reject_clobbering_output(output, model)

    from llm.cli._model_loading import load_trained_model
    from llm.quantization.prune import PruningConfig, prune_model

    try:
        typer.echo(f"Loading model from {model}...")
        model_obj = load_trained_model(model)
    except Exception as exc:
        typer.echo(f"Error: failed to load model {model}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    config = PruningConfig(
        ratio=ratio,
        method=method,
        target_modules=_resolve_target_modules(target_modules),
        random_seed=seed,
    )
    try:
        typer.echo(f"Pruning {method} ratio={ratio} target_modules={config.target_modules or 'all'}...")
        sparsity = prune_model(model_obj, config)
    except Exception as exc:
        typer.echo(f"Error: pruning failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        _atomic_save_blob(model_obj, output)
    except Exception as exc:
        typer.echo(f"Error: failed to save pruned model to {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Pruned model saved to {output}")
    typer.echo(f"Achieved sparsity: {sparsity:.2%}")


if __name__ == "__main__":
    app()
