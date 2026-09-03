"""``llm-prune`` — weight-prune a pretrained model checkpoint.

Thin wrapper over :func:`llm.quantization.prune.prune_model` (the sibling of the
``llm-quantize`` CLI). Loads a bare ``torch.save`` model blob (a ``DecoderModel``,
plain or already quantized), zeroes a fraction of each ``nn.Linear``'s weights via
a persistent ``weight_mask``, and writes a new pruned blob atomically.

Exit codes:

    0 — success (sparsity reported)
    1 — argument validation failed (bad ratio / method / clobbered output)
    2 — runtime failure (model load, pruning, save)
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    help="Prune (sparsify) the linear weights of a model checkpoint.",
)


def _die(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _validate_ratio(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        _die(f"--ratio must be in (0, 1); got {ratio}.")
    return ratio


def _validate_method(method: str) -> str:
    if method not in ("magnitude", "random"):
        _die(f"--method must be 'magnitude' or 'random'; got {method!r}.")
    return method


def _reject_clobbering_input(output: Path, model: Path) -> None:
    """Refuse to overwrite the source checkpoint in place (RIL ISS-161 pattern)."""
    if output.resolve() == model.resolve():
        _die(
            f"--output ({output}) must not be the same file as --model ({model}); "
            "writing would destroy the source checkpoint. Choose a different output path."
        )


def _validate_model_path(model: Path) -> None:
    if not model.exists():
        _die(f"--model {model} does not exist.")
    if not model.is_file():
        _die(f"--model {model} is not a regular file.")


def _atomic_save_blob(module, output: Path) -> None:
    """Write ``module`` atomically (temp + fsync + rename)."""
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    torch.save(module, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(output)


def _resolve_target_modules(target_modules: str | None) -> list[str] | None:
    """Parse the comma-separated ``--target-modules`` flag into a list.

    Empty / whitespace-only input normalizes to ``None`` ("all") so the CLI
    summary echoes ``all`` only when every nn.Linear is actually targeted. The
    old ``[]`` filtered to ZERO layers and then failed at runtime (RIL ISS-330).
    """
    if target_modules is None:
        return None
    parsed = [tok.strip() for tok in target_modules.split(",") if tok.strip()]
    return parsed or None


@app.command()
def prune(
    model: Path = typer.Option(..., "--model", help="Path to model blob (.pt torch.save of a DecoderModel)."),
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
    _validate_model_path(model)
    _reject_clobbering_input(output, model)

    import torch

    from llm.quantization.prune import PruningConfig, prune_model
    from llm.utils.serialization import register_framework_safe_globals

    try:
        typer.echo(f"Loading model from {model}...")
        register_framework_safe_globals()
        model_obj = torch.load(model, map_location="cpu", weights_only=True)
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
