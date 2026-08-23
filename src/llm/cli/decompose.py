"""``llm-decompose`` — low-rank (SVD U-V) factorize a pretrained model.

Thin wrapper over :func:`llm.quantization.lowrank.decompose_model` (the sibling
of ``llm-prune`` in the compression family). Loads a bare ``torch.save`` model
blob, factorizes each ``nn.Linear`` weight into ``u @ v`` at the requested rank,
and writes a new blob atomically, reporting rank / compression ratio / Frobenius
reconstruction error.

Exit codes:

    0 — success
    1 — argument validation failed (bad/conflicting rank, clobbered output)
    2 — runtime failure (model load, decomposition, save)
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    help="Low-rank (SVD U-V) factorize the linear weights of a model checkpoint.",
)


def _die(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _reject_clobbering_input(output: Path, model: Path) -> None:
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
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    torch.save(module, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(output)


def _resolve_target_modules(target_modules: str | None) -> list[str] | None:
    if target_modules is None:
        return None
    return [tok.strip() for tok in target_modules.split(",") if tok.strip()]


@app.command()
def decompose(
    model: Path = typer.Option(..., "--model", help="Path to model blob (.pt torch.save of a DecoderModel)."),
    output: Path = typer.Option(..., "--output", help="Output path for the low-rank model blob."),
    rank: int | None = typer.Option(None, "--rank", help="Explicit rank r (mutually exclusive with --rank-ratio)."),
    rank_ratio: float | None = typer.Option(
        None, "--rank-ratio", help="Auto rank = ratio * min(out, in) (mutually exclusive with --rank)."
    ),
    target_modules: str | None = typer.Option(
        None, "--target-modules", help="Comma-separated module-name substrings to decompose (default: all Linear)."
    ),
) -> None:
    """Low-rank factorize a pretrained model's linear weights."""
    if (rank is None) == (rank_ratio is None):
        _die("must supply exactly one of --rank or --rank-ratio.")
    if rank is not None and rank <= 0:
        _die(f"--rank must be > 0; got {rank}.")
    if rank_ratio is not None and not 0.0 < rank_ratio <= 1.0:
        _die(f"--rank-ratio must be in (0, 1]; got {rank_ratio}.")
    _validate_model_path(model)
    _reject_clobbering_input(output, model)

    import torch

    from llm.quantization.lowrank import LowRankConfig, decompose_model
    from llm.utils.serialization import register_framework_safe_globals

    try:
        typer.echo(f"Loading model from {model}...")
        register_framework_safe_globals()
        model_obj = torch.load(model, map_location="cpu", weights_only=True)
    except Exception as exc:
        typer.echo(f"Error: failed to load model {model}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    config = LowRankConfig(
        rank=rank,
        rank_ratio=rank_ratio,
        target_modules=_resolve_target_modules(target_modules),
    )
    try:
        stats = decompose_model(model_obj, config)
    except Exception as exc:
        typer.echo(f"Error: decomposition failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        _atomic_save_blob(model_obj, output)
    except Exception as exc:
        typer.echo(f"Error: failed to save low-rank model to {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Low-rank model saved to {output}")
    typer.echo(f"Compression ratio: {stats['compression_ratio']:.3f}x")
    typer.echo(f"Mean reconstruction error: {stats['relative_error']:.4f}")
    typer.echo(f"Layers decomposed: {len(stats['layers'])}")
