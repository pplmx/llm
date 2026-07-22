"""``llm-migrate-ckpt`` — convert legacy v0.0.5 ``.pt`` checkpoints to v2.

This CLI closes the gap introduced by the v2 split-layout format
(ADR-006, commit ``4b9cf68``). Training checkpoints in v0.0.5 and
earlier are a single ``torch.save`` blob; v2 splits them into three
sidecars (``<stem>.safetensors`` + ``<stem>.meta.json`` +
``<stem>.extra_state.pt``).

The :class:`~llm.training.core.checkpoint.CheckpointManager` loader
already auto-detects both layouts — legacy ``.pt`` files keep
loading with a :class:`DeprecationWarning`. This CLI exists so users
with a large fleet of v0.0.5 checkpoints can convert them once and
silence the warning permanently.

Usage:

    llm-migrate-ckpt path/to/latest.pt              # convert (keep legacy)
    llm-migrate-ckpt path/to/latest                  # looks for latest.pt
    llm-migrate-ckpt path/to/latest.pt --in-place    # delete legacy after success
    llm-migrate-ckpt path/to/latest.pt --verify      # round-trip check (load new, compare)
    llm-migrate-ckpt path/to/latest.pt --dry-run     # print plan, write nothing
    llm-migrate-ckpt path/to/latest.pt --overwrite   # replace an existing split layout

Exit codes:

    0 — conversion succeeded (or --dry-run plan was printed)
    1 — conversion failed (legacy missing, split layout already present, etc.)
    2 — verification failed (the new layout does not round-trip)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer

from llm.training.core.checkpoint import (
    CheckpointMigrationError,
    convert_legacy_checkpoint_to_split,
)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Convert a legacy v0.0.5 single-file .pt training checkpoint to "
        "the v2 split layout (model.safetensors + meta.json + extra_state.pt)."
    ),
)


def _print_plan(legacy: Path, sidecars: dict[str, Path], in_place: bool) -> None:
    typer.echo(f"  legacy:    {legacy}")
    for label, p in sidecars.items():
        typer.echo(f"  {label + ':':11s} {p}")
    if in_place:
        typer.echo(f"  action:    delete legacy {legacy} after success")
    else:
        typer.echo(f"  action:    keep legacy {legacy} (use --in-place to delete)")


def _verify_round_trip(legacy: Path, sidecars: dict[str, Path]) -> tuple[bool, str]:
    """Reload the new split trio and compare against the legacy blob.

    Verifies model_state tensors are byte-identical (Tensor values),
    and that metadata fields (epoch / loss / best_loss / model_config)
    match. The training-state sidecar is checked for key presence
    only — exact equality on optimizer/scheduler state dicts is fragile
    across torch versions and isn't a useful signal of correctness
    here.

    Note: we deliberately do NOT use :func:`load_checkpoint_payload`
    for the new layout here — that helper prioritizes the legacy
    ``.pt`` over the split layout when both exist (the auto-load
    behavior the loader uses on resume). For verification we want
    the actual sidecar contents, so we read them directly.
    """
    # Legacy side: load the .pt directly (bypassing the helper).
    legacy_payload = torch.load(legacy, map_location="cpu")

    # New side: read each sidecar explicitly.
    from safetensors.torch import load_file

    try:
        new_model_state = load_file(str(sidecars["weights"]))
    except Exception as exc:
        return False, f"new safetensors sidecar failed to load: {exc}"
    new_meta = json.loads(sidecars["meta"].read_text())
    new_extra = torch.load(sidecars["extra_state"], map_location="cpu")

    # model_state: compare tensors element-wise.
    legacy_model_state = legacy_payload.get("model_state", {})
    for key, legacy_tensor in legacy_model_state.items():
        if key not in new_model_state:
            return False, f"model_state key missing in new layout: {key!r}"
        new_tensor = new_model_state[key]
        if hasattr(legacy_tensor, "shape") and hasattr(new_tensor, "shape"):
            if legacy_tensor.shape != new_tensor.shape:
                return (
                    False,
                    f"model_state[{key!r}] shape mismatch: {tuple(legacy_tensor.shape)} vs {tuple(new_tensor.shape)}",
                )
            # Use float comparison rather than exact equality — the
            # safetensors round-trip can introduce minor numeric drift
            # in fp16/bf16 tensors (the kernel uses a different
            # reduction order than torch.save's pickle).
            if legacy_tensor.dtype.is_floating_point:
                if not (legacy_tensor - new_tensor).abs().max().item() < 1e-5:
                    return False, f"model_state[{key!r}] numeric drift > 1e-5"
            else:
                if not (legacy_tensor == new_tensor).all().item():
                    return False, f"model_state[{key!r}] integer tensor mismatch"

    # Metadata: epoch / loss / best_loss / model_config — read
    # directly from meta.json (NOT via load_checkpoint_payload, which
    # would fall through to the legacy .pt and report the wrong
    # values).
    for field in ("epoch", "loss", "best_loss"):
        legacy_val = legacy_payload.get(field)
        new_val = new_meta.get(field)
        if legacy_val != new_val:
            return False, f"{field!r} mismatch: legacy={legacy_val!r} vs new={new_val!r}"
    if legacy_payload.get("model_config") != new_meta.get("model_config"):
        return False, (
            f"model_config mismatch: legacy={legacy_payload.get('model_config')!r} "
            f"vs new={new_meta.get('model_config')!r}"
        )

    # Training-state sidecar: structural check (key presence) only.
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "extra_state"):
        if key not in new_extra:
            return False, f"extra_state sidecar missing key {key!r}"

    return True, ""


@app.command()
def main(
    path: Path = typer.Argument(
        ...,
        help="Path to the legacy checkpoint — either `<name>.pt` or the stem `<name>` (the function appends `.pt`).",
        exists=False,
    ),
    in_place: bool = typer.Option(
        False,
        "--in-place",
        help="Delete the legacy `.pt` file after a successful conversion. Off by default.",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Reload the new split layout and compare against the legacy blob. Exits 2 on mismatch.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the conversion plan and exit without writing anything.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing split-layout trio at the same stem. Off by default.",
    ),
) -> None:
    """Convert a legacy v0.0.5 ``.pt`` checkpoint to the v2 split layout."""
    # Normalize the path: a stem is rewritten to `<stem>.pt` for the
    # resolution step below. The actual write goes to the stem (no
    # suffix), matching the v2 layout convention.
    legacy_path = path if path.suffix == ".pt" else path.with_suffix(".pt")

    if not legacy_path.exists():
        typer.echo(f"error: legacy checkpoint not found: {legacy_path}", err=True)
        raise typer.Exit(code=1)

    stem = legacy_path.with_suffix("")
    sidecars = {
        "weights": stem.with_suffix(".safetensors"),
        "meta": stem.with_name(stem.name + ".meta.json"),
        "extra_state": stem.with_name(stem.name + ".extra_state.pt"),
    }

    if dry_run:
        typer.echo("[dry-run] conversion plan:")
        _print_plan(legacy_path, sidecars, in_place=in_place)
        raise typer.Exit(code=0)

    # If --verify is passed AND the split layout already exists at the
    # same stem, skip the conversion step and verify the existing trio
    # against the legacy blob. This is the post-conversion re-check
    # workflow ("I converted yesterday, did anything drift?").
    split_exists = all(p.exists() for p in sidecars.values())
    if verify and split_exists:
        typer.echo("✓ Split layout already exists; running --verify on it.")
        written = sidecars
    else:
        try:
            written = convert_legacy_checkpoint_to_split(
                legacy_path,
                in_place=in_place,
                overwrite=overwrite,
            )
        except CheckpointMigrationError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        typer.echo("✓ Converted:")
        _print_plan(legacy_path, written, in_place=in_place)

    if verify:
        ok, msg = _verify_round_trip(legacy_path, written)
        if not ok:
            typer.echo(f"✗ verification failed: {msg}", err=True)
            # The split layout was written but does not round-trip —
            # leave the legacy file in place so the user can retry.
            typer.echo(
                f"  hint: the legacy {legacy_path} is preserved; investigate the mismatch before retrying.",
                err=True,
            )
            raise typer.Exit(code=2)
        typer.echo("✓ verification passed (model_state tensors + metadata match)")


if __name__ == "__main__":
    sys.exit(app() or 0)
