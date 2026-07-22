"""Checkpoint save/load with atomic-write semantics and split file layout.

This module is the v2 checkpoint format (introduced with the
checkpoint-format unification slice, see ADR-006):

- **``<name>.safetensors``** — model weights (state dict only). Saved
  with ``safetensors.torch.save_file`` so the file is zero-copy
  loadable, free of pickle deserialization risk, and aligned with
  the HuggingFace compat layer (``llm.compat.hf_publisher``).
- **``<name>.meta.json``** — JSON-encoded training metadata: epoch,
  loss, best_loss, model_config, format_version.
- **``<name>.extra_state.pt``** — the optimizer / scheduler / scaler
  state dicts + the optional ``extra_state`` dict. These are
  arbitrary Python objects (Adam moment estimates, lr-scheduler
  step counts, ``StreamDataState`` instances) that don't fit cleanly
  into safetensors or JSON, so they stay as a single ``torch.save``
  blob — the same pattern as the legacy format, just isolated from
  the model weights.

The :class:`CheckpointManager` API is unchanged: callers still call
:meth:`CheckpointManager.save_checkpoint` and
:meth:`CheckpointManager.load_checkpoint`. The only observable
difference is the file layout on disk. The legacy single-file
``<name>.pt`` layout from v0.0.5 and earlier is auto-detected and
loaded by :meth:`load_checkpoint` — a ``DeprecationWarning`` fires
once per load, recommending ``llm-migrate-ckpt <path>`` (a future
CLI helper) for in-place conversion.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler

from llm.training.core.config import CheckpointConfig
from llm.training.distributed import load_model_state_dict, model_state_dict

logger = logging.getLogger(__name__)

#: Bumped on any backward-incompatible change to the on-disk schema.
#: ``meta.json`` carries a ``format_version`` field; future migrations
#: gate on this constant.
CHECKPOINT_FORMAT_VERSION = "2.0"

#: Filename suffix for the safetensors weights sidecar.
SAFETENSORS_SUFFIX = ".safetensors"

#: Filename suffix for the JSON metadata sidecar.
META_SUFFIX = ".meta.json"

#: Filename suffix for the pickled training-state sidecar.
EXTRA_STATE_SUFFIX = ".extra_state.pt"

#: Legacy single-file extension (v0.0.5 and earlier). Detected and
#: loaded by :func:`_load_legacy_checkpoint`.
LEGACY_SUFFIX = ".pt"


def _safetensors_available() -> bool:
    """True when the ``safetensors`` package is importable.

    Lazy import — the package is only required at save/load time, so
    the test suite and minimal envs can still import this module.
    """
    try:
        import safetensors  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_checkpoint_paths(name_or_path: str | Path) -> tuple[Path, Path, Path, Path]:
    """Resolve ``name_or_path`` to all four candidate paths.

    Always returns the full set:
      - ``legacy_pt_path`` — the v0.0.5 single-file layout
      - ``safetensors_path`` — model weights sidecar
      - ``meta_path`` — JSON metadata sidecar
      - ``extra_state_path`` — pickled training-state sidecar

    ``name_or_path`` can be a bare name (``"latest"``), an absolute
    path to ``<name>.pt`` (the legacy extension is stripped to find
    the stem), or a path without an extension. The returned paths
    share the same stem, just with different suffixes — even for
    relative inputs.
    """
    p = Path(name_or_path)
    stem = p.name[: -len(LEGACY_SUFFIX)] if p.suffix == LEGACY_SUFFIX else p.stem
    base = p.parent / stem
    legacy = base.with_suffix(LEGACY_SUFFIX) if p.suffix == LEGACY_SUFFIX else base.with_name(stem + LEGACY_SUFFIX)
    return (
        legacy,
        base.with_name(stem + SAFETENSORS_SUFFIX),
        base.with_name(stem + META_SUFFIX),
        base.with_name(stem + EXTRA_STATE_SUFFIX),
    )


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write ``payload`` to ``target`` atomically via temp + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)


def _save_weights_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    """Save ``state_dict`` to ``path`` as safetensors (contiguous + clone).

    Raises ``ImportError`` when safetensors is not installed — the
    caller is expected to gate on :func:`_safetensors_available` or
    :data:`llm.compat.hf_publisher.SAFETENSORS_AVAILABLE`.
    """
    from safetensors.torch import save_file

    # safetensors rejects non-contiguous tensors AND non-tensor values.
    # Filter to tensors only, then clone to detach from any gradient /
    # view chains and to make the file truly standalone. Optimizer
    # state dicts (which contain ints for step counts, etc.) go into
    # the .extra_state.pt sidecar instead.
    contiguous = {k: v.detach().contiguous().clone() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(contiguous, str(tmp))
    tmp.replace(path)


def _save_metadata_json(meta: dict[str, Any], path: Path) -> None:
    """Save ``meta`` to ``path`` as pretty-printed JSON.

    ``meta`` MUST be JSON-serializable — if you add a field that
    isn't, encode it explicitly (e.g. ``str(pathlib.Path)``) before
    passing it in.
    """
    payload = json.dumps(meta, indent=2, sort_keys=True, default=str)
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _save_extra_state_pt(
    optimizer_state: dict[str, Any] | None,
    scheduler_state: dict[str, Any] | None,
    scaler_state: dict[str, Any] | None,
    extra_state: dict[str, Any] | None,
    path: Path,
) -> None:
    """Save training-state sidecars to ``path`` as ``torch.save``.

    Wraps all four sub-states in a single dict so the loader can
    tell them apart — the top-level keys mirror the legacy single-file
    schema, but only the training-state slots are present.
    """
    blob = {
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "extra_state": extra_state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def _load_split_checkpoint(stem_dir: Path) -> dict[str, Any] | None:
    """Load the v2 three-file layout; return ``None`` if incomplete.

    A "complete" split checkpoint has all three sidecars. Missing
    files → ``None`` (the caller will try the legacy layout).
    """
    safetensors_path = stem_dir.with_name(stem_dir.name + SAFETENSORS_SUFFIX)
    meta_path = stem_dir.with_name(stem_dir.name + META_SUFFIX)
    extra_state_path = stem_dir.with_name(stem_dir.name + EXTRA_STATE_SUFFIX)
    if not (safetensors_path.exists() and meta_path.exists() and extra_state_path.exists()):
        return None

    from safetensors.torch import load_file

    model_state = load_file(str(safetensors_path))
    # Re-wrap into a torch state_dict (still tensors, just not safetensors-format).
    model_state = {k: v for k, v in model_state.items()}

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        logger.warning(
            "Checkpoint meta.json format_version=%s does not match expected %s; "
            "loading anyway but check for compatibility.",
            meta.get("format_version"),
            CHECKPOINT_FORMAT_VERSION,
        )

    extra = torch.load(extra_state_path, map_location="cpu")
    return {
        "model_state": model_state,
        "model_config": meta.get("model_config"),
        "epoch": meta.get("epoch", 0),
        "loss": meta.get("loss"),
        "best_loss": meta.get("best_loss", float("inf")),
        "optimizer_state": extra.get("optimizer_state"),
        "scheduler_state": extra.get("scheduler_state"),
        "scaler_state": extra.get("scaler_state"),
        "extra_state": extra.get("extra_state"),
        "format_version": meta.get("format_version"),
    }


def _load_legacy_checkpoint(legacy_path: Path) -> dict[str, Any] | None:
    """Load a v0.0.5-era single-file checkpoint; return ``None`` if absent."""
    if not legacy_path.exists():
        return None
    warnings.warn(
        f"Loading legacy checkpoint format from {legacy_path}. "
        "This format is deprecated; run `llm-migrate-ckpt <path>` to convert "
        "to the v2 split layout (model.safetensors + meta.json + extra_state.pt).",
        DeprecationWarning,
        stacklevel=3,
    )
    return torch.load(legacy_path, map_location="cpu")


def load_checkpoint_payload(path: str | Path) -> dict[str, Any] | None:
    """Public helper: load a checkpoint from ``path`` in either layout.

    Resolution order (when ``path`` ends in ``.pt``):
      1. Legacy single-file ``<path>`` — preferred if it exists (no
         deprecation warning in that case is *not* correct: we still
         emit the warning, but the file is loaded).
      2. Split three-file layout at the same stem — when the legacy
         file is absent, look for ``<stem>.safetensors`` /
         ``<stem>.meta.json`` / ``<stem>.extra_state.pt``.

    Returns the unified dict (keys: ``model_state``, ``model_config``,
    ``epoch``, ``loss``, ``best_loss``, ``optimizer_state``,
    ``scheduler_state``, ``scaler_state``, ``extra_state``,
    ``format_version``) or ``None`` if neither layout is present at
    ``path``.

    Useful for callers that want to introspect a checkpoint without
    instantiating a full :class:`CheckpointManager`.
    """
    legacy_path, _safetensors, _meta, _extra = _resolve_checkpoint_paths(path)
    stem = legacy_path.with_suffix("")

    # Legacy first — if the legacy .pt exists at the exact path the
    # caller gave, that's the most explicit signal. The split layout
    # only wins when the legacy file is missing.
    if legacy_path.exists():
        return _load_legacy_checkpoint(legacy_path)

    # Split layout at the same stem.
    split = _load_split_checkpoint(stem)
    if split is not None:
        return split

    return None


# ---------------------------------------------------------------------------
# Conversion: legacy v0.0.5 single-file .pt -> v2 split trio
# ---------------------------------------------------------------------------


class CheckpointMigrationError(RuntimeError):
    """Raised when a legacy->split checkpoint migration cannot proceed.

    Distinct from generic :class:`RuntimeError` so callers (CLI,
    tests) can catch migration-specific failures without masking
    unrelated runtime errors. Examples: legacy file missing, split
    layout already present, both layouts coexist (ambiguous).
    """


def convert_legacy_checkpoint_to_split(
    path: str | Path,
    *,
    in_place: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Convert a legacy v0.0.5 single-file ``.pt`` to the v2 split layout.

    Reads ``<stem>.pt``, writes three sidecars at the same stem
    (``<stem>.safetensors``, ``<stem>.meta.json``,
    ``<stem>.extra_state.pt``). The split layout is the v2 format
    that :class:`CheckpointManager` writes by default; legacy
    single-file checkpoints are auto-detected and loaded by
    :meth:`CheckpointManager.load_checkpoint` (with a
    :class:`DeprecationWarning`), so this conversion is purely
    hygienic — the loader works either way.

    Atomicity: each sidecar is written to a ``.tmp`` file in the
    same directory and then renamed over the target. If a sidecar
    write fails partway, the ``.tmp`` file may remain; the next
    successful migration overwrites it. The legacy ``.pt`` is
    NEVER touched unless ``in_place=True``.

    Args:
        path: Path to the legacy ``.pt`` (or its stem). When the
            stem is given, the function looks for ``<stem>.pt``
            alongside.
        in_place: When True, delete the legacy ``.pt`` file after
            a successful conversion. Default False (the legacy
            file is preserved so the user can verify the new
            layout before removing the old one).
        overwrite: When True, overwrite an existing split-layout
            trio at the same stem. Default False (refuse to
            clobber an already-converted checkpoint — passes
            through ``CheckpointMigrationError``).

    Returns:
        A dict mapping the three sidecar names (``"weights"``,
        ``"meta"``, ``"extra_state"``) to their resolved
        :class:`Path` on disk. Useful for the CLI to print
        "converted to: ...".

    Raises:
        CheckpointMigrationError: when the legacy file is missing,
            the split layout already exists (and ``overwrite`` is
            False), both layouts coexist at the same stem
            (ambiguous), or the path is neither a ``.pt`` nor a
            stem.
        ImportError: when ``safetensors`` is not installed.
    """
    legacy_path = Path(path)
    if legacy_path.suffix != LEGACY_SUFFIX:
        # Accept a stem; resolve to the legacy .pt next to it.
        legacy_path = legacy_path.with_suffix(LEGACY_SUFFIX)
    if not legacy_path.exists():
        raise CheckpointMigrationError(f"Legacy checkpoint not found: {legacy_path}")

    stem = legacy_path.with_suffix("")
    safetensors_path = stem.with_name(stem.name + SAFETENSORS_SUFFIX)
    meta_path = stem.with_name(stem.name + META_SUFFIX)
    extra_state_path = stem.with_name(stem.name + EXTRA_STATE_SUFFIX)

    # Refuse to clobber an existing split layout unless explicitly asked.
    if not overwrite and any(p.exists() for p in (safetensors_path, meta_path, extra_state_path)):
        raise CheckpointMigrationError(
            f"Split layout already exists at {stem.name}{{.{SAFETENSORS_SUFFIX},"
            f".{META_SUFFIX},{EXTRA_STATE_SUFFIX}}} — pass overwrite=True to replace "
            "(or move the existing sidecars aside first)."
        )

    # Load the legacy blob.
    payload = torch.load(legacy_path, map_location="cpu")

    # Write the split trio.
    _save_weights_safetensors(payload["model_state"], safetensors_path)
    _save_metadata_json(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": payload.get("epoch", 0),
            "loss": payload.get("loss"),
            "best_loss": payload.get("best_loss", float("inf")),
            "model_config": payload.get("model_config"),
        },
        meta_path,
    )
    _save_extra_state_pt(
        payload.get("optimizer_state"),
        payload.get("scheduler_state"),
        payload.get("scaler_state"),
        payload.get("extra_state"),
        extra_state_path,
    )

    # Optionally delete the legacy file.
    if in_place:
        legacy_path.unlink()

    return {
        "weights": safetensors_path,
        "meta": meta_path,
        "extra_state": extra_state_path,
    }


class CheckpointManager:
    """Save/load checkpoints with retention and atomic-write semantics.

    Writes the v2 split layout (three sidecar files per checkpoint
    name). :meth:`load_checkpoint` accepts both the new layout AND
    the legacy single-file ``.pt`` layout — auto-detected on read.
    """

    def __init__(self, config: CheckpointConfig, rank: int, logger):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.best_loss = float("inf")
        self.loaded_extra_state: dict | None = None
        self.checkpoints_saved: list[Path] = []
        if self.rank == 0:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ---- save side --------------------------------------------------------

    def _save_split(
        self,
        *,
        name: str,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, Any] | None,
        scheduler_state: dict[str, Any] | None,
        scaler_state: dict[str, Any] | None,
        epoch: int,
        loss: float,
        best_loss: float,
        model_config: dict | None,
        extra_state: dict | None,
    ) -> tuple[Path, Path, Path]:
        """Write all three sidecars for ``name``; return the paths."""
        if not _safetensors_available():
            raise ImportError(
                f"Saving checkpoint {name} requires the 'safetensors' package. "
                "Install with `uv sync --group compat` or `pip install llm[compat]`."
            )
        base = Path(self.config.checkpoint_dir) / name
        weights_path = base.with_name(base.name + SAFETENSORS_SUFFIX)
        meta_path = base.with_name(base.name + META_SUFFIX)
        extra_state_path = base.with_name(base.name + EXTRA_STATE_SUFFIX)

        _save_weights_safetensors(model_state, weights_path)
        _save_metadata_json(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "model_config": model_config,
            },
            meta_path,
        )
        _save_extra_state_pt(
            optimizer_state,
            scheduler_state,
            scaler_state,
            extra_state,
            extra_state_path,
        )
        return weights_path, meta_path, extra_state_path

    def save_checkpoint(
        self,
        epoch: int,
        model: DistributedDataParallel,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        loss: float,
        extra_state: dict | None = None,
        model_config: dict | None = None,
    ):
        if self.rank != 0:
            return

        model_state_to_save = model_state_dict(model)

        # ``save_best`` updates best_loss before writing the rest of the
        # meta, so the .meta.json best_loss is consistent across files.
        if self.config.save_best and loss < self.best_loss:
            self.best_loss = loss
            self.logger.info(f"🏆 New best model saved with loss {loss:.4f}")
            self._save_split(
                name="best",
                model_state=model_state_to_save,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                epoch=epoch,
                loss=loss,
                best_loss=self.best_loss,
                model_config=model_config,
                extra_state=extra_state,
            )

        self._save_split(
            name="latest",
            model_state=model_state_to_save,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            scaler_state=scaler.state_dict() if scaler is not None else None,
            epoch=epoch,
            loss=loss,
            best_loss=self.best_loss,
            model_config=model_config,
            extra_state=extra_state,
        )

        if (epoch + 1) % self.config.save_interval == 0:
            epoch_name = f"epoch_{epoch + 1}"
            _, _, _ = self._save_split(
                name=epoch_name,
                model_state=model_state_to_save,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                epoch=epoch,
                loss=loss,
                best_loss=self.best_loss,
                model_config=model_config,
                extra_state=extra_state,
            )
            epoch_pt = Path(self.config.checkpoint_dir) / f"{epoch_name}{LEGACY_SUFFIX}"
            self.checkpoints_saved.append(epoch_pt)
            self._cleanup_old_checkpoints()
            self.logger.debug(f"Checkpoint saved to {epoch_name}{SAFETENSORS_SUFFIX}")

    def _cleanup_old_checkpoints(self):
        while len(self.checkpoints_saved) > self.config.keep_last_n:
            oldest_pt = self.checkpoints_saved.pop(0)
            # The list tracks the legacy .pt paths for backward
            # compat, but on disk we have the split layout — clean up
            # all three sidecars at the same stem.
            stem = oldest_pt.with_suffix("")
            for suffix in (SAFETENSORS_SUFFIX, META_SUFFIX, EXTRA_STATE_SUFFIX):
                target = stem.with_name(stem.name + suffix)
                if target.exists():
                    try:
                        target.unlink()
                        self.logger.debug(f"Removed old checkpoint sidecar: {target}")
                    except OSError as e:
                        self.logger.warning(f"Could not remove {target}: {e}")
            # Best-effort: also remove the legacy .pt if it happens to
            # exist (older runs that wrote the legacy format here).
            if oldest_pt.exists():
                try:
                    oldest_pt.unlink()
                    self.logger.debug(f"Removed old checkpoint: {oldest_pt}")
                except OSError as e:
                    self.logger.warning(f"Could not remove {oldest_pt}: {e}")

    # ---- load side --------------------------------------------------------

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        device: torch.device,
    ) -> tuple[int, float]:
        if not self.config.resume_from_checkpoint:
            return 0, float("inf")

        ckp_path = Path(self.config.resume_from_checkpoint)
        # Probe both layouts at this stem: the legacy .pt (if the
        # caller pointed there explicitly) AND the v2 split trio
        # (the v2 default). ``load_checkpoint_payload`` handles the
        # priority — legacy wins when the .pt exists, split otherwise.
        legacy_path, safetensors_path, meta_path, extra_state_path = _resolve_checkpoint_paths(ckp_path)
        if not (legacy_path.exists() or safetensors_path.exists() or meta_path.exists() or extra_state_path.exists()):
            self.logger.warning(
                f"Checkpoint file not found: {ckp_path} (checked legacy + split layouts). Starting from scratch."
            )
            return 0, float("inf")

        try:
            payload = load_checkpoint_payload(ckp_path)
            if payload is None:
                self.logger.warning(
                    f"Checkpoint at {ckp_path} exists but no recognized layout "
                    "(neither split nor legacy). Starting from scratch."
                )
                return 0, float("inf")

            load_model_state_dict(model, payload["model_state"])
            if optimizer is not None and payload.get("optimizer_state") is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
            if scheduler is not None and payload.get("scheduler_state") is not None:
                scheduler.load_state_dict(payload["scheduler_state"])
            if scaler is not None and payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])
            start_epoch = payload["epoch"] + 1
            best_loss = payload.get("best_loss", float("inf"))
            self.best_loss = best_loss
            self.loaded_extra_state = payload.get("extra_state")
            self.logger.info(
                f"✅ Resumed training from epoch {start_epoch} using checkpoint {ckp_path} "
                f"(format={payload.get('format_version', 'legacy')})"
            )
            return start_epoch, best_loss
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to load checkpoint from {ckp_path}: {e}")
            self.logger.warning("Starting from scratch due to checkpoint loading error.")
            return 0, float("inf")
