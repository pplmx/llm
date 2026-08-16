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
import pickle
import uuid
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
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


#: Model-config fields that MUST be identical between the checkpoint and the
#: current run for a resume to be correct — they define the tensor shapes and
#: the KV-cache / context geometry. Anything else (dropout, mroe tuning knobs
#: that don't change state_dict shapes) does not invalidate a resume.
_CONFIG_COMPAT_FIELDS: tuple[str, ...] = (
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "intermediate_size",
    "num_layers",
    "vocab_size",
    "max_seq_len",
    "use_glu",
    "num_experts",
    "top_k",
    "attn_impl",
    "mlp_impl",
    "norm_impl",
    "use_kv_cache",
    # Model-defining flags (RIL ISS-129). These must also match on resume —
    # they change tensor shapes (bias/no-bias) and the position-encoding
    # behavior, so a mismatch would silently load weights into the wrong
    # architecture.
    "pos_encoding_learned",
    "mlp_activation",
    "norm_first",
    "qkv_bias",
    "mlp_bias",
    "lm_head_bias",
    "use_rope",
    "rope_theta",
)


def _model_config_mismatches(expected: dict, actual: dict) -> dict[str, tuple[Any, Any]]:
    """Return ``{field: (ckpt_value, current_value)}`` for every
    architecture-defining config field where the two sidecars disagree.

    Compares only :data:`_CONFIG_COMPAT_FIELDS` — the fields a resume's
    correctness depends on. Non-architectural drift (dropout, logging etc.)
    is intentionally ignored so users can change those between runs.
    """
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _CONFIG_COMPAT_FIELDS:
        if field not in expected or field not in actual:
            continue
        if expected[field] != actual[field]:
            mismatches[field] = (actual[field], expected[field])
    return mismatches


def _fsync_file(path: Path) -> None:
    """Force ``path``'s bytes to durable storage (POSIX ``fsync``).

    Without this, ``os.replace`` only makes the rename atomic in the page
    cache; a power loss shortly after a save could leave the new file name
    pointing at an un-persisted file (or, worse, a partially-flushed rename).
    The directory entry itself must be fsynced too — a rename is a directory
    metadata operation. Best-effort: some filesystems/CI sandboxes reject
    ``fsync`` on the directory or lack ``O_DIRECT`` support.
    """
    import contextlib
    import os

    with path.open("rb") as f, contextlib.suppress(OSError):
        os.fsync(f.fileno())
    with contextlib.suppress(OSError):
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write ``payload`` to ``target`` atomically via temp + fsync + rename.

    Fsync is performed BEFORE the rename so that when the rename lands, the
    file contents are already durable — a crash after the rename can no
    longer leave an empty/partial file under the final name (RIL ISS-127).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    _fsync_file(tmp)
    tmp.replace(target)


def _save_weights_safetensors(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    *,
    save_id: str | None = None,
    epoch: int | None = None,
) -> None:
    """Save ``state_dict`` to ``path`` as safetensors (contiguous + clone).

    ``save_id``/``epoch`` (RIL ISS-127 + ISS-166) are stored in the file's
    metadata header so the loader can prove the WEIGHTS sidecar came from the
    same atomic save as the meta.json and extra_state.pt sidecars. Previously
    the generation marker lived only in meta/extra, so a crash between the
    weights write and the meta write left fresh weights beside a stale but
    mutually-consistent meta/extra pair that the meta↔extra check accepted.

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
    metadata: dict[str, str] | None = None
    if save_id is not None:
        metadata = {"save_id": save_id}
        if epoch is not None:
            metadata["epoch"] = str(epoch)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(contiguous, str(tmp), metadata=metadata)
    _fsync_file(tmp)
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
    *,
    save_id: str,
    epoch: int,
) -> None:
    """Save training-state sidecars to ``path`` as ``torch.save``.

    Wraps all four sub-states in a single dict so the loader can
    tell them apart — the top-level keys mirror the legacy single-file
    schema, but only the training-state slots are present.

    ``save_id`` and ``epoch`` are stamped into the sidecar so the loader
    can cross-check the three sidecars came from the SAME atomic save
    (RIL ISS-127): a crash mid-save that leaves a fresh weights file with
    a stale meta/extra trio would otherwise resume silently with
    mismatched optimizer/scheduler state.
    """
    blob = {
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "extra_state": extra_state,
        "save_id": save_id,
        "epoch": epoch,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    _fsync_file(tmp)
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

    from safetensors.torch import load_file, safe_open

    model_state = load_file(str(safetensors_path))
    # Re-wrap into a torch state_dict (still tensors, just not safetensors-format).
    model_state = dict(model_state.items())
    # The weights-sidecar generation stamp (RIL ISS-166): read the safetensors
    # header metadata written by ``_save_weights_safetensors``.
    weights_meta: dict[str, str] = {}
    try:
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            weights_meta = dict(f.metadata() or {})
    except OSError, ValueError, RuntimeError:
        weights_meta = {}

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        logger.warning(
            "Checkpoint meta.json format_version=%s does not match expected %s; "
            "loading anyway but check for compatibility.",
            meta.get("format_version"),
            CHECKPOINT_FORMAT_VERSION,
        )

    # ``weights_only=True`` (RIL ISS-170): this sidecar is written by
    # CheckpointManager and contains only optimizer/scheduler/scaler state
    # (tensors + primitives) plus the ``save_id``/``epoch`` stamps — no
    # arbitrary user objects. Loading with arbitrary-pickle semantics let a
    # user-supplied/untrusted checkpoint execute ``__reduce__`` code on
    # resume. Parse failures are normalized (RIL ISS-172) so a corrupt
    # sidecar degrades to the fail-loud path instead of crashing.
    extra = _torch_load_checkpoint_sidecar(extra_state_path)

    # Cross-check the trio came from ONE atomic save (RIL ISS-127 + ISS-166).
    # Each save stamps a ``save_id`` and ``epoch`` into the weights file, the
    # meta.json, and the extra_state.pt; if a crash interrupted the save, one
    # sidecar can sit beside a stale pair from another save. Resuming that
    # would silently re-train from the old optimizer/epoch with new weights.
    # (Backward-compatible: checkpoints written before the stamps have no
    # ``save_id`` and are loaded as-is.)
    meta_save_id = meta.get("save_id")
    extra_save_id = extra.get("save_id")
    weights_save_id = weights_meta.get("save_id")
    if meta_save_id is not None and extra_save_id is not None and meta_save_id != extra_save_id:
        raise ValueError(
            f"Checkpoint {stem_dir} is inconsistent: meta.json save_id={meta_save_id} != "
            f"extra_state.pt save_id={extra_save_id}. The save was interrupted mid-write; "
            "restore all three sidecars from the same save (or a previous checkpoint)."
        )
    if weights_save_id is not None and meta_save_id is not None and weights_save_id != meta_save_id:
        raise ValueError(
            f"Checkpoint {stem_dir} is inconsistent: weights.safetensors save_id={weights_save_id} != "
            f"meta.json save_id={meta_save_id}. A fresh weights file sits beside a stale "
            "meta/extra pair from another save — restore all three sidecars from the same "
            "save (or a previous checkpoint)."
        )
    meta_epoch = meta.get("epoch")
    extra_epoch = extra.get("epoch")
    weights_epoch = weights_meta.get("epoch")
    if meta_epoch is not None and extra_epoch is not None and meta_epoch != extra_epoch:
        raise ValueError(
            f"Checkpoint {stem_dir} is inconsistent: meta.json epoch={meta_epoch} != "
            f"extra_state.pt epoch={extra_epoch}. The save was interrupted mid-write; "
            "restore all three sidecars from the same save (or a previous checkpoint)."
        )
    if weights_epoch is not None and meta_epoch is not None and str(weights_epoch) != str(meta_epoch):
        raise ValueError(
            f"Checkpoint {stem_dir} is inconsistent: weights.safetensors epoch={weights_epoch} != "
            f"meta.json epoch={meta_epoch}. The save was interrupted mid-write; "
            "restore all three sidecars from the same save (or a previous checkpoint)."
        )

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


def _torch_load_checkpoint_sidecar(path: Path) -> Any:
    """``torch.load`` a framework sidecar with ``weights_only=True``.

    Normalizes any parse failure into :class:`pickle.UnpicklingError` so the
    caller's fail-loud / degrade logic sees a stable error class. torch's
    weights-only Unpickler raises internal exceptions for a malformed stream
    (``IndexError: pop from empty list``, ``EOFError``, ``AttributeError``,
    ``ValueError``, ...) that are NOT in the standard pickle exceptions —
    without normalization those escaped the resume degrade path as raw
    tracebacks (RIL ISS-172).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (
        pickle.PickleError,
        EOFError,
        IndexError,
        AttributeError,
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        raise pickle.UnpicklingError(f"corrupt checkpoint sidecar {path}: {exc}") from exc


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
    # ``weights_only=True`` (RIL ISS-170): legacy checkpoints are a plain
    # ``torch.save`` of a dict (model_state tensors + model_config dict +
    # optimizer/scheduler/scaler primitives) — nothing that needs arbitrary
    # pickle. Loading with ``weights_only=False`` here would execute
    # ``__reduce__`` from an untrusted file at resume/serve time.
    return _torch_load_checkpoint_sidecar(legacy_path)


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

    # Load the legacy blob. ``weights_only=True`` (RIL ISS-170): the migration
    # path runs on user-supplied files, and a framework legacy dict is a plain
    # tensors+primitives payload — arbitrary-pickle loading here would be an
    # RCE on `llm-migrate-ckpt attacker.pt`.
    payload = torch.load(legacy_path, map_location="cpu", weights_only=True)

    # Validate the payload is a v0.0.5 training dict BEFORE indexing it. A
    # `.pt` file that loads but is not a training checkpoint — e.g. a bare
    # ``torch.save(model)`` blob from ``llm-quantize``, or any module /
    # non-dict pickle — raises TypeError/KeyError on ``payload["model_state"]``
    # below. The CLI only catches CheckpointMigrationError, so the raw
    # exception would have escaped as a traceback instead of the documented
    # clean one-line error (RIL ISS-162).
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointMigrationError(
            f"{legacy_path} does not appear to be a v0.0.5 training checkpoint "
            "(missing 'model_state'). Only training checkpoints produced by "
            "CheckpointManager can be migrated."
        )

    # Write the split trio. Stamp a shared save_id into the weights file AND
    # meta + extra_state so the loader can cross-check the trio came from one
    # write (RIL ISS-127; the weights stamp closes the meta↔extra-only window
    # of RIL ISS-166 for freshly-migrated checkpoints too).
    migrated_save_id = uuid.uuid4().hex
    _save_weights_safetensors(
        payload["model_state"],
        safetensors_path,
        save_id=migrated_save_id,
        epoch=payload.get("epoch", 0),
    )
    _save_metadata_json(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": payload.get("epoch", 0),
            "loss": payload.get("loss"),
            "best_loss": payload.get("best_loss", float("inf")),
            "model_config": payload.get("model_config"),
            "save_id": migrated_save_id,
        },
        meta_path,
    )
    _save_extra_state_pt(
        payload.get("optimizer_state"),
        payload.get("scheduler_state"),
        payload.get("scaler_state"),
        payload.get("extra_state"),
        extra_state_path,
        save_id=migrated_save_id,
        epoch=payload.get("epoch", 0),
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
                "Install with `uv sync --extra compat` or `pip install llm[compat]`."
            )
        base = Path(self.config.checkpoint_dir) / name
        weights_path = base.with_name(base.name + SAFETENSORS_SUFFIX)
        meta_path = base.with_name(base.name + META_SUFFIX)
        extra_state_path = base.with_name(base.name + EXTRA_STATE_SUFFIX)

        # Per-save generation marker (RIL ISS-127): written into BOTH the
        # meta and the extra_state sidecars so the loader can prove the trio
        # came from one atomic save. A crash between writing the weights and
        # the extra_state would otherwise leave meta/extra from a previous
        # save paired with fresh weights — silently re-training from a stale
        # optimizer/epoch. ``epoch`` is cross-checked the same way.
        save_id = uuid.uuid4().hex

        # Stamp the save_id into the weights sidecar FIRST (RIL ISS-166): the
        # three sidecars only prove they came from one atomic save if the
        # weights file also carries the generation marker — a crash between a
        # weights write and the meta/extra write would otherwise leave a fresh
        # weights file beside a stale but mutually-consistent meta/extra pair
        # that the meta↔extra check cannot detect.
        _save_weights_safetensors(model_state, weights_path, save_id=save_id, epoch=epoch)
        _save_metadata_json(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "model_config": model_config,
                "save_id": save_id,
            },
            meta_path,
        )
        _save_extra_state_pt(
            optimizer_state,
            scheduler_state,
            scaler_state,
            extra_state,
            extra_state_path,
            save_id=save_id,
            epoch=epoch,
        )
        return weights_path, meta_path, extra_state_path

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        scheduler: LRScheduler | None,
        scaler: torch.amp.GradScaler | None,
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
                optimizer_state=optimizer.state_dict() if optimizer is not None else None,
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
            optimizer_state=optimizer.state_dict() if optimizer is not None else None,
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
                optimizer_state=optimizer.state_dict() if optimizer is not None else None,
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
        optimizer: optim.Optimizer | None,
        scheduler: LRScheduler | None,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
        expected_model_config: dict | None = None,
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

            # Resume-config compatibility check (RIL ISS-126): a checkpoint
            # whose model_config differs from the current run's in any
            # architecture-defining field means the user changed config (or
            # pointed at the wrong checkpoint). Weight-shape mismatches are
            # already caught loudly by load_model_state_dict below; but
            # non-tensor differences the state dict CANNOT expose — a changed
            # scheduler_type/epochs/warmup whose state dict loads fine into a
            # differently-shaped scheduler, a changed tokenizer re-tokenizing
            # with new ids over old weights, a changed max_seq_len — would
            # silently corrupt or mis-anneal the resume. Compare the fields
            # that define the model/schedule and refuse loudly.
            if expected_model_config and payload.get("model_config"):
                mismatches = _model_config_mismatches(expected_model_config, payload["model_config"])
                if mismatches:
                    raise ValueError(
                        f"Checkpoint at {ckp_path} was saved with a different model/schedule "
                        f"configuration: {', '.join(f'{k} (ckpt {v[0]}, now {v[1]})' for k, v in mismatches.items())}. "
                        "Refusing to silently resume with mismatched weights — fix the config or "
                        "point resume_from_checkpoint at a compatible checkpoint."
                    )

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
        except (
            OSError,
            RuntimeError,
            KeyError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
            TypeError,
        ) as e:
            # A state_dict shape/key incompatibility means the current model
            # architecture does NOT match the checkpoint — the user changed
            # config (e.g. hidden_size) or pointed at the wrong checkpoint.
            # Silently falling back to scratch would train a full run that
            # looks like a resume but discards all prior state (RIL ISS-108);
            # fail loudly instead so the mismatch is surfaced immediately.
            #
            # EOFError / pickle.UnpicklingError / TypeError are the exact
            # classes a truncated or wrong-kind ``.pt`` file raises (corrupt
            # pickle stream, ``torch.load`` of a non-dict, ``[None]*n``
            # padding) — the designed degrade-to-scratch path (RIL ISS-172).
            # ``pickle`` is imported at module top for the weights_only fix.
            message = str(e)
            if isinstance(e, RuntimeError) and any(
                marker in message
                for marker in ("size mismatch", "unexpected key", "missing key", "too many dimensions")
            ):
                raise RuntimeError(
                    f"Checkpoint at {ckp_path} does not match the current model architecture "
                    f"(state_dict mismatch: {message.splitlines()[0]}). Refusing to silently "
                    "restart from scratch — fix the model config or point resume_from_checkpoint "
                    "at a compatible checkpoint."
                ) from e

            # Two data-integrity/refusal errors are NOT recoverable format
            # issues — they mean the user pointed at the wrong checkpoint or a
            # save was interrupted:
            #  - ISS-127: the three sidecars disagree about which save they
            #    belong to (meta vs extra_state save_id/epoch mismatch), so
            #    resuming would pair fresh weights with stale optimizer state.
            #  - ISS-126: the checkpoint's model_config differs from the
            #    current run's architecture-defining fields, so the weights
            #    were not trained for this model/schedule.
            # Both must propagate to the caller instead of silently
            # restarting from scratch.
            if isinstance(e, ValueError) and ("inconsistent" in message or "Refusing to silently resume" in message):
                raise

            self.logger.error(f"Failed to load checkpoint from {ckp_path}: {e}")
            self.logger.warning("Starting from scratch due to checkpoint loading error.")
            return 0, float("inf")
