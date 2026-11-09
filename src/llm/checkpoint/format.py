"""The v2 checkpoint file format: read, write, and migration.

Layout (ADR-006):

- ``<name>.safetensors`` — model weights only. Zero-copy loadable, no pickle
  deserialization risk, aligned with the HuggingFace compat layer.
- ``<name>.meta.json`` — training metadata: epoch, loss, best_loss,
  model_config, format_version.
- ``<name>.extra_state.pt`` — optimizer / scheduler / scaler state plus the
  optional ``extra_state`` dict. These are arbitrary Python objects (Adam
  moment estimates, scheduler step counts, ``StreamDataState`` instances) that
  fit neither safetensors nor JSON, so they stay one ``torch.save`` blob,
  isolated from the weights.

The legacy single-file ``.pt`` layout is still read and can be migrated with
:func:`convert_legacy_checkpoint_to_split`.

This module is pure file I/O plus payload-shape validation. It names fields
such as ``optimizer_state`` because the format contains them, but it does not
interpret a training run. ``torch`` and ``safetensors`` are imported lazily so
that merely importing the package stays cheap and works without the optional
``[compat]`` extra.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import uuid
import warnings
from pathlib import Path
from typing import Any

import torch

from llm.checkpoint.errors import CheckpointIncompleteError, CheckpointMigrationError

logger = logging.getLogger(__name__)

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

#: Matches a periodic epoch snapshot file (e.g. ``epoch_12.safetensors``,
#: ``epoch_12.meta.json``, ``epoch_12.extra_state.pt`` or the legacy
#: ``epoch_12.pt``). Used by :func:`_cleanup_old_checkpoints` to discover
#: every epoch snapshot on disk across resumed runs.
_EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)\.(safetensors|meta\.json|extra_state\.pt|pt)$")


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
    # Model-defining flags. These must also match on resume —
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

    # fsync on a READ-WRITE handle: Windows `os.fsync` rejects read-only handles
    # with EBADF (POSIX tolerates them); "rb+" keeps fsync-before-rename durable.
    with path.open("rb+") as f, contextlib.suppress(OSError):
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
    longer leave an empty/partial file under the final name.
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

    ``save_id``/``epoch`` are stored in the file's
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
    can cross-check the three sidecars came from the SAME atomic save:
    a crash mid-save that leaves a fresh weights file with
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
    """Load the v2 three-file layout.

    A "complete" split checkpoint has all three sidecars. If any (but not
    all) exist the save was interrupted mid-write: raise
    :class:`CheckpointIncompleteError` rather than returning ``None``
    (which the caller would treat as "nothing here" and silently fall back
    to scratch). Only when NO sidecar exists does the caller
    legitimately fall through to the legacy layout.
    """
    safetensors_path = stem_dir.with_name(stem_dir.name + SAFETENSORS_SUFFIX)
    meta_path = stem_dir.with_name(stem_dir.name + META_SUFFIX)
    extra_state_path = stem_dir.with_name(stem_dir.name + EXTRA_STATE_SUFFIX)
    present = [p for p in (safetensors_path, meta_path, extra_state_path) if p.exists()]
    if len(present) == 0:
        return None
    if len(present) < 3:
        missing = [p.name for p in (safetensors_path, meta_path, extra_state_path) if not p.exists()]
        raise CheckpointIncompleteError(
            f"Checkpoint {stem_dir} is incomplete: present sidecars "
            f"{[p.name for p in present]}, missing {missing}. The save was "
            "interrupted mid-write; point resume_from_checkpoint at a previous "
            "intact epoch checkpoint (or run llm migrate-ckpt on the legacy "
            ".pt if one exists at this stem)."
        )

    from safetensors.torch import load_file, safe_open

    model_state = load_file(str(safetensors_path))
    # Re-wrap into a torch state_dict (still tensors, just not safetensors-format).
    model_state = dict(model_state.items())
    # The weights-sidecar generation stamp: read the safetensors
    # header metadata written by ``_save_weights_safetensors``.
    weights_meta: dict[str, str] = {}
    try:
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            weights_meta = dict(f.metadata() or {})
    except OSError, ValueError, RuntimeError:
        weights_meta = {}

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        # A torn or hand-edited meta.json currently escapes as
        # ``JSONDecodeError: Expecting property name ... line 1 column 2``,
        # which names the parser, not the problem. The sibling sidecar loader
        # (``_torch_load_checkpoint_sidecar``) already normalizes its parse
        # failures; do the same so every corrupt-sidecar path says what to do.
        raise ValueError(
            f"corrupt checkpoint sidecar {meta_path}: not valid JSON ({exc}). "
            "The save was interrupted or the file was edited; restore all three "
            "sidecars from the same save (or a previous checkpoint)."
        ) from exc
    if meta.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        logger.warning(
            "Checkpoint meta.json format_version=%s does not match expected %s; "
            "loading anyway but check for compatibility.",
            meta.get("format_version"),
            CHECKPOINT_FORMAT_VERSION,
        )

    # ``weights_only=True``: this sidecar is written by
    # CheckpointManager and contains only optimizer/scheduler/scaler state
    # (tensors + primitives) plus the ``save_id``/``epoch`` stamps — no
    # arbitrary user objects. Loading with arbitrary-pickle semantics let a
    # user-supplied/untrusted checkpoint execute ``__reduce__`` code on
    # resume. Parse failures are normalized so a corrupt
    # sidecar degrades to the fail-loud path instead of crashing.
    extra = _torch_load_checkpoint_sidecar(extra_state_path)

    # Cross-check the trio came from ONE atomic save.
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
    tracebacks.
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
        "This format is deprecated; run `llm migrate-ckpt <path>` to convert "
        "to the v2 split layout (model.safetensors + meta.json + extra_state.pt).",
        DeprecationWarning,
        stacklevel=3,
    )
    # ``weights_only=True``: legacy checkpoints are a plain
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
    legacy_path, safetensors_path, meta_path, extra_state_path = _resolve_checkpoint_paths(path)
    stem = legacy_path.with_suffix("")

    # The split layout is authoritative whenever ANY split sidecar exists.
    # ``CheckpointManager.save_checkpoint`` writes only the split trio, and
    # ``llm migrate-ckpt`` keeps the legacy ``.pt`` by default
    # (``in_place=False``) — so a coexisting legacy file is a FROZEN
    # pre-migration snapshot that would silently stale-resume every
    # post-migration epoch. Legacy is the fallback only when
    # no split sidecar is present at all.
    any_split = any(p.exists() for p in (safetensors_path, meta_path, extra_state_path))
    if any_split or not legacy_path.exists():
        split = _load_split_checkpoint(stem)
        if split is not None:
            return split
        if any_split:
            # Partial trio already raised CheckpointIncompleteError; an
            # all-present-but-None split is impossible.
            raise CheckpointIncompleteError(f"Checkpoint at {stem} has split sidecars but could not be loaded.")
        # No split sidecar and no legacy file: nothing at this stem.
        return None

    # Only reachable when the legacy .pt exists AND no split sidecar does.
    return _load_legacy_checkpoint(legacy_path)


# ---------------------------------------------------------------------------
# Conversion: legacy v0.0.5 single-file .pt -> v2 split trio
# ---------------------------------------------------------------------------


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
    if not _safetensors_available():
        # The new layout's weights sidecar is safetensors; converting without
        # the package raises a raw ImportError from deep inside
        # ``_save_weights_safetensors``. Gate up front with the documented
        # error so the CLI stays on its one-line exit-1 contract.
        raise CheckpointMigrationError(
            f"Converting {legacy_path} requires the 'safetensors' package. "
            "Install with `uv sync --extra compat` or `pip install llm[compat]`."
        )

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

    # Load the legacy blob. ``weights_only=True``: the migration
    # path runs on user-supplied files, and a framework legacy dict is a plain
    # tensors+primitives payload — arbitrary-pickle loading here would be an
    # RCE on `llm migrate-ckpt attacker.pt`.
    try:
        payload = torch.load(legacy_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, OSError) as exc:
        # A corrupt file (garbage bytes, truncated zip archive) or a bare
        # ``torch.save(model)`` blob raises UnpicklingError/RuntimeError here —
        # BEFORE the dict validation below. Map it to CheckpointMigrationError so
        # the CLI surfaces the documented clean one-line exit-1 error instead of
        # a raw traceback.
        raise CheckpointMigrationError(f"{legacy_path} could not be loaded as a torch checkpoint: {exc}") from exc

    # Validate the payload is a v0.0.5 training dict BEFORE indexing it. A
    # `.pt` file that loads but is not a training checkpoint — e.g. a bare
    # ``torch.save(model)`` blob from ``llm quantize``, or any module /
    # non-dict pickle — raises TypeError/KeyError on ``payload["model_state"]``
    # below. The CLI only catches CheckpointMigrationError, so the raw
    # exception would have escaped as a traceback instead of the documented
    # clean one-line error.
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointMigrationError(
            f"{legacy_path} does not appear to be a v0.0.5 training checkpoint "
            "(missing 'model_state'). Only training checkpoints produced by "
            "CheckpointManager can be migrated."
        )

    # Write the split trio. Stamp a shared save_id into the weights file AND
    # meta + extra_state so the loader can cross-check the trio came from one
    # write (the weights stamp closes the meta↔extra-only window
    # of for freshly-migrated checkpoints too).
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


def _discover_epoch_snapshots(checkpoint_dir: Path) -> dict[int, Path]:
    """Map every ``epoch_<N>`` snapshot on disk to one of its files.

    Retention must span resumes: the in-memory list of a fresh (resumed)
    process is empty, so only a directory scan can see prior runs' periodic
    snapshots. Returns ``{epoch_number: any sidecar path}``
    so callers can stat mtime and reconstruct the full trio's stem.
    """
    found: dict[int, Path] = {}
    if not checkpoint_dir.exists():
        return found
    for path in checkpoint_dir.iterdir():
        m = _EPOCH_FILE_RE.match(path.name)
        if m is not None:
            epoch_no = int(m.group(1))
            found.setdefault(epoch_no, path)
    return found
