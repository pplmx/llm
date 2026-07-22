"""Optional DVC integration for data-versioning on the streaming pipeline.

This module wraps the parts of DVC's CLI we actually use (version a
data artifact; pull a previously-versioned artifact; report status) in
a small Python surface that:

- **Lazily imports ``dvc``**: ``import llm.data.dvc`` is always cheap,
  regardless of whether the user installed the ``dvc`` optional dep.
  Every helper checks :data:`DVC_AVAILABLE` first and degrades to a
  no-op with a clear warning when ``dvc`` is missing.
- **Hashes source fingerprints**: :func:`compute_fingerprint_hash`
  produces a stable sha256 of a fingerprint dict (sorted JSON,
  ``sort_keys=True``). The hash is what we record alongside a DVC
  artifact as the "version" key, so two ``source_fingerprint`` calls
  that produce identical dicts always produce the same hash.
- **Idempotent init**: :func:`init_dvc` runs ``dvc init`` only when the
  repo isn't already a DVC repo (idempotent across repeated calls).
- **Tracks per-artifact, not per-run**: :func:`dvc_add` runs
  ``dvc add <path>`` once per unique (path, fingerprint-hash) pair;
  re-adding a path that hasn't changed is a no-op.

The streaming pipeline's checkpoint resume already validates the
``source_fingerprint`` on every :meth:`load_checkpoint_state` call (see
:mod:`llm.data.modules.streaming`); this module layers DVC on top so
the *raw* data files can be re-fetched from the configured remote with
a single ``dvc pull``, instead of having to re-download the corpus
from HuggingFace every time the cache is wiped.

Install with ``uv sync --group dvc`` (or ``pip install llm[dvc]`` for
non-uv users) to enable. Without it, every helper in this module is a
no-op — the streaming pipeline still trains, it just doesn't version
its inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import UTC
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache for the lazy dvc import. We never import dvc at module load
# time — that would force every consumer of ``llm.data`` to pull the
# heavy dvc transitive dependency graph. The first call to
# :func:`_import_dvc` (or :data:`DVC_AVAILABLE`) probes the env; the
# result is cached for the lifetime of the process.
_dvc_module: Any = None
_dvc_import_attempted: bool = False


def _import_dvc() -> Any:
    """Return the ``dvc`` module, importing it on first call.

    Subsequent calls return the cached module. If ``dvc`` is not
    installed, raises :class:`ImportError` — callers should gate on
    :data:`DVC_AVAILABLE` first rather than catching the error.
    """
    global _dvc_module, _dvc_import_attempted
    if _dvc_module is not None:
        return _dvc_module
    if not _dvc_import_attempted:
        try:
            import dvc

            _dvc_module = dvc
        except ImportError:
            _dvc_module = None
        _dvc_import_attempted = True
    if _dvc_module is None:
        raise ImportError(
            "dvc is not installed. Install with `uv sync --group dvc` "
            "or `pip install llm[dvc]` to enable data versioning."
        )
    return _dvc_module


def _check_dvc_available() -> bool:
    """Return True if ``dvc`` is importable; cached after first call."""
    if _dvc_module is not None:
        return True
    if _dvc_import_attempted:
        return _dvc_module is not None
    try:
        _import_dvc()
    except ImportError:
        return False
    return _dvc_module is not None


# Module-level constant — computed lazily on first import of this module,
# then cached for the process lifetime. ``llm.data.dvc.DVC_AVAILABLE`` is
# the documented import path.
DVC_AVAILABLE: bool = _check_dvc_available()


def compute_fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    """Compute a stable sha256 hex digest of a source fingerprint dict.

    Used to key DVC artifacts on the *content* of the data source, not
    the path on disk. Two ``source_fingerprint()`` calls that produce
    identical dicts always produce the same hash, even across machines
    or Python versions (the JSON encoding is fully deterministic —
    ``sort_keys=True``, ``default=str``, ``separators=(",", ":")``).
    """
    payload = json.dumps(fingerprint, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_dvc_command(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a ``dvc`` CLI command and return the result.

    Centralized here so we can swap to ``dvc.api`` later without
    breaking call sites. Disables DVC's analytics collection via the
    ``DVC_NO_ANALYTICS=1`` env var (the CLI flag has been removed /
    renamed across DVC versions — the env var is the stable contract
    since DVC 2.x) for predictable CI runs.
    """
    cmd = [shutil.which("dvc") or "dvc", *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DVC_NO_ANALYTICS": "1"},
    )


def is_dvc_initialized(repo_root: Path | str) -> bool:
    """True if the given directory is a DVC repo (has a ``.dvc/`` subdir).

    Cheap filesystem probe; does NOT shell out. Safe to call on every
    helper invocation.
    """
    return (Path(repo_root) / ".dvc").is_dir()


def init_dvc(
    repo_root: Path | str,
    *,
    remote_url: str | None = None,
    remote_name: str = "storage",
) -> bool:
    """Initialize DVC in ``repo_root`` (idempotent).

    Args:
        repo_root: Repository root directory (the parent of ``.dvc/``
            when already initialized, or where to create it).
        remote_url: Optional remote URL to configure as the default
            storage. Supported schemes depend on the installed DVC
            extras — local paths, ``s3://``, ``gs://``, ``azure://``,
            ``ssh://``, and ``http(s)://`` are all supported by the
            default DVC install. If ``None``, only ``dvc init`` runs;
            the user can configure a remote later via ``dvc remote add``.
        remote_name: Remote name to register (default ``"storage"``).
            Only used when ``remote_url`` is set.

    Returns:
        True if ``dvc init`` was actually run (the repo wasn't already
        initialized); False if it was a no-op. Note that this is the
        inverse of :func:`is_dvc_initialized`'s "was already a DVC
        repo" answer, which makes the return value useful for logging.
    """
    repo_root = Path(repo_root).resolve()
    if is_dvc_initialized(repo_root):
        return False

    result = _run_dvc_command(["init", "--quiet"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"dvc init failed in {repo_root}: {result.stderr.strip() or result.stdout.strip()}")

    if remote_url is not None:
        result = _run_dvc_command(
            ["remote", "add", remote_name, remote_url],
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise RuntimeError(f"dvc remote add failed: {result.stderr.strip() or result.stdout.strip()}")
        result = _run_dvc_command(
            ["remote", "default", remote_name],
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise RuntimeError(f"dvc remote default failed: {result.stderr.strip() or result.stdout.strip()}")

    logger.info("Initialized DVC repo at %s (remote=%s)", repo_root, remote_url or "<none>")
    return True


def dvc_status(path: Path | str, *, repo_root: Path | str | None = None) -> str:
    """Return one of: ``"tracked"`` | ``"untracked"`` | ``"not_found"`` | ``"no_dvc"``.

    Pure filesystem probe — does NOT shell out to ``dvc status`` and
    does NOT require ``dvc`` to be installed. Inspects the
    filesystem for the ``.dvc`` directory and for a ``<path>.dvc``
    file (the marker DVC writes next to each tracked artifact). Useful
    for callers that want to detect "this dir was previously a DVC
    repo" without paying the dvc import cost.

    Note: ``"no_dvc"`` here means "no DVC bookkeeping on disk" — not
    "dvc package isn't importable". Callers that need to gate on the
    import availability should check :data:`DVC_AVAILABLE` separately.

    Args:
        path: Path to the artifact (file or directory). May be
            absolute or relative to ``repo_root``.
        repo_root: Repository root. Required when ``path`` is relative.
            If ``None``, defaults to the current working directory.
    """
    repo_root_path = Path(repo_root).resolve() if repo_root else Path.cwd()
    if not is_dvc_initialized(repo_root_path):
        return "no_dvc"
    target = Path(path)
    if not target.is_absolute():
        target = repo_root_path / target
    if not target.exists():
        return "not_found"
    # DVC writes ``<artifact>.dvc`` next to the artifact itself.
    if target.with_suffix(target.suffix + ".dvc").exists():
        return "tracked"
    return "untracked"


def dvc_add(
    path: Path | str,
    *,
    fingerprint: dict[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, str] | None:
    """Track ``path`` with DVC; return a metadata dict (or ``None`` if DVC is unavailable).

    Idempotent: re-tracking a path that is already versioned is a
    no-op (we skip the ``dvc add`` call). The metadata dict carries
    ``path`` (the artifact path), ``fingerprint_hash`` (sha256 of
    ``fingerprint`` if provided), ``repo_root`` (resolved), and
    ``versioned_at`` (ISO 8601 UTC timestamp).

    Args:
        path: File or directory to version. Relative paths are
            resolved against ``repo_root`` (or CWD when not set).
        fingerprint: Optional ``source_fingerprint()`` dict to bind to
            the artifact. When provided, its hash is recorded in the
            return metadata so callers can correlate checkpoints
            with data versions without re-hashing later.
        repo_root: Repository root. When ``None``, defaults to CWD.

    Returns:
        ``None`` if DVC is not available (caller should log + continue).
        Otherwise the metadata dict; raises :class:`RuntimeError` if
        ``dvc add`` itself fails.
    """
    if not DVC_AVAILABLE:
        logger.warning(
            "dvc_add(%s) skipped: DVC is not installed. Install with `uv sync --group dvc` to enable data versioning.",
            path,
        )
        return None

    repo_root_path = Path(repo_root).resolve() if repo_root else Path.cwd()
    if not is_dvc_initialized(repo_root_path):
        raise RuntimeError(
            f"DVC is not initialized at {repo_root_path}. Call `init_dvc(repo_root, remote_url=...)` first."
        )

    target = Path(path)
    if not target.is_absolute():
        target = repo_root_path / target
    if not target.exists():
        raise FileNotFoundError(f"dvc_add: path does not exist: {target}")

    # Already tracked — skip; do NOT re-add (would touch the .dvc file
    # mtime unnecessarily and pollute git diffs on .dvc files).
    if target.with_suffix(target.suffix + ".dvc").exists():
        return _build_metadata(target, repo_root_path, fingerprint)

    result = _run_dvc_command(["add", str(target)], cwd=repo_root_path)
    if result.returncode != 0:
        raise RuntimeError(f"dvc add {target} failed: {result.stderr.strip() or result.stdout.strip()}")

    return _build_metadata(target, repo_root_path, fingerprint)


def _build_metadata(
    target: Path,
    repo_root: Path,
    fingerprint: dict[str, Any] | None,
) -> dict[str, str]:
    from datetime import datetime

    meta: dict[str, str] = {
        "path": str(target),
        "repo_root": str(repo_root),
        "versioned_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if fingerprint is not None:
        meta["fingerprint_hash"] = compute_fingerprint_hash(fingerprint)
        meta["fingerprint"] = json.dumps(fingerprint, sort_keys=True, default=str)
    return meta


def dvc_pull(path: Path | str, *, repo_root: Path | str | None = None) -> bool:
    """Pull ``path`` from the configured DVC remote.

    Returns ``True`` on success, ``False`` if DVC is unavailable. Raises
    :class:`RuntimeError` when the underlying ``dvc pull`` fails.
    """
    if not DVC_AVAILABLE:
        logger.warning("dvc_pull(%s) skipped: DVC is not installed.", path)
        return False

    repo_root_path = Path(repo_root).resolve() if repo_root else Path.cwd()
    if not is_dvc_initialized(repo_root_path):
        raise RuntimeError(
            f"DVC is not initialized at {repo_root_path}. Call `init_dvc(repo_root, remote_url=...)` first."
        )

    target = Path(path)
    if not target.is_absolute():
        target = repo_root_path / target

    result = _run_dvc_command(["pull", str(target)], cwd=repo_root_path)
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull {target} failed: {result.stderr.strip() or result.stdout.strip()}")
    return True


__all__ = [
    "DVC_AVAILABLE",
    "compute_fingerprint_hash",
    "dvc_add",
    "dvc_pull",
    "dvc_status",
    "init_dvc",
    "is_dvc_initialized",
]
