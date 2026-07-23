"""Tests for :mod:`llm.data.dvc` — optional DVC integration.

The integration degrades gracefully when ``dvc`` isn't installed (the
CI environment in this repo doesn't install the ``dvc`` extra). Tests
are split into two halves:

1. **DVC-not-installed path** — :data:`DVC_AVAILABLE` is False; every
   helper is a no-op or returns a sentinel value; nothing raises.
2. **DVC-installed path** — gated behind ``pytest.importorskip("dvc")``
   so the tests skip cleanly when the package is missing.

The fingerprint-hash tests run unconditionally because they don't
need DVC; they're the foundation everything else builds on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.data import dvc as dvc_mod
from llm.data.dvc import (
    DVC_AVAILABLE,
    compute_fingerprint_hash,
    dvc_status,
    is_dvc_initialized,
)

# Module-level skip — but only on the *whole module* if dvc isn't even
# available as an import. We use a class-level guard instead for the
# installed-path tests, so the rest of the file runs unconditionally.
# Tests that need the actual ``dvc`` CLI / lib should call
# ``pytest.importorskip("dvc")`` inside the test (or be in classes
# decorated with ``@pytest.mark.skipif(not DVC_AVAILABLE, ...)``).


# ---------------------------------------------------------------------------
# Fingerprint hashing — foundation; runs without dvc installed
# ---------------------------------------------------------------------------


class TestComputeFingerprintHash:
    """``compute_fingerprint_hash`` produces a stable sha256 of the dict."""

    def test_returns_64_char_hex(self):
        h = compute_fingerprint_hash({"type": "local"})
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_dict_same_hash(self):
        a = compute_fingerprint_hash({"type": "local", "dataset_path": "/data/x.txt"})
        b = compute_fingerprint_hash({"type": "local", "dataset_path": "/data/x.txt"})
        assert a == b

    def test_different_dict_different_hash(self):
        a = compute_fingerprint_hash({"type": "local", "dataset_path": "/data/x.txt"})
        b = compute_fingerprint_hash({"type": "local", "dataset_path": "/data/y.txt"})
        assert a != b

    def test_key_order_irrelevant(self):
        # json.dumps(sort_keys=True) makes ordering irrelevant — same
        # dict with keys in different orders must hash the same.
        a = compute_fingerprint_hash({"type": "local", "path": "/x"})
        b = compute_fingerprint_hash({"path": "/x", "type": "local"})
        assert a == b

    def test_handles_nested_structures(self):
        a = compute_fingerprint_hash({"type": "dedup", "inner": {"type": "local", "path": "/x"}, "algo": "sha256"})
        b = compute_fingerprint_hash({"type": "dedup", "inner": {"type": "local", "path": "/x"}, "algo": "sha256"})
        assert a == b
        c = compute_fingerprint_hash({"type": "dedup", "inner": {"type": "local", "path": "/x"}, "algo": "blake2"})
        assert a != c

    def test_handles_non_json_native_via_default_str(self):
        # Path objects aren't JSON-serializable; ``default=str`` in the
        # encoder means they round-trip through ``str(path)`` and produce
        # a stable hash. Verifies the helper is friendly to source
        # fingerprints that contain ``pathlib.Path`` instances.
        a = compute_fingerprint_hash({"type": "local", "dataset_path": Path("/data/x.txt")})
        b = compute_fingerprint_hash({"type": "local", "dataset_path": "/data/x.txt"})
        # The Path instance is coerced to "/data/x.txt" via str(), so
        # the hash equals the plain-string variant.
        assert a == b

    def test_empty_dict(self):
        h = compute_fingerprint_hash({})
        # sha256 of "{}" — the empty-dict sentinel for "no fingerprint".
        assert isinstance(h, str)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# DVC-not-installed path
# ---------------------------------------------------------------------------


class TestDvcNotInstalledPath:
    """When ``dvc`` isn't importable, the writing helpers degrade gracefully."""

    def test_dvc_available_flag_is_bool(self):
        # True iff dvc is importable; regardless of value, the constant
        # must be a bool (so callers can write ``if DVC_AVAILABLE:``).
        assert isinstance(DVC_AVAILABLE, bool)

    def test_dvc_add_returns_none_when_dvc_unavailable(self, tmp_path: Path):
        if DVC_AVAILABLE:
            pytest.skip("DVC is installed; covered by the installed-path tests below")
        target = tmp_path / "data.txt"
        target.write_text("hello")
        result = dvc_mod.dvc_add(target, repo_root=tmp_path)
        assert result is None

    def test_dvc_pull_returns_false_when_dvc_unavailable(self, tmp_path: Path):
        if DVC_AVAILABLE:
            pytest.skip("DVC is installed; covered by the installed-path tests below")
        target = tmp_path / "data.txt"
        target.write_text("hello")
        assert dvc_mod.dvc_pull(target, repo_root=tmp_path) is False


# ---------------------------------------------------------------------------
# Filesystem-only helpers — work regardless of DVC availability
# ---------------------------------------------------------------------------


class TestIsDvcInitialized:
    """Pure filesystem probe; no shell-out."""

    def test_false_on_plain_directory(self, tmp_path: Path):
        assert is_dvc_initialized(tmp_path) is False

    def test_true_when_dot_dvc_subdir_exists(self, tmp_path: Path):
        (tmp_path / ".dvc").mkdir()
        assert is_dvc_initialized(tmp_path) is True

    def test_accepts_string_path(self, tmp_path: Path):
        assert is_dvc_initialized(str(tmp_path)) is False


class TestDvcStatus:
    """Cheap status probe; no shell-out."""

    def test_not_found_when_path_missing(self, tmp_path: Path):
        # ``tmp_path`` is plain (no .dvc subdir), so the function must
        # return ``"no_dvc"`` for any path within it.
        assert dvc_status(tmp_path / "does_not_exist.txt", repo_root=tmp_path) == "no_dvc"

    def test_untracked_when_dvc_initialized_but_no_dvc_file(self, tmp_path: Path):
        (tmp_path / ".dvc").mkdir()
        (tmp_path / "data.txt").write_text("hello")
        assert dvc_status(tmp_path / "data.txt", repo_root=tmp_path) == "untracked"

    def test_tracked_when_dvc_file_present(self, tmp_path: Path):
        (tmp_path / ".dvc").mkdir()
        (tmp_path / "data.txt").write_text("hello")
        (tmp_path / "data.txt.dvc").write_text("md5: stub\n")
        assert dvc_status(tmp_path / "data.txt", repo_root=tmp_path) == "tracked"

    def test_accepts_string_path(self, tmp_path: Path):
        (tmp_path / ".dvc").mkdir()
        (tmp_path / "data.txt").write_text("hello")
        assert dvc_status(str(tmp_path / "data.txt"), repo_root=str(tmp_path)) == "untracked"


# ---------------------------------------------------------------------------
# DVC-installed path — gated via class-level skipif so the suite still
# passes on a host without the ``dvc`` optional dep installed.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_dvc_repo(tmp_path: Path) -> Path:
    """Initialize a DVC repo in ``tmp_path``; skip on failure.

    Some sandboxes / CI hosts may not have the ``git`` and ``dvc``
    executables on PATH; in that case we skip rather than fail the
    whole test session.
    """
    import shutil
    import subprocess

    pytest.importorskip("dvc")
    if shutil.which("git") is None:
        pytest.skip("git is required for dvc init but is not on PATH")
    if shutil.which("dvc") is None:
        pytest.skip("dvc CLI is required for dvc init but is not on PATH")

    # ``dvc init`` requires a git repo first. Make a minimal one.
    git_exe = shutil.which("git")
    subprocess.run([git_exe, "init", "-q"], cwd=tmp_path, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603 — git_exe from shutil.which, args hardcoded
        [git_exe, "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(  # noqa: S603 — git_exe from shutil.which, args hardcoded
        [git_exe, "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )

    dvc_mod.init_dvc(tmp_path)
    # Either it ran (True) or it was already initialized (False) — both
    # mean we have a working DVC repo to test against.
    assert is_dvc_initialized(tmp_path)
    return tmp_path


@pytest.mark.skipif(not DVC_AVAILABLE, reason="dvc optional dep not installed")
class TestInitDvc:
    def test_init_runs_once(self, fresh_dvc_repo: Path):
        # fresh_dvc_repo already initialized; second call must be a no-op.
        assert dvc_mod.init_dvc(fresh_dvc_repo) is False
        assert is_dvc_initialized(fresh_dvc_repo) is True

    def test_init_raises_when_dvc_cli_fails(self, tmp_path: Path):
        # No git repo here → ``dvc init`` fails. The helper must surface
        # the underlying error rather than silently swallowing it.
        import shutil

        if shutil.which("dvc") is None or shutil.which("git") is None:
            pytest.skip("dvc/git CLI not on PATH")
        with pytest.raises(RuntimeError, match="dvc init failed"):
            dvc_mod.init_dvc(tmp_path)


@pytest.mark.skipif(not DVC_AVAILABLE, reason="dvc optional dep not installed")
class TestDvcAdd:
    def test_add_tracks_file_and_returns_metadata(self, fresh_dvc_repo: Path):
        target = fresh_dvc_repo / "data.txt"
        target.write_text("hello\nworld\n")
        meta = dvc_mod.dvc_add(target, repo_root=fresh_dvc_repo)
        assert meta is not None
        assert meta["path"] == str(target)
        assert "fingerprint_hash" not in meta  # we didn't pass fingerprint
        # A ``<file>.dvc`` marker should now exist next to the artifact.
        assert (target.parent / (target.name + ".dvc")).exists()

    def test_add_with_fingerprint_records_hash(self, fresh_dvc_repo: Path):
        target = fresh_dvc_repo / "data.txt"
        target.write_text("hello")
        fingerprint = {"type": "local", "dataset_path": "/data/x.txt"}
        meta = dvc_mod.dvc_add(target, fingerprint=fingerprint, repo_root=fresh_dvc_repo)
        assert meta is not None
        assert meta["fingerprint_hash"] == compute_fingerprint_hash(fingerprint)
        assert json.loads(meta["fingerprint"]) == fingerprint

    def test_add_is_idempotent_on_already_tracked(self, fresh_dvc_repo: Path):
        target = fresh_dvc_repo / "data.txt"
        target.write_text("hello")
        first = dvc_mod.dvc_add(target, repo_root=fresh_dvc_repo)
        second = dvc_mod.dvc_add(target, repo_root=fresh_dvc_repo)
        assert first is not None
        assert second is not None
        # Same metadata either way; no error on re-add.
        assert first["path"] == second["path"]

    def test_add_raises_on_missing_file(self, fresh_dvc_repo: Path):
        with pytest.raises(FileNotFoundError):
            dvc_mod.dvc_add(fresh_dvc_repo / "nope.txt", repo_root=fresh_dvc_repo)


@pytest.mark.skipif(not DVC_AVAILABLE, reason="dvc optional dep not installed")
class TestDvcPull:
    def test_pull_is_noop_when_artifact_local_and_no_remote(self, fresh_dvc_repo: Path):
        # When the artifact is already local (DVC's ``.dvc`` cache has
        # it) and no remote is configured, ``dvc pull`` is a successful
        # no-op — the user doesn't see an error, they just don't get
        # network sync. This is the expected behavior in a single-host
        # setup where the local cache IS the source of truth.
        target = fresh_dvc_repo / "data.txt"
        target.write_text("hello")
        dvc_mod.dvc_add(target, repo_root=fresh_dvc_repo)
        # Should not raise.
        assert dvc_mod.dvc_pull(target, repo_root=fresh_dvc_repo) is True

    def test_pull_without_remote_fails_for_untracked_path(self, fresh_dvc_repo: Path):
        # Asking DVC to pull a path that has no ``.dvc`` marker should
        # fail — DVC doesn't know what to pull.
        target = fresh_dvc_repo / "never_tracked.txt"
        target.write_text("hello")
        with pytest.raises(RuntimeError, match="dvc pull"):
            dvc_mod.dvc_pull(target, repo_root=fresh_dvc_repo)
