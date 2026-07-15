"""Tests for the DedupTextSource wrapper and its DataConfig plumbing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llm.data.sources import (
    DedupTextSource,
    HFStreamTextSource,
    LocalLineTextSource,
    build_text_source,
    validate_source_fingerprint,
)
from llm.training.core.config import DataConfig


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Core dedup behavior ---------------------------------------------------


def test_dedup_drops_exact_duplicates(tmp_path):
    path = _write(tmp_path, "data.txt", ["a", "b", "a", "c", "b", "a"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts())

    assert deduped == ["a", "b", "c"]


def test_dedup_collapses_internal_whitespace(tmp_path):
    path = _write(tmp_path, "ws.txt", ["hello world", "hello   world", "hello\tworld"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts())

    # All three collapse to the same canonical form.
    assert deduped == ["hello world"]


def test_dedup_is_case_sensitive_by_default(tmp_path):
    path = _write(tmp_path, "case.txt", ["Apple", "apple", "APPLE"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts())

    # Case-only differences must NOT be conflated.
    assert deduped == ["Apple", "apple", "APPLE"]


def test_dedup_strips_leading_and_trailing_whitespace(tmp_path):
    path = _write(tmp_path, "strip.txt", ["  hello  ", "hello", "\thello\n"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts())

    assert deduped == ["hello"]


def test_dedup_skip_delegates_to_inner_source(tmp_path):
    path = _write(tmp_path, "skip.txt", ["a", "b", "c", "d"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts(skip=2))

    # skip semantics match non-dedup sources: input-record skip, so
    # 'c' and 'd' are the ones the inner source yields, and they are
    # unique → both come through.
    assert deduped == ["c", "d"]


def test_dedup_custom_normalize_function(tmp_path):
    path = _write(tmp_path, "lower.txt", ["Apple", "APPLE", "apple"])

    deduped = list(DedupTextSource(LocalLineTextSource(path), normalize=str.lower).iter_texts())

    # Custom normalize is honoured.
    assert deduped == ["Apple"]


def test_dedup_custom_hash_algo(tmp_path):
    path = _write(tmp_path, "algo.txt", ["alpha", "alpha", "beta"])

    deduped = list(DedupTextSource(LocalLineTextSource(path), hash_algo="sha1").iter_texts())

    assert deduped == ["alpha", "beta"]


def test_dedup_rejects_unknown_hash_algo(tmp_path):
    path = _write(tmp_path, "x.txt", ["a"])

    with pytest.raises(ValueError, match="unsupported"):
        DedupTextSource(LocalLineTextSource(path), hash_algo="not-a-real-hash")


# --- Cross-run persistence -------------------------------------------------


def test_dedup_loads_existing_seen_hashes(tmp_path):
    data_path = _write(tmp_path, "data.txt", ["a", "b", "c"])
    seen_path = tmp_path / "seen.txt"
    # Pre-load only the hash of 'a'.
    h_a = hashlib.sha256(b"a").hexdigest()
    seen_path.write_text(h_a + "\n", encoding="utf-8")

    deduped = list(DedupTextSource(LocalLineTextSource(data_path), seen_hashes_path=seen_path).iter_texts())

    assert deduped == ["b", "c"]


def test_dedup_loads_seen_hashes_with_blank_lines_and_trailing_ws(tmp_path):
    data_path = _write(tmp_path, "data.txt", ["a", "b"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("\n  \n  hash  \n\n", encoding="utf-8")

    # The malformed/blank lines are ignored; 'a' is still seen.
    deduped = list(DedupTextSource(LocalLineTextSource(data_path), seen_hashes_path=seen_path).iter_texts())

    assert deduped == ["a", "b"]


def test_dedup_write_seen_hashes_appends_new_hashes(tmp_path):
    data_path = _write(tmp_path, "data.txt", ["a", "b", "a"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("", encoding="utf-8")  # pre-create empty file

    deduped = list(
        DedupTextSource(
            LocalLineTextSource(data_path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        ).iter_texts()
    )

    assert deduped == ["a", "b"]

    h_a = hashlib.sha256(b"a").hexdigest()
    h_b = hashlib.sha256(b"b").hexdigest()
    written = seen_path.read_text(encoding="utf-8").splitlines()
    assert written == [h_a, h_b]


def test_dedup_write_seen_hashes_requires_path():
    inner = LocalLineTextSource(__file__)  # any existing local source
    with pytest.raises(ValueError, match="seen_hashes_path"):
        DedupTextSource(inner, write_seen_hashes=True)


def test_dedup_persistence_round_trip(tmp_path):
    """First run drops 'a'; second run with the persisted seen-file
    still drops 'a' without re-yielding it.
    """
    data_path = _write(tmp_path, "data.txt", ["a", "b", "a"])
    seen_path = tmp_path / "seen.txt"

    first_pass = list(
        DedupTextSource(
            LocalLineTextSource(data_path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        ).iter_texts()
    )
    assert first_pass == ["a", "b"]

    second_pass = list(
        DedupTextSource(
            LocalLineTextSource(data_path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        ).iter_texts()
    )
    # 'a' is already in seen.txt, so it is dropped in pass 2.
    assert second_pass == []


def test_dedup_missing_seen_file_is_silent_noop(tmp_path):
    data_path = _write(tmp_path, "data.txt", ["a"])
    seen_path = tmp_path / "never_created.txt"
    assert not seen_path.exists()

    deduped = list(DedupTextSource(LocalLineTextSource(data_path), seen_hashes_path=seen_path).iter_texts())

    assert deduped == ["a"]


# --- Fingerprint -----------------------------------------------------------


def test_dedup_fingerprint_includes_inner_and_dedup_config(tmp_path):
    path = _write(tmp_path, "data.txt", ["a"])
    seen_path = tmp_path / "seen.txt"

    source = DedupTextSource(
        LocalLineTextSource(path),
        seen_hashes_path=seen_path,
        hash_algo="sha1",
    )
    fp = source.source_fingerprint()

    assert fp["type"] == "dedup"
    assert fp["hash_algo"] == "sha1"
    assert fp["inner"] == {"type": "local", "dataset_path": str(path.resolve())}
    assert fp["seen_hashes_path"] == str(seen_path.resolve())


def test_dedup_fingerprint_omits_seen_hashes_path_when_unset(tmp_path):
    path = _write(tmp_path, "data.txt", ["a"])

    fp = DedupTextSource(LocalLineTextSource(path)).source_fingerprint()

    assert "seen_hashes_path" not in fp


def test_dedup_fingerprint_mismatch_raises(tmp_path):
    """Checkpoint-resume validation should catch dedup-config drift."""
    path = _write(tmp_path, "data.txt", ["a"])

    expected = DedupTextSource(LocalLineTextSource(path), hash_algo="sha256")
    actual = DedupTextSource(LocalLineTextSource(path), hash_algo="sha1")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_source_fingerprint(
            expected.source_fingerprint(),
            actual.source_fingerprint(),
        )


# --- DataConfig + registry plumbing ----------------------------------------


def test_data_config_accepts_dedup_data_sources():
    DataConfig(data_source="dedup_local", dataset_path="/tmp/data.txt")  # noqa: S108
    DataConfig(data_source="dedup_hf", dataset_name="foo/bar")


def test_data_config_rejects_unknown_data_source():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DataConfig(data_source="mystery")


def test_build_text_source_resolves_dedup_local(tmp_path):
    path = _write(tmp_path, "data.txt", ["a"])

    cfg = DataConfig(data_source="dedup_local", dataset_path=str(path))
    source = build_text_source(cfg)

    assert isinstance(source, DedupTextSource)
    assert isinstance(source.inner, LocalLineTextSource)


def test_build_text_source_resolves_dedup_hf():
    cfg = DataConfig(data_source="dedup_hf", dataset_name="foo/bar")
    source = build_text_source(cfg)

    assert isinstance(source, DedupTextSource)
    assert isinstance(source.inner, HFStreamTextSource)


def test_build_text_source_dedup_local_forwards_knobs(tmp_path):
    path = _write(tmp_path, "data.txt", ["a"])
    seen_path = tmp_path / "seen.txt"

    cfg = DataConfig(
        data_source="dedup_local",
        dataset_path=str(path),
        seen_hashes_path=str(seen_path),
        write_seen_hashes=True,
        hash_algo="sha1",
    )
    source = build_text_source(cfg)

    assert source.seen_hashes_path == seen_path
    assert source.write_seen_hashes is True
    assert source.hash_algo == "sha1"


def test_build_text_source_dedup_local_requires_dataset_path():
    cfg = DataConfig(data_source="dedup_local")
    with pytest.raises(ValueError, match="dataset_path"):
        build_text_source(cfg)


def test_build_text_source_dedup_hf_requires_dataset_name():
    cfg = DataConfig(data_source="dedup_hf")
    with pytest.raises(ValueError, match="dataset_name"):
        build_text_source(cfg)


def test_data_config_hash_algo_pattern_rejects_garbage():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DataConfig(data_source="dedup_local", hash_algo="SHA-256-bad!")


# --- Inner-source agnostic end-to-end --------------------------------------


def test_dedup_handles_inner_source_returning_no_records(tmp_path):
    empty = _write(tmp_path, "empty.txt", [])

    deduped = list(DedupTextSource(LocalLineTextSource(empty)).iter_texts())

    assert deduped == []
