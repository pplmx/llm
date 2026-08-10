"""Tests for the DedupTextSource wrapper and its DataConfig plumbing."""

from __future__ import annotations

import functools
import hashlib
import re
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


def _custom_norm(text, *, flag=False):
    """Module-level normalize used by the fingerprint-stability tests.

    ``flag`` genuinely changes the normalization so two ``partial``
    wrappers with different frozen kwargs produce distinct fingerprints.
    """
    return text.strip().upper() if flag else text.strip()


class _CallableNormalizer:
    """Module-level callable instance (no ``__name__``)."""

    def __call__(self, text: str) -> str:
        return text.strip()


def test_dedup_fingerprint_normalize_descriptor_is_stable(tmp_path):
    """The ``normalize`` descriptor must be deterministic across processes.

    Regression test: the old ``repr()`` fallback embedded the callable's
    heap address for non-named callables (e.g. ``functools.partial``), so
    ``source_fingerprint`` differed between processes for byte-identical
    pipelines. That silently broke the DVC version key and triggered
    spurious ``fingerprint mismatch`` failures on checkpoint resume.
    """
    path = _write(tmp_path, "data.txt", ["a"])

    named = DedupTextSource(LocalLineTextSource(path), normalize=_custom_norm)
    partial = DedupTextSource(
        LocalLineTextSource(path),
        normalize=functools.partial(_custom_norm, flag=True),
    )
    fp_named = named.source_fingerprint()
    fp_partial = partial.source_fingerprint()

    # Named callables keep their (stable) name.
    assert fp_named["normalize"] == "_custom_norm"

    # A partial now renders to a stable descriptor with no heap address,
    # where the old repr() fell back to something like
    # ``functools.partial(<function _custom_norm at 0x7f...>)``.
    assert fp_partial["normalize"] == "partial(_custom_norm)[flag=True]"
    assert re.search(r"0x[0-9a-f]{4,}", fp_partial["normalize"]) is None

    # Two byte-identical pipelines produce identical fingerprints.
    again = DedupTextSource(
        LocalLineTextSource(path),
        normalize=functools.partial(_custom_norm, flag=True),
    ).source_fingerprint()
    assert again == fp_partial

    # Genuine config drift is still detected: different frozen kwargs on
    # the partial yield a different fingerprint.
    drift = DedupTextSource(
        LocalLineTextSource(path),
        normalize=functools.partial(_custom_norm, flag=False),
    ).source_fingerprint()
    assert drift != fp_partial
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_source_fingerprint(fp_partial, drift)


def test_dedup_fingerprint_callable_instance_uses_type_name(tmp_path):
    """A ``__call__`` instance without a stable name falls back to its type
    name rather than ``repr()``, which would leak a heap address."""
    path = _write(tmp_path, "data.txt", ["a"])

    src = DedupTextSource(LocalLineTextSource(path), normalize=_CallableNormalizer())
    desc = src.source_fingerprint()["normalize"]
    assert desc == "_CallableNormalizer"
    assert re.search(r"0x[0-9a-f]{4,}", desc) is None


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


def test_dedup_persisted_state_makes_skip_resume_exact(tmp_path):
    """With seen_hashes_path + write_seen_hashes, a skip-based resume
    (the streaming cursor) does NOT re-process records that were already
    consumed and hashed in the previous run — unlike in-memory-only dedup,
    whose fresh seen-set re-yields them as duplicates."""
    path = _write(tmp_path, "data.txt", ["a", "b", "a", "c", "b", "d", "e", "f"])
    seen_path = tmp_path / "seen.txt"

    def make_source():
        return DedupTextSource(
            LocalLineTextSource(path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        )

    run1 = list(make_source().iter_texts())
    assert run1 == ["a", "b", "c", "d", "e", "f"]

    # Resume: fresh source (as after a checkpoint restore), skip what the
    # streaming cursor consumed (6 dedup-passing records), write to the
    # same seen-hashes file. Every record's hash was persisted in run 1,
    # so nothing may be re-yielded (an in-memory-only dedup would yield
    # ['e', 'f'] here, duplicating consumed data).
    resumed = list(make_source().iter_texts(skip=len(run1)))
    assert resumed == []


def test_dedup_opens_seen_hashes_file_once_per_pass(monkeypatch, tmp_path):
    """With write_seen_hashes=True, the seen-hashes file must be opened
    once per iteration pass (not once per surviving record). A web-scale
    corpus with millions of records must not pay one open() syscall per
    record; opening once per pass with per-write flush preserves both the
    dedup result and the persisted-hashes durability contract.
    """
    import io

    data_path = _write(tmp_path, "data.txt", ["a", "b", "a", "c", "b", "a", "d"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("", encoding="utf-8")

    append_opens: list[int] = []
    real_io_open = io.open

    def counting_open(file, mode="r", *args, **kwargs):
        if mode.startswith("a"):
            append_opens.append(mode)
        return real_io_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", counting_open)

    source = DedupTextSource(
        LocalLineTextSource(data_path),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )
    deduped = list(source.iter_texts())

    # Dedup behavior is unchanged: only unique records survive.
    assert deduped == ["a", "b", "c", "d"]
    # And the persisted hashes are complete and decodable.
    h_a = hashlib.sha256(b"a").hexdigest()
    h_b = hashlib.sha256(b"b").hexdigest()
    h_c = hashlib.sha256(b"c").hexdigest()
    h_d = hashlib.sha256(b"d").hexdigest()
    assert seen_path.read_text(encoding="utf-8").splitlines() == [h_a, h_b, h_c, h_d]

    # The append file was opened exactly once for the whole pass — not
    # once per surviving record. (write_text above is not an append, and
    # the io.open patch only counts append-mode opens.)
    assert append_opens == ["a"], f"expected a single append open, got {append_opens}"
