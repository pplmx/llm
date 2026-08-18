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


def test_dedup_skip_counts_survivors_of_this_source(tmp_path):
    path = _write(tmp_path, "skip.txt", ["a", "b", "c", "d"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts(skip=2))

    # ``skip`` counts the records *of this source* (the deduplicated
    # survivors the caller observes), matching the ``TextSource.iter_texts``
    # contract ("skip the first ``skip`` records"). 'c' and 'd' come
    # through after skipping the 2 survivors 'a' and 'b'.
    assert deduped == ["c", "d"]


def test_dedup_inmemory_skip_counts_survivors_not_raw_records(tmp_path):
    """In-memory dedup (no seen-hashes file): ``skip=2`` skips the first 2
    *survivors* ('a','b'), NOT 2 raw records (which would land on 'b').
    The streaming resume cursor counts survivors, so the two must agree
    or a resumed pass re-processes the tail of the consumed window
    (RIL ISS-088)."""
    path = _write(tmp_path, "dup.txt", ["a", "a", "b", "c", "d"])

    deduped = list(DedupTextSource(LocalLineTextSource(path)).iter_texts(skip=2))

    assert deduped == ["c", "d"]


def test_dedup_inmemory_skip_rebuilds_seen_set_for_resume(tmp_path):
    """A mid-corpus resume past 2 consumed survivors ('x','a' — raw records
    x,a,b) must not re-yield the window tail and must still drop a window
    duplicate ('a') beyond the resume point. The resumed pass re-hashes the
    skip window so its seen-set matches the pre-resume session exactly."""
    path = _write(tmp_path, "res.txt", ["x", "a", "b", "a", "c"])

    resumed = list(DedupTextSource(LocalLineTextSource(path)).iter_texts(skip=2))

    # Survivors of the full stream: x,a,b,c. Skip x,a → [b,c]. The raw 'a'
    # after 'b' is a duplicate of the skipped survivor 'a' and must be kept
    # out via the rebuilt seen-set (old raw-skip behavior yielded ['b','a','c']).
    assert resumed == ["b", "c"]


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


def test_dedup_nonzero_rank_does_not_bloat_seen_file(tmp_path):
    """Only rank 0 may append to the shared seen-hashes file: every DDP rank
    walks the whole corpus, so without the gate each hash is written once per
    rank and the file grows up to world_size times its size (round-79 TASK-194 / ISS-233)."""
    from unittest.mock import patch

    data_path = _write(tmp_path, "data.txt", ["a", "b", "a"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("", encoding="utf-8")

    def build():
        return DedupTextSource(
            LocalLineTextSource(data_path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        )

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        deduped = list(build().iter_texts())
    assert deduped == ["a", "b"]
    assert seen_path.read_text(encoding="utf-8") == ""  # non-zero rank: read-only

    # Rank 0 still persists every record.
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        deduped = list(build().iter_texts())
    assert deduped == ["a", "b"]
    written = seen_path.read_text(encoding="utf-8").splitlines()
    assert written == [hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()]


def test_dedup_single_process_still_persists(tmp_path):
    """No process group -> the single process writes as before."""
    from unittest.mock import patch

    data_path = _write(tmp_path, "data.txt", ["a", "b"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("", encoding="utf-8")

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        list(
            DedupTextSource(
                LocalLineTextSource(data_path),
                seen_hashes_path=seen_path,
                write_seen_hashes=True,
            ).iter_texts()
        )
    assert len(seen_path.read_text(encoding="utf-8").splitlines()) == 2


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


def test_dedup_reset_cross_run_seen_recycles_corpus(tmp_path):
    """Regression (RIL ISS-064): once a corpus has been fully consumed and
    hashed in an earlier run, a *fresh* source yields nothing (cross-run
    dedup). The streaming engine then calls :meth:`reset_cross_run_seen` on
    corpus-cycle reset; the next pass must re-yield the corpus (scoped to
    in-memory per-pass dedup) instead of the engine raising "streaming
    corpus is empty".

    Without the fix the persisted cross-run seen-set made the FIRST pass of
    a recycled corpus classify every record as already-seen -> 0 texts.
    """
    data_path = _write(tmp_path, "data.txt", ["a", "b", "c", "d", "e"])
    seen_path = tmp_path / "seen.txt"

    # Run 1: consume + persist every record.
    run1 = list(
        DedupTextSource(
            LocalLineTextSource(data_path),
            seen_hashes_path=seen_path,
            write_seen_hashes=True,
        ).iter_texts()
    )
    assert run1 == ["a", "b", "c", "d", "e"]

    # Fresh source (as after a new run): cross-run dedup drops everything.
    fresh = DedupTextSource(
        LocalLineTextSource(data_path),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )
    assert list(fresh.iter_texts()) == []

    # Engine recycle: reset_cross_run_seen restores consumability.
    fresh.reset_cross_run_seen()
    recycled = list(fresh.iter_texts())
    assert recycled == ["a", "b", "c", "d", "e"]


def test_dedup_reset_cross_run_seen_still_removes_in_pass_duplicates(tmp_path):
    """reset_cross_run_seen clears only the *cross-run* baseline; in-corpus
    duplicates are still removed per pass (per-pass dedup semantics from
    RIL ISS-038 remain intact)."""
    data_path = _write(tmp_path, "data.txt", ["a", "a", "b", "b", "c"])
    seen_path = tmp_path / "seen.txt"
    src = DedupTextSource(
        LocalLineTextSource(data_path),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )
    list(src.iter_texts())  # consume once, persist a+b+c
    src.reset_cross_run_seen()

    assert list(src.iter_texts()) == ["a", "b", "c"]


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


def test_dedup_reiterates_cleanly_on_second_pass(tmp_path):
    """A corpus cycle re-iterates the same source object. The seen-set
    must be scoped to a single pass: pass 2 re-yields the full corpus
    (minus genuine in-pass duplicates) instead of classifying every
    record as already-seen and yielding nothing (RIL ISS-038).
    """
    path = _write(tmp_path, "data.txt", ["a", "b", "a", "c", "b"])

    source = DedupTextSource(LocalLineTextSource(path))

    pass1 = list(source.iter_texts())
    pass2 = list(source.iter_texts())

    assert pass1 == ["a", "b", "c"]
    assert pass2 == ["a", "b", "c"]


def test_dedup_cycle_with_persistence_rewrites_no_duplicate_hashes(tmp_path):
    """With seen_hashes_path + write_seen_hashes, a corpus cycle re-yields
    the records (cross-run dedup baseline only) but the file must not grow
    duplicate hash lines across passes within the same session."""
    path = _write(tmp_path, "data.txt", ["a", "b", "a", "c"])
    seen_path = tmp_path / "seen.txt"
    seen_path.write_text("", encoding="utf-8")

    source = DedupTextSource(
        LocalLineTextSource(path),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )

    pass1 = list(source.iter_texts())
    pass2 = list(source.iter_texts())

    assert pass1 == ["a", "b", "c"]
    assert pass2 == ["a", "b", "c"]

    h_a = hashlib.sha256(b"a").hexdigest()
    h_b = hashlib.sha256(b"b").hexdigest()
    h_c = hashlib.sha256(b"c").hexdigest()
    written = seen_path.read_text(encoding="utf-8").splitlines()
    # Every hash appears exactly once, even though two full passes ran.
    assert written == [h_a, h_b, h_c]


def test_dedup_cycle_keeps_persisted_baseline_dropped(tmp_path):
    """Cross-run dedup still holds across a cycle: a record whose hash was
    persisted at construction stays dropped on every pass, while a record
    persisted mid-session is re-yielded on the next pass (it belongs to the
    next cycle, not to a previous run)."""
    data_path = _write(tmp_path, "data.txt", ["a", "b"])
    seen_path = tmp_path / "seen.txt"
    h_a = hashlib.sha256(b"a").hexdigest()
    seen_path.write_text(h_a + "\n", encoding="utf-8")

    source = DedupTextSource(
        LocalLineTextSource(data_path),
        seen_hashes_path=seen_path,
        write_seen_hashes=True,
    )

    assert list(source.iter_texts()) == ["b"]
    assert list(source.iter_texts()) == ["b"]

    # 'b' was appended once on the first pass, not duplicated on pass 2.
    assert seen_path.read_text(encoding="utf-8").splitlines() == [h_a, hashlib.sha256(b"b").hexdigest()]


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
