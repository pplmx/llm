"""Tests for ``llm-migrate-ckpt`` — convert legacy v0.0.5 ``.pt`` to v2 split layout.

The CLI is a thin wrapper around
:func:`llm.training.core.checkpoint.convert_legacy_checkpoint_to_split`,
plus an optional round-trip ``--verify`` check. These tests exercise
both halves:

1. **Conversion logic** — atomic writes, refusal to clobber existing
   sidecars, ``--in-place`` deletion, error paths (missing legacy,
   already-converted, ambiguous path).
2. **CLI surface** — Typer invocation via ``CliRunner`` covers
   ``--dry-run`` (no writes), ``--in-place`` (legacy deletion),
   ``--verify`` (round-trip + exit code 2 on mismatch), and the
   error exit code 1 paths.

The tests use a hand-written v0.0.5-era blob (single ``torch.save``
with the legacy key schema) instead of running a real training
job — that's enough to exercise every code path without bringing
in the full training engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from llm.cli.migrate_ckpt import app
from llm.training.core.checkpoint import (
    EXTRA_STATE_SUFFIX,
    META_SUFFIX,
    SAFETENSORS_SUFFIX,
    CheckpointMigrationError,
    convert_legacy_checkpoint_to_split,
    load_checkpoint_payload,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_checkpoint(tmp_path: Path) -> Path:
    """Write a v0.0.5-era single-file .pt blob at ``tmp_path/legacy.pt``.

    The blob mirrors the schema ``CheckpointManager.save_checkpoint``
    produced before commit ``4b9cf68``: a single dict with model +
    optimizer + scheduler + scaler + extra_state + epoch + loss +
    best_loss + model_config keys.
    """
    legacy = tmp_path / "legacy.pt"
    blob = {
        "epoch": 7,
        "loss": 0.42,
        "best_loss": 0.18,
        "model_state": {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bias": torch.zeros(4),
        },
        "model_config": {
            "vocab_size": 100,
            "hidden_size": 16,
            "num_layers": 1,
        },
        "optimizer_state": {"step": 100, "param_groups": [{"lr": 1e-3}]},
        "scheduler_state": None,
        "scaler_state": None,
        "extra_state": {"stream_data": {"0": {"line_index": 13}}},
    }
    torch.save(blob, legacy)
    return legacy


@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer's in-process CLI runner; no subprocess overhead."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Conversion logic (the public helper)
# ---------------------------------------------------------------------------


class TestConvertLegacyCheckpointToSplit:
    """The ``convert_legacy_checkpoint_to_split`` helper."""

    def test_writes_three_sidecars(self, legacy_checkpoint: Path, tmp_path: Path):
        result = convert_legacy_checkpoint_to_split(legacy_checkpoint)
        stem = legacy_checkpoint.with_suffix("")
        assert result["weights"] == stem.with_suffix(SAFETENSORS_SUFFIX)
        assert result["meta"] == Path(str(stem) + META_SUFFIX)
        assert result["extra_state"] == Path(str(stem) + EXTRA_STATE_SUFFIX)
        for p in result.values():
            assert p.exists()

    def test_meta_json_carries_metadata_and_format_version(
        self, legacy_checkpoint: Path
    ):
        result = convert_legacy_checkpoint_to_split(legacy_checkpoint)
        meta = json.loads(result["meta"].read_text())
        assert meta["format_version"] == "2.0"
        assert meta["epoch"] == 7
        assert meta["loss"] == 0.42
        assert meta["best_loss"] == 0.18
        assert meta["model_config"]["vocab_size"] == 100

    def test_extra_state_carries_optimizer_and_extra_state(
        self, legacy_checkpoint: Path
    ):
        result = convert_legacy_checkpoint_to_split(legacy_checkpoint)
        blob = torch.load(
            result["extra_state"], map_location="cpu", weights_only=False
        )
        assert blob["optimizer_state"]["step"] == 100
        assert blob["extra_state"]["stream_data"]["0"]["line_index"] == 13

    def test_model_state_round_trips_through_safetensors(
        self, legacy_checkpoint: Path
    ):
        result = convert_legacy_checkpoint_to_split(legacy_checkpoint)
        # The new layout is loadable via the public helper.
        payload = load_checkpoint_payload(
            result["weights"].with_suffix("")
        )
        assert payload is not None
        # Tensor values are byte-equal under the float32 default.
        assert torch.equal(
            payload["model_state"]["weight"],
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        )
        assert torch.equal(payload["model_state"]["bias"], torch.zeros(4))

    def test_in_place_deletes_legacy(self, legacy_checkpoint: Path):
        assert legacy_checkpoint.exists()
        convert_legacy_checkpoint_to_split(legacy_checkpoint, in_place=True)
        assert not legacy_checkpoint.exists()

    def test_default_keeps_legacy(self, legacy_checkpoint: Path):
        convert_legacy_checkpoint_to_split(legacy_checkpoint)
        assert legacy_checkpoint.exists()

    def test_accepts_stem_path(self, legacy_checkpoint: Path):
        stem = legacy_checkpoint.with_suffix("")
        result = convert_legacy_checkpoint_to_split(stem)
        assert all(p.exists() for p in result.values())

    def test_missing_legacy_raises(self, tmp_path: Path):
        with pytest.raises(CheckpointMigrationError, match="not found"):
            convert_legacy_checkpoint_to_split(tmp_path / "nope.pt")

    def test_refuses_when_split_already_exists(
        self, legacy_checkpoint: Path, tmp_path: Path
    ):
        # Pre-create a sidecar at the same stem.
        stem = legacy_checkpoint.with_suffix("")
        safetensors = stem.with_suffix(SAFETENSORS_SUFFIX)
        safetensors.write_bytes(b"existing")
        with pytest.raises(CheckpointMigrationError, match="already exists"):
            convert_legacy_checkpoint_to_split(legacy_checkpoint)
        # The pre-existing file must NOT be touched.
        assert safetensors.read_bytes() == b"existing"

    def test_overwrite_replaces_existing_sidecars(
        self, legacy_checkpoint: Path, tmp_path: Path
    ):
        stem = legacy_checkpoint.with_suffix("")
        safetensors = stem.with_suffix(SAFETENSORS_SUFFIX)
        safetensors.write_bytes(b"stale")
        result = convert_legacy_checkpoint_to_split(
            legacy_checkpoint, overwrite=True
        )
        # The stale contents are replaced with the new ones.
        assert safetensors.read_bytes() != b"stale"
        assert result["weights"] == safetensors

    def test_atomic_writes_via_temp_files(
        self, legacy_checkpoint: Path, tmp_path: Path
    ):
        # After a successful conversion, no .tmp files should remain.
        convert_legacy_checkpoint_to_split(legacy_checkpoint)
        stem = legacy_checkpoint.with_suffix("")
        for suffix in (SAFETENSORS_SUFFIX, META_SUFFIX, EXTRA_STATE_SUFFIX):
            target = Path(str(stem) + suffix)
            tmp = target.with_suffix(target.suffix + ".tmp")
            assert target.exists()
            assert not tmp.exists(), f"stale .tmp file remains: {tmp}"


# ---------------------------------------------------------------------------
# CLI surface (Typer CliRunner)
# ---------------------------------------------------------------------------


class TestMigrateCkptCli:
    """The ``llm-migrate-ckpt`` CLI surface."""

    def test_help_exits_zero(self, cli_runner: CliRunner):
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Convert" in result.stdout
        assert "--in-place" in result.stdout

    def test_dry_run_writes_nothing(self, cli_runner: CliRunner, legacy_checkpoint: Path):
        result = cli_runner.invoke(app, [str(legacy_checkpoint), "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run]" in result.stdout
        # The legacy file is preserved; nothing was written next to it.
        assert legacy_checkpoint.exists()
        stem = legacy_checkpoint.with_suffix("")
        assert not stem.with_suffix(SAFETENSORS_SUFFIX).exists()
        assert not Path(str(stem) + META_SUFFIX).exists()
        assert not Path(str(stem) + EXTRA_STATE_SUFFIX).exists()

    def test_happy_path_writes_sidecars(
        self, cli_runner: CliRunner, legacy_checkpoint: Path
    ):
        result = cli_runner.invoke(app, [str(legacy_checkpoint)])
        assert result.exit_code == 0, result.stderr
        assert "✓ Converted" in result.stdout
        stem = legacy_checkpoint.with_suffix("")
        assert stem.with_suffix(SAFETENSORS_SUFFIX).exists()
        assert Path(str(stem) + META_SUFFIX).exists()
        assert Path(str(stem) + EXTRA_STATE_SUFFIX).exists()
        # Legacy is kept by default.
        assert legacy_checkpoint.exists()

    def test_in_place_flag_deletes_legacy(
        self, cli_runner: CliRunner, legacy_checkpoint: Path
    ):
        result = cli_runner.invoke(app, [str(legacy_checkpoint), "--in-place"])
        assert result.exit_code == 0, result.stderr
        assert "delete legacy" in result.stdout
        assert not legacy_checkpoint.exists()

    def test_verify_passes_for_clean_legacy(
        self, cli_runner: CliRunner, legacy_checkpoint: Path
    ):
        result = cli_runner.invoke(
            app, [str(legacy_checkpoint), "--verify"]
        )
        assert result.exit_code == 0, result.stderr
        assert "verification passed" in result.stdout

    def test_verify_exits_2_on_mismatch(
        self, cli_runner: CliRunner, legacy_checkpoint: Path, tmp_path: Path
    ):
        # First convert cleanly.
        convert_legacy_checkpoint_to_split(legacy_checkpoint)
        stem = legacy_checkpoint.with_suffix("")
        # Now corrupt the new metadata — bump the epoch by hand.
        meta_path = Path(str(stem) + META_SUFFIX)
        meta = json.loads(meta_path.read_text())
        meta["epoch"] = 999
        meta_path.write_text(json.dumps(meta))

        result = cli_runner.invoke(app, [str(legacy_checkpoint), "--verify"])
        assert result.exit_code == 2, (
            f"expected exit 2, got {result.exit_code}: {result.stdout!r} / {result.stderr!r}"
        )
        assert "verification failed" in result.stderr or "verification failed" in result.stdout

    def test_missing_legacy_exits_1(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        result = cli_runner.invoke(app, [str(tmp_path / "nope.pt")])
        assert result.exit_code == 1
        assert "not found" in result.stderr

    def test_split_layout_already_present_exits_1(
        self, cli_runner: CliRunner, legacy_checkpoint: Path
    ):
        # Pre-create a sidecar so the convert refuses.
        stem = legacy_checkpoint.with_suffix("")
        stem.with_suffix(SAFETENSORS_SUFFIX).write_bytes(b"existing")

        result = cli_runner.invoke(app, [str(legacy_checkpoint)])
        assert result.exit_code == 1
        assert "already exists" in result.stderr

    def test_accepts_stem_path(
        self, cli_runner: CliRunner, legacy_checkpoint: Path
    ):
        stem = legacy_checkpoint.with_suffix("")
        result = cli_runner.invoke(app, [str(stem)])
        assert result.exit_code == 0, result.stderr
        assert stem.with_suffix(SAFETENSORS_SUFFIX).exists()
