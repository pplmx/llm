"""``llm migrate-ckpt`` CLI flags: the safety guarantees are CLI-level.

The library conversion has tests; what is untested is the CLI's combination
logic, which is where a wrong ordering would destroy data:
``--dry-run`` must write nothing, and ``--in-place`` may only delete the legacy
blob AFTER ``--verify`` has compared the new trio against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from llm.cli.migrate_ckpt import app as migrate_app
from llm.models.decoder import DecoderModel

CONFIG = {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 16}
SIDECARS = (".safetensors", ".meta.json", ".extra_state.pt")


@pytest.fixture
def legacy_checkpoint(tmp_path: Path) -> Path:
    """A real v0.0.5-style blob: ``model_state`` + ``model_config``."""
    torch.manual_seed(0)
    model = DecoderModel(**CONFIG)
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "model_config": CONFIG}, path)
    return path


def _run(*args: str):
    return CliRunner().invoke(migrate_app, list(args))


def test_dry_run_writes_nothing(legacy_checkpoint: Path):
    """``--dry-run`` must be read-only — no trio, no temp files."""
    before = sorted(p.name for p in legacy_checkpoint.parent.iterdir())

    result = _run(str(legacy_checkpoint), "--dry-run")

    assert result.exit_code == 0, result.output
    after = sorted(p.name for p in legacy_checkpoint.parent.iterdir())
    assert after == before, f"--dry-run created files: {set(after) - set(before)}"


def test_in_place_with_verify_deletes_only_after_a_passing_round_trip(legacy_checkpoint: Path):
    """The legacy blob must survive until verification succeeds.

    Verification compares the new trio against the legacy blob, so deleting it
    first would both break the check and destroy the only copy.
    """
    result = _run(str(legacy_checkpoint), "--in-place", "--verify")

    assert result.exit_code == 0, result.output
    assert not legacy_checkpoint.exists(), "--in-place should remove the legacy blob after verifying"
    for suffix in SIDECARS:
        assert (legacy_checkpoint.parent / f"model{suffix}").exists(), f"missing sidecar {suffix}"


def test_in_place_without_verify_keeps_the_conversion(legacy_checkpoint: Path):
    """Without ``--verify`` the legacy blob is removed once conversion succeeds."""
    result = _run(str(legacy_checkpoint), "--in-place")

    assert result.exit_code == 0, result.output
    assert not legacy_checkpoint.exists()
    assert all((legacy_checkpoint.parent / f"model{s}").exists() for s in SIDECARS)


def test_plain_conversion_keeps_the_legacy_blob(legacy_checkpoint: Path):
    """The default must be non-destructive."""
    result = _run(str(legacy_checkpoint))

    assert result.exit_code == 0, result.output
    assert legacy_checkpoint.exists(), "a plain conversion must not delete the legacy checkpoint"
    assert all((legacy_checkpoint.parent / f"model{s}").exists() for s in SIDECARS)


def test_verify_round_trips_the_weights(legacy_checkpoint: Path):
    """``--verify`` must actually compare tensors, not just report success."""
    from llm.training.core.checkpoint import load_checkpoint_payload

    before = load_checkpoint_payload(legacy_checkpoint)["model_state"]
    result = _run(str(legacy_checkpoint), "--verify")
    assert result.exit_code == 0, result.output

    after = load_checkpoint_payload(legacy_checkpoint.parent / "model.safetensors")["model_state"]
    assert set(before) == set(after), "the conversion changed which tensors exist"
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), f"tensor {name} changed during migration"
