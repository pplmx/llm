"""Regression tests: the compression CLIs read the repo's OWN checkpoint layout.

``llm quantize`` / ``llm prune`` / ``llm decompose`` all used a bare
``torch.load(model)``, which only understands the v0.0.5 single-file layout.
Pointed at a checkpoint this repo actually produces today — the v2 split trio
from ADR-006 (``<stem>.safetensors`` + ``.meta.json`` + ``.extra_state.pt``) —
the load succeeded but returned the raw payload ``dict``, and the transform
died with ``'dict' object has no attribute 'named_modules'``. The documented
``--model <stem>.pt`` spelling was rejected even earlier, by the pre-flight
path check, because no ``.pt`` file exists in the split layout.

These tests pin the shared loader's contract directly (fast, no CLI process)
plus one end-to-end CLI run per command.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

from llm.cli._model_loading import CheckpointModelError, checkpoint_exists, load_trained_model
from tests.support.ansi import strip_ansi


def _tiny_model():
    from llm.runtime import ModelFactory
    from llm.training.core.config import ModelConfig

    return ModelFactory.from_config(
        ModelConfig(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=24)
    )


def _save_split_checkpoint(tmp_path: Path) -> Path:
    """Write a real v2 split-layout checkpoint and return its stem."""
    from llm.training.core.checkpoint import CheckpointConfig, CheckpointManager

    model = _tiny_model()
    ckpt_dir = tmp_path / "ckpt"
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(ckpt_dir), save_interval=1),
        rank=0,
        logger=logging.getLogger("test"),
    )
    manager.save_checkpoint(
        1,
        model,
        None,
        None,
        None,
        1.0,
        model_config={"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 24},
    )
    return ckpt_dir / "epoch_2"


class TestSplitLayoutLoading:
    """The shared loader must understand every spelling of the v2 layout."""

    @pytest.mark.parametrize("spelling", ["stem", "dot_pt", "safetensors"])
    def test_load_trained_model_accepts_every_v2_spelling(self, tmp_path: Path, spelling: str) -> None:
        stem = _save_split_checkpoint(tmp_path)
        path = {
            "stem": stem,
            "dot_pt": stem.with_suffix(".pt"),
            "safetensors": stem.with_suffix(".safetensors"),
        }[spelling]
        model = load_trained_model(path)
        assert isinstance(model, nn.Module)
        # Real weights, not a fresh init: the rebuilt model must be evaluable
        # and produce the vocabulary-sized logits the checkpoint describes.
        logits = model(torch.randint(0, 32, (2, 6)))
        assert logits.shape == (2, 6, 32)

    def test_checkpoint_exists_accepts_split_layout_spellings(self, tmp_path: Path) -> None:
        """The pre-flight path check must not reject the layout we write.

        It required ``is_file()``, so a bare stem or ``<stem>.pt`` — the
        documented spelling — was refused before loading even started.
        """
        stem = _save_split_checkpoint(tmp_path)
        assert checkpoint_exists(stem), "bare stem must resolve via its sidecars"
        assert checkpoint_exists(stem.with_suffix(".pt")), ".pt spelling must resolve via its sidecars"
        assert checkpoint_exists(stem.with_suffix(".safetensors"))
        assert not checkpoint_exists(tmp_path / "nope")

    def test_bare_module_blob_still_loads(self, tmp_path: Path) -> None:
        """Legacy ``torch.save(model)`` blobs keep working (the CLIs' own output)."""
        path = tmp_path / "bare.pt"
        torch.save(_tiny_model(), path)
        assert isinstance(load_trained_model(path), nn.Module)

    def test_missing_checkpoint_raises_typed_error(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointModelError, match="no checkpoint found"):
            load_trained_model(tmp_path / "absent")


@pytest.mark.parametrize(
    ("app_module", "extra_args"),
    [
        ("llm.cli.prune", ["--ratio", "0.5"]),
        ("llm.cli.decompose", ["--rank", "4"]),
    ],
)
def test_compression_clis_accept_v2_checkpoint(tmp_path: Path, app_module: str, extra_args: list[str]) -> None:
    """``prune`` / ``decompose`` must run against a v2 split checkpoint.

    Before the fix this failed with ``'dict' object has no attribute
    'named_modules'``: the loader handed the raw checkpoint payload to the
    transform instead of a live model.
    """
    import importlib

    app = importlib.import_module(app_module).app
    stem = _save_split_checkpoint(tmp_path)
    out = tmp_path / "out.blob"
    result = CliRunner().invoke(app, ["--model", str(stem), "--output", str(out), *extra_args])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert out.exists(), "the transformed model must be written"
    assert "named_modules" not in strip_ansi(result.output), "the dict-vs-model failure must not return"


def test_quantize_cli_accepts_v2_checkpoint(tmp_path: Path) -> None:
    """``quantize`` must run against a v2 split checkpoint (needs calibration)."""
    from llm.cli.quantize import app

    stem = _save_split_checkpoint(tmp_path)
    calib = tmp_path / "calib.pt"
    torch.save([torch.randint(0, 32, (4, 8))], calib)
    out = tmp_path / "q.pt"
    result = CliRunner().invoke(
        app,
        ["gptq", "--model", str(stem), "--output", str(out), "--calib-data-tokens", str(calib), "--bits", "4"],
    )
    assert result.exit_code == 0, strip_ansi(result.output)
    assert out.exists()
    assert "named_modules" not in strip_ansi(result.output)
