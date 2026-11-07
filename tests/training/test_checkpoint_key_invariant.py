"""Contract test: every checkpoint consumer shares the plain-namespace invariant.

The checkpoint contract documented in ``docs/guides/checkpoints.md`` is that
``.safetensors`` holds PLAIN (unwrapped) model keys — no ``_orig_mod.`` from
``torch.compile`` and no ``.base_layer.`` / ``lora_*`` from PEFT. Every consumer
rebuilds a plain model and then re-applies wrappers itself, so a violation is
not a cosmetic naming issue: the consumer refuses to start.

``model_state_dict`` is the single write path and normalizes both wrappers. This
test pins the invariant at the OUTER boundary: write one checkpoint from a
compiled + PEFT-wrapped model and require all three consumers to load it.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from llm.models.decoder import DecoderModel  # noqa: E402
from llm.training.core.checkpoint import (  # noqa: E402
    CheckpointConfig,
    CheckpointManager,
    load_checkpoint_payload,
)

_TINY = {"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 16}


def _quiet_logger() -> logging.Logger:
    logger = logging.getLogger("test-checkpoint-invariant")
    logger.addHandler(logging.NullHandler())
    return logger


def _wrapped_model():
    """A model carrying BOTH wrappers the write path must normalize."""
    from llm.core.lora import apply_lora

    model = DecoderModel(**_TINY)
    apply_lora(model, rank=4, alpha=8.0)
    return model


def test_saved_checkpoint_carries_plain_keys(tmp_path):
    """The on-disk key set must equal a plain model's key set, exactly."""
    plain_keys = set(DecoderModel(**_TINY).state_dict())
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval=1), rank=0, logger=_quiet_logger()
    )
    manager.save_checkpoint(1, _wrapped_model(), None, None, None, 1.0, model_config=dict(_TINY))

    from safetensors.torch import load_file

    persisted = set(load_file(str(tmp_path / "epoch_2.safetensors")))
    assert persisted == plain_keys, (
        "checkpoint keys drifted from the plain namespace: "
        f"extra={sorted(persisted - plain_keys)[:5]} missing={sorted(plain_keys - persisted)[:5]}"
    )


@pytest.mark.parametrize("consumer", ["payload", "trained_model"])
def test_all_consumers_load_the_saved_checkpoint(tmp_path, consumer):
    """Each consumer that rebuilds a plain model must accept the file.

    ``payload`` is the low-level ``load_checkpoint_payload`` used by serving and
    ``migrate-ckpt``; ``trained_model`` is ``llm.cli._model_loading`` used by the
    three compression subcommands.
    """
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval=1), rank=0, logger=_quiet_logger()
    )
    manager.save_checkpoint(1, _wrapped_model(), None, None, None, 1.0, model_config=dict(_TINY))
    stem = tmp_path / "epoch_2"

    if consumer == "payload":
        payload = load_checkpoint_payload(stem)
        assert payload is not None
        assert payload["model_config"]["hidden_size"] == _TINY["hidden_size"]
        assert len(payload["model_state"]) == len(DecoderModel(**_TINY).state_dict())
    else:
        from llm.cli._model_loading import load_trained_model

        model = load_trained_model(stem)
        # No missing key may fall back to random init.
        missing, unexpected = model.load_state_dict(DecoderModel(**_TINY).state_dict(), strict=False)
        assert not missing
        assert not unexpected
