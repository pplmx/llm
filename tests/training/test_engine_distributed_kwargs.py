"""Contract test: the engine forwards every FSDP knob from the config.

``wrap_model_for_training`` has always accepted ``fsdp_mixed_precision`` /
``fsdp_auto_wrap_min_params`` / ``fsdp_cpu_offload``, and
``DistributedConfig`` declares and documents all three — but the engine's call
site forwarded only the parallel-shape arguments, so those three YAML keys were
silently ignored and every FSDP run got the wrapper defaults. Nothing caught it
because no test asserted the engine -> wrapper argument contract.

The test captures the kwargs the engine actually passes and compares them with
the config, for both the declared and the non-default values.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.sft_task import SFTTask


def _tiny_config(**distributed_overrides) -> Config:
    cfg = Config()
    cfg.model.hidden_size = 8
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 16
    cfg.model.num_layers = 1
    cfg.model.vocab_size = 32
    cfg.model.max_seq_len = 8
    cfg.training.epochs = 1
    for key, value in distributed_overrides.items():
        setattr(cfg.distributed, key, value)
    return cfg


def _build_engine(cfg: Config):
    """Construct an engine with no-op dataloader shims; return (engine, captured)."""
    empty_iter = MagicMock()
    empty_iter.__iter__ = MagicMock(return_value=iter([]))
    empty_iter.__len__ = MagicMock(return_value=0)
    loader_pair = (empty_iter, None)
    fake_data_module = MagicMock(
        train_dataloader=MagicMock(return_value=loader_pair),
        val_dataloader=MagicMock(return_value=loader_pair),
        is_streaming=False,
    )

    captured: dict = {}

    def _capture(model, **kwargs):
        captured.update(kwargs)
        return model

    with patch("llm.training.core.engine.wrap_model_for_training", side_effect=_capture):
        task = SFTTask(cfg, data_module=fake_data_module)
        task.build_model = lambda: torch.nn.Linear(cfg.model.hidden_size, cfg.model.vocab_size)
        engine = TrainingEngine(
            config=cfg,
            task=task,
            rank=0,
            world_size=1,
            data_module=fake_data_module,
            callbacks=None,
        )
    return engine, captured


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fsdp_mixed_precision", "fp32"),
        ("fsdp_auto_wrap_min_params", 1234),
        ("fsdp_cpu_offload", True),
    ],
)
def test_engine_forwards_fsdp_knobs_from_config(field: str, value) -> None:
    """Changing a declared FSDP knob must change what the wrapper receives.

    Before the fix the wrapper always saw its own defaults, so a YAML that set
    ``distributed.fsdp_mixed_precision: fp32`` trained in bf16 anyway.
    """
    cfg = _tiny_config(**{field: value})
    _, captured = _build_engine(cfg)
    assert field in captured, f"engine never forwarded {field}; passed keys: {sorted(captured)}"
    assert captured[field] == value, f"{field}: engine passed {captured[field]!r}, config says {value!r}"


def test_engine_forwards_fsdp_defaults_when_unset() -> None:
    """The defaults must also be forwarded, not just left to the wrapper.

    Forwarding explicitly keeps the single source of truth in the config: a
    future default change in ``DistributedConfig`` takes effect without also
    editing the wrapper's signature default.
    """
    cfg = _tiny_config()
    _, captured = _build_engine(cfg)
    for field in ("fsdp_mixed_precision", "fsdp_auto_wrap_min_params", "fsdp_cpu_offload"):
        assert captured[field] == getattr(cfg.distributed, field)
