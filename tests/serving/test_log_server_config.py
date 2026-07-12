"""Unit tests for ``_log_server_config`` (audit Finding AZ).

Verifies that the structured startup log line includes all fields an operator
needs for incident triage: model class, parameter count, dtype, device, max_seq_len,
attn/mlp impl, generation backend, prefix cache flag, paged attention flag,
api_key_set (bool only — never the key value).
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from llm.serving.api import _log_server_config


class _FakeService:
    """Minimal stand-in for ServingGenerationService.

    The real service holds (model, tokenizer, backend, device); we only need
    the model here because ``_log_server_config`` reads model fields.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model


@pytest.fixture
def tiny_model() -> nn.Module:
    return nn.Sequential(OrderedDict([("linear", nn.Linear(8, 4))]))


@pytest.fixture
def fake_service(tiny_model):
    return _FakeService(tiny_model)


class _CapturingHandler(logging.Handler):
    """Capture log records so tests can inspect extras."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log_capture() -> _CapturingHandler:
    handler = _CapturingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    prior_level = root.level
    root.setLevel(logging.DEBUG)
    yield handler
    root.removeHandler(handler)
    root.setLevel(prior_level)


_REQUIRED_KEYS = {
    "event",
    "model_class",
    "param_count_total",
    "param_count_trainable",
    "dtype",
    "device",
    "max_seq_len",
    "attn_impl",
    "mlp_impl",
    "generation_backend",
    "enable_prefix_cache",
    "use_paged_attention",
    "api_key_set",
}


def test_log_server_config_emits_required_keys(fake_service, log_capture):
    """All required fields are present on the emitted record."""
    from llm.serving.config import ServingConfig

    config = ServingConfig(
        max_seq_len=512,
        attn_impl="mha",
        mlp_impl="mlp",
        generation_backend="eager",
        enable_prefix_cache=True,
        use_paged_attention=False,
        api_key="some-secret",  # should appear as api_key_set=True, never as value
    )

    _log_server_config(fake_service, config)

    records = [r for r in log_capture.records if r.getMessage() == "server_config"]
    assert len(records) == 1
    rec = records[0]
    missing = _REQUIRED_KEYS - set(rec.__dict__.keys())
    assert not missing, f"missing required keys: {missing}"


def test_log_server_config_param_counts(fake_service, log_capture):
    from llm.serving.config import ServingConfig

    config = ServingConfig()

    _log_server_config(fake_service, config)

    rec = next(r for r in log_capture.records if r.getMessage() == "server_config")
    # nn.Linear(8, 4) has 8*4 + 4 = 36 params
    assert rec.param_count_total == 36
    assert rec.param_count_trainable == 36


def test_log_server_config_api_key_set_true(fake_service, log_capture):
    from llm.serving.config import ServingConfig

    config = ServingConfig(api_key="some-secret")
    _log_server_config(fake_service, config)
    rec = next(r for r in log_capture.records if r.getMessage() == "server_config")
    assert rec.api_key_set is True
    # And the secret must not leak anywhere on the record.
    assert "some-secret" not in str(rec.__dict__)


def test_log_server_config_api_key_set_false(fake_service, log_capture):
    from llm.serving.config import ServingConfig

    config = ServingConfig(api_key=None)
    _log_server_config(fake_service, config)
    rec = next(r for r in log_capture.records if r.getMessage() == "server_config")
    assert rec.api_key_set is False


def test_log_server_config_dtype_device_from_model(fake_service, log_capture):
    """dtype/device are pulled from the model's actual parameters, not the config."""
    from llm.serving.config import ServingConfig

    config = ServingConfig()

    # Cast model to float32 on cpu for predictability.
    fake_service.model.to(torch.float32)

    _log_server_config(fake_service, config)
    rec = next(r for r in log_capture.records if r.getMessage() == "server_config")
    assert rec.dtype == "torch.float32"
    assert rec.device == "cpu"


def test_log_server_config_handles_paramless_model(log_capture):
    """A model with no parameters still produces a valid log line."""
    from llm.serving.config import ServingConfig

    class _NoParams(nn.Module):
        def forward(self, x):  # pragma: no cover - not invoked
            return x

    service = _FakeService(_NoParams())
    config = ServingConfig()

    _log_server_config(service, config)

    rec = next(r for r in log_capture.records if r.getMessage() == "server_config")
    assert rec.param_count_total == 0
    assert rec.param_count_trainable == 0
    # dtype/device fall back to "unknown" rather than crashing
    assert rec.dtype == "unknown"
    assert rec.device == "unknown"
