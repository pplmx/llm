"""Tests for :func:`llm.utils.serialization.register_framework_safe_globals`.

Pins the failure semantics (RIL ISS-243): a submodule import failure during
the allowlist walk must be LOUD (a warning naming the culprit) and must NOT
finalise registration — otherwise one broken module permanently pins a
partial allowlist (``torch.load(weights_only=True)`` later raises
``UnpicklingError`` with no repair path). A clean later walk self-heals.
"""

from __future__ import annotations

import importlib
import logging

import pytest
import torch

from llm.utils import serialization as ser


@pytest.fixture(autouse=True)
def _reset_registered_flag():
    """Each test starts (and ends) with an un-finalised allowlist."""
    ser._SAFE_GLOBALS_REGISTERED = False
    yield
    ser._SAFE_GLOBALS_REGISTERED = False


def test_register_clean_walk_finalises(caplog: pytest.LogCaptureFixture):
    """A clean walk pins the flag and emits no partial-allowlist warning."""
    with caplog.at_level(logging.WARNING):
        ser.register_framework_safe_globals()
    assert ser._SAFE_GLOBALS_REGISTERED is True
    assert "PARTIAL" not in caplog.text


def test_register_allows_framework_module_loads_weights_only():
    """After registration a ``torch.save``'d framework model loads under
    ``weights_only=True`` (the security contract the allowlist exists for).
    Mirrors the real serve/quantize blob path — the ``torch.save`` codec, NOT
    raw ``pickle.dumps`` (which embeds in-band storage buffers that the
    weights-only unpickler handles differently). The REFUSAL side of the
    contract is pinned by the existing weights_only RCE tests."""
    import io

    from llm.models.decoder import DecoderModel

    buf = io.BytesIO()
    torch.save(DecoderModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=4), buf)
    buf.seek(0)

    ser.register_framework_safe_globals()
    restored = torch.load(buf, weights_only=True)
    assert isinstance(restored, DecoderModel)


def test_partial_walk_warns_and_does_not_finalise(monkeypatch, caplog: pytest.LogCaptureFixture):
    """A broken submodule is LOUD and leaves registration retryable.

    Regression for ISS-243: the walk used to swallow every import error yet
    still set ``_SAFE_GLOBALS_REGISTERED = True``, permanently freezing a
    partial allowlist with no way to repair it.
    """
    real_import = importlib.import_module
    broke_once = {"n": 0}
    break_name = "llm.core.moe.moe"  # a real framework submodule, broken once

    def _flaky_import(name: str, *args, **kwargs):
        if name == break_name and broke_once["n"] == 0:
            broke_once["n"] += 1
            raise ImportError("simulated broken optional dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _flaky_import)
    with caplog.at_level(logging.WARNING):
        ser.register_framework_safe_globals()
    assert "PARTIAL" in caplog.text
    assert break_name in caplog.text
    assert ser._SAFE_GLOBALS_REGISTERED is False  # NOT pinned while partial

    # A later clean walk self-heals and finalises.
    ser.register_framework_safe_globals()
    assert ser._SAFE_GLOBALS_REGISTERED is True
