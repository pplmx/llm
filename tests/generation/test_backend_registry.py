"""Tests for generation backend registry."""

import pytest

from llm.generation.backends import BatchedGenerationBackend, EagerGenerationBackend
from llm.generation.registry import BACKEND_REGISTRY, ensure_backends_registered, get_generation_backend


def test_backend_registry_contains_builtins():
    ensure_backends_registered()
    assert "eager" in BACKEND_REGISTRY
    assert "batched" in BACKEND_REGISTRY


def test_get_generation_backend_eager():
    backend = get_generation_backend("eager")
    assert isinstance(backend, EagerGenerationBackend)


def test_get_generation_backend_batched_requires_engine(tiny_model, device):
    from llm.serving.batch_engine import ContinuousBatchingEngine

    with pytest.raises(ValueError, match="requires a ContinuousBatchingEngine"):
        get_generation_backend("batched")

    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=object(),
        device=device,
    )
    backend = get_generation_backend("batched", engine=engine)
    assert isinstance(backend, BatchedGenerationBackend)


def test_unknown_backend_raises():
    ensure_backends_registered()
    with pytest.raises(ValueError, match="not found"):
        get_generation_backend("unknown-backend")
