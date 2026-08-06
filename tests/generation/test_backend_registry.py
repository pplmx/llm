"""Tests for generation backend registry and backend behavior."""

import pytest

from llm.generation.backends import BatchedGenerationBackend, EagerGenerationBackend, GenerationConfig
from llm.generation.registry import BACKEND_REGISTRY, ensure_backends_registered, get_generation_backend
from llm.serving.batch_engine import ContinuousBatchingEngine


def test_get_generation_backend_eager():
    ensure_backends_registered()
    backend = get_generation_backend("eager")
    assert type(backend) is EagerGenerationBackend


def test_get_generation_backend_batched_requires_engine(tiny_model, device):
    ensure_backends_registered()

    with pytest.raises(ValueError, match="requires a ContinuousBatchingEngine"):
        get_generation_backend("batched")

    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=object(),
        device=device,
    )
    backend = get_generation_backend("batched", engine=engine)
    assert type(backend) is BatchedGenerationBackend
    assert backend.engine is engine


def test_unknown_backend_raises():
    ensure_backends_registered()
    with pytest.raises(ValueError, match="not found"):
        get_generation_backend("unknown-backend")


def test_duplicate_backend_registration_raises():
    ensure_backends_registered()
    with pytest.raises(ValueError, match="already registered"):
        BACKEND_REGISTRY.register("eager", lambda **kwargs: object())


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.max_new_tokens == 128
    assert config.use_cache is True


def test_eager_backend_generate(tiny_model, device, stub_tokenizer):
    backend = EagerGenerationBackend()
    output = backend.generate(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        prompt="hi",
        config=GenerationConfig(max_new_tokens=2, use_cache=False),
    )
    assert output.startswith("hi")
    assert len(output) == len("hi") + 2


def test_eager_backend_batch_generate(tiny_model, device, stub_tokenizer):
    backend = EagerGenerationBackend()
    outputs = backend.batch_generate(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        prompts=["a", "b"],
        config=GenerationConfig(max_new_tokens=1, temperature=0.0, use_cache=False),
    )
    assert len(outputs) == 2
    assert all(text.endswith("x") for text in outputs)
