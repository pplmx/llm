"""Tests for generation backend abstraction."""

import pytest

from llm.generation.backends import EagerGenerationBackend, GenerationConfig, get_generation_backend


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
    from llm.generation.backends import BatchedGenerationBackend

    assert isinstance(backend, BatchedGenerationBackend)


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.max_new_tokens == 128
    assert config.use_cache is True


def test_eager_backend_generate(tiny_model, device):
    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "x"

        pad_token_id = 0

    backend = EagerGenerationBackend()
    output = backend.generate(
        model=tiny_model.to(device),
        tokenizer=_Tok(),
        prompt="hi",
        config=GenerationConfig(max_new_tokens=2, use_cache=False),
    )
    assert output.startswith("hi")


def test_eager_backend_batch_generate(tiny_model, device):
    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "x"

        pad_token_id = 0

    backend = EagerGenerationBackend()
    outputs = backend.batch_generate(
        model=tiny_model.to(device),
        tokenizer=_Tok(),
        prompts=["a", "b"],
        config=GenerationConfig(max_new_tokens=1, temperature=0.0, use_cache=False),
    )
    assert len(outputs) == 2
    assert all(text.startswith("x") or len(text) >= 1 for text in outputs)
