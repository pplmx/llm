"""Tests for generation backend abstraction."""

from llm.generation.backends import EagerGenerationBackend, GenerationConfig, get_generation_backend


def test_get_generation_backend_eager():
    backend = get_generation_backend("eager")
    assert isinstance(backend, EagerGenerationBackend)


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
