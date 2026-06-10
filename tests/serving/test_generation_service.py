"""Tests for ServingGenerationService."""

import pytest

from llm.generation.backends import EagerGenerationBackend
from llm.serving.generation_service import ServingGenerationService


@pytest.mark.quick
def test_generation_service_generate(tiny_model, device):
    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "x"

        pad_token_id = 0

    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=_Tok(),
        backend=EagerGenerationBackend(),
        device=device,
    )

    output = service.generate(prompt="hi", max_new_tokens=2, temperature=0.0)
    assert output.startswith("hi")
    assert len(output) > len("hi")


@pytest.mark.quick
def test_generation_service_stream(tiny_model, device):
    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "x"

        pad_token_id = 0

    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=_Tok(),
        backend=EagerGenerationBackend(),
        device=device,
    )

    chunks = list(service.stream(prompt="hi", max_new_tokens=2, temperature=0.0))
    assert len(chunks) == 2
