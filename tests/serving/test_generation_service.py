"""Tests for ServingGenerationService."""

import pytest

from llm.generation.backends import EagerGenerationBackend
from llm.serving.generation_service import ServingGenerationService


@pytest.mark.quick
def test_generation_service_generate(tiny_model, device, stub_tokenizer):
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    output = service.generate(prompt="hi", max_new_tokens=2, temperature=0.0)
    assert output.startswith("hi")
    assert len(output) > len("hi")


@pytest.mark.quick
def test_generation_service_stream(tiny_model, device, stub_tokenizer):
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    chunks = list(service.stream(prompt="hi", max_new_tokens=2, temperature=0.0))
    assert len(chunks) == 2


@pytest.mark.quick
def test_generation_service_rejects_max_new_tokens_above_context(tiny_model, device, stub_tokenizer):
    """max_new_tokens >= the model's max_seq_len leaves no room for a prompt.

    The eager backend used to slice the input to an *empty* tensor in that
    case and 500 on the forward; the service now rejects it up front with a
    clear ValueError so the API can surface a 4xx instead.
    """
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )
    max_seq = getattr(tiny_model, "max_seq_len", None)
    assert max_seq is not None, "tiny_model must expose max_seq_len"

    with pytest.raises(ValueError, match="max_new_tokens"):
        service.generate(prompt="hi", max_new_tokens=max_seq, temperature=0.0)
