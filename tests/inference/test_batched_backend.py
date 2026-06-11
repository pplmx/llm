"""Tests for BatchedGenerationBackend."""

import pytest

from llm.generation.backends import BatchedGenerationBackend, GenerationConfig
from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.schemas import GenerationRequest


@pytest.mark.quick
def test_batched_backend_generate(tiny_model, device, stub_tokenizer):
    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        device=device,
        max_batch_size=2,
    )
    backend = BatchedGenerationBackend(engine)
    output = backend.generate(
        model=engine.model,
        tokenizer=stub_tokenizer,
        prompt="hi",
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
    )
    assert output.startswith("hi")


@pytest.mark.quick
def test_engine_stream_request_respects_max_new_tokens(tiny_model, device, stub_tokenizer):
    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        device=device,
        max_batch_size=2,
    )
    request = GenerationRequest(prompt="test", max_new_tokens=2)
    chunks = list(engine.stream_request(request))
    assert len(chunks) == 2
