"""Validation contract tests for the serving request schemas."""

import pytest
from pydantic import ValidationError

from llm.serving.schemas import BatchGenerationRequest, ChatCompletionRequest, GenerationRequest


def test_generation_request_accepts_up_to_four_stops():
    req = GenerationRequest(prompt="hi", stop=["a", "b", "c", "d"])
    assert req.stop == ["a", "b", "c", "d"]


def test_generation_request_rejects_more_than_four_stops():
    """OpenAI caps stop at 4 strings; the schema must enforce the documented
    contract so the per-token suffix check stays bounded."""
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", stop=["a"] * 5)


def test_batch_generation_request_rejects_more_than_four_stops():
    with pytest.raises(ValidationError):
        BatchGenerationRequest(prompts=["hi"], stop=["a"] * 5)


def test_chat_completion_request_rejects_more_than_four_stops():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stop=["a"] * 5)


def test_generation_request_rejects_empty_stop_list():
    """[] means 'no stop'; an empty list is accepted and normalized later."""
    req = GenerationRequest(prompt="hi", stop=[])
    assert req.stop == []
