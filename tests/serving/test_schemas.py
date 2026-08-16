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


def test_generation_request_accepts_long_single_stop_string():
    """The documented contract is 'a single string OR a list of up to 4
    strings'.  ``max_length=4`` on the union rejected any single stop longer
    than 4 characters (e.g. ``"\\n\\n###"`` or ``"Human:"``) with a 422 —
    list cardinality, not string length, is what's capped."""
    req = GenerationRequest(prompt="hi", stop="\n\n###")
    assert req.stop == "\n\n###"
    req2 = GenerationRequest(prompt="hi", stop="Human: I am")
    assert req2.stop == "Human: I am"


def test_batch_generation_request_accepts_long_single_stop_string():
    req = BatchGenerationRequest(prompts=["hi"], stop="\n\n###")
    assert req.stop == "\n\n###"


def test_chat_completion_request_accepts_long_single_stop_string():
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stop="\n\n###")
    assert req.stop == "\n\n###"


def test_top_p_accepts_openai_default_one():
    """top_p=1.0 is the OpenAI-compatible identity ('no nucleus truncation')
    and the overwhelmingly common client value; `lt=1.0` rejected it as a 422.
    It must be accepted."""
    req = GenerationRequest(prompt="hi", top_p=1.0)
    assert req.top_p == 1.0
    req2 = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], top_p=1.0)
    assert req2.top_p == 1.0


# --- Request-input bounds (RIL ISS-171): schema-level 422 fast-fail caps ---


def test_generation_request_caps_prompt_length():
    """prompt over MAX_PROMPT_CHARS is rejected before any encoding."""
    from llm.serving.schemas import MAX_PROMPT_CHARS, GenerationRequest

    with pytest.raises(ValidationError):
        GenerationRequest(prompt="x" * (MAX_PROMPT_CHARS + 1))
    # Exactly at the cap is still fine.
    req = GenerationRequest(prompt="x" * MAX_PROMPT_CHARS)
    assert len(req.prompt) == MAX_PROMPT_CHARS


@pytest.mark.parametrize(
    "req_kwargs",
    [
        {"prompt": "hi", "stop": ["a" * 600]},
        {"prompt": "hi", "stop": "b" * 600},
    ],
)
def test_generation_request_rejects_oversized_stop(req_kwargs):
    from llm.serving.schemas import GenerationRequest

    with pytest.raises(ValidationError):
        GenerationRequest(**req_kwargs)


def test_batch_generation_request_caps_each_prompt():
    from llm.serving.schemas import MAX_PROMPT_CHARS, BatchGenerationRequest

    with pytest.raises(ValidationError):
        BatchGenerationRequest(prompts=["ok", "y" * (MAX_PROMPT_CHARS + 1)])
    # Cardinality over the batch cap is also rejected.
    with pytest.raises(ValidationError):
        BatchGenerationRequest(prompts=["ok"] * 33)


def test_chat_completion_request_caps_messages_and_content():
    from llm.serving.schemas import MAX_MESSAGES, MAX_PROMPT_CHARS, ChatCompletionRequest

    # Single message content over cap.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "z" * (MAX_PROMPT_CHARS + 1)}])
    # Too many messages.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}] * (MAX_MESSAGES + 1))
    # Legit list still validates.
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.messages[0].content == "hi"
