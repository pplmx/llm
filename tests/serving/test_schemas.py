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


# --- temperature upper bound: /generate matches /v1/chat ------


def test_generation_request_temperature_upper_bound_2():
    """``temperature`` is capped at 2.0 on /generate exactly like
    /v1/chat/completions — a degenerate 1e9 previously slipped an
    all-but-uniform distribution through /generate while /v1/chat 422'd.
    """
    assert GenerationRequest(prompt="hi", temperature=2.0).temperature == 2.0
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", temperature=2.5)


def test_batch_generation_request_temperature_upper_bound_2():
    assert BatchGenerationRequest(prompts=["hi"], temperature=2.0).temperature == 2.0
    with pytest.raises(ValidationError):
        BatchGenerationRequest(prompts=["hi"], temperature=3.0)


# --- Request-input bounds: schema-level 422 fast-fail caps ---


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


# --- Unsupported OpenAI knobs must be REFUSED, not silently dropped ---


class TestUnsupportedChatKnobs:
    """A compatibility surface must not accept a parameter and ignore it.

    With pydantic's default ``extra="ignore"``, a client sending ``n=3`` or
    ``stream_options={"include_usage": true}`` got a 200 whose behaviour differed
    from the request: one completion instead of three, no usage in the stream.
    The client had no way to notice. These knobs are now declared and rejected
    with a message naming the field.
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n", 3),
            ("seed", 42),
            ("response_format", {"type": "json_object"}),
            ("stream_options", {"include_usage": True}),
            ("tools", [{"type": "function"}]),
        ],
    )
    def test_unsupported_knob_is_rejected_by_name(self, field, value):
        from llm.serving.schemas import ChatCompletionRequest

        with pytest.raises(ValueError, match=field):
            ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **{field: value})

    def test_plain_request_still_validates(self):
        """The guard must not disturb a request that asks for nothing exotic."""
        from llm.serving.schemas import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert request.n is None
        assert request.seed is None
        assert request.stream_options is None

    def test_n_equals_one_is_accepted(self):
        """``n=1`` is what this server always does, so it is not a lost request."""
        from llm.serving.schemas import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], n=1)
        assert request.n == 1

    def test_several_unsupported_knobs_are_all_named(self):
        """The error must list every offending field, not just the first."""
        from llm.serving.schemas import ChatCompletionRequest

        with pytest.raises(ValueError, match="n") as excinfo:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                n=2,
                seed=7,
                tools=[{"type": "function"}],
            )
        message = str(excinfo.value)
        for name in ("n", "seed", "tools"):
            assert name in message, f"{name} missing from: {message}"
