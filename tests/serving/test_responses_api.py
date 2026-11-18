"""Tests for the OpenAI Responses API endpoint (``POST /v1/responses``).

OpenAI recommends the Responses API over chat completions, so compatibility
means more than serving ``/v1/chat/completions``. The shape differs enough that
the endpoint cannot be an alias — these tests pin each difference against the
SDK's own types.
"""

from __future__ import annotations

import json

import pytest


def _schema():
    from llm.serving import schemas

    return schemas


def test_a_string_input_becomes_a_single_user_message():
    """``input="hi"`` is the documented shorthand for one user message."""
    from llm.serving.routers.responses import _to_chat_messages

    schema = _schema()
    request = schema.ResponsesRequest(input="hello")
    messages = _to_chat_messages(request)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "hello"


def test_instructions_become_a_leading_system_message():
    """``instructions`` is the Responses spelling of a system message.

    The server keeps ONE prompt renderer for both endpoints, so instructions
    must land in the message list the chat template already knows how to render
    — dropping them would silently discard the client's highest-priority text.
    """
    from llm.serving.routers.responses import _to_chat_messages

    schema = _schema()
    request = schema.ResponsesRequest(input="hello", instructions="be brief")
    messages = _to_chat_messages(request)

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "be brief"


def test_the_developer_role_is_folded_into_system():
    """``developer`` is the Responses-era name for ``system``.

    Passing it through unchanged would make the chat template render
    ``developer: ...``, which no chat format expects.
    """
    from llm.serving.routers.responses import _to_chat_messages

    schema = _schema()
    request = schema.ResponsesRequest(
        input=[
            schema.ResponsesInputMessage(role="developer", content="be brief"),
            schema.ResponsesInputMessage(role="user", content="hello"),
        ]
    )
    messages = _to_chat_messages(request)

    assert [m.role for m in messages] == ["system", "user"]


def test_output_tokens_reports_incomplete_not_finish_reason():
    """Truncation is ``status="incomplete"`` + ``incomplete_details``.

    The Responses API has no ``finish_reason``; a client that hit the budget is
    told through the response status, so a server emitting only the chat-style
    signal would leave those clients unable to see the truncation at all.
    """
    from llm.serving.routers.responses import _build_response

    schema = _schema()
    request = schema.ResponsesRequest(input="hi", max_output_tokens=4)

    truncated = _build_response(
        request=request,
        completion="abcd",
        completion_tokens=4,
        prompt_tokens=10,
        max_output_tokens=4,
    )
    assert truncated.status == "incomplete"
    assert truncated.incomplete_details is not None
    assert truncated.incomplete_details.reason == "max_output_tokens"

    finished = _build_response(
        request=request,
        completion="ab",
        completion_tokens=2,
        prompt_tokens=10,
        max_output_tokens=4,
    )
    assert finished.status == "completed"
    assert finished.incomplete_details is None


def test_the_response_object_uses_the_responses_field_names():
    """Field names must match the SDK, not the chat-completion ones."""
    from llm.serving.routers.responses import _build_response

    schema = _schema()
    request = schema.ResponsesRequest(input="hi", max_output_tokens=8)
    response = _build_response(
        request=request,
        completion="abc",
        completion_tokens=3,
        prompt_tokens=5,
        max_output_tokens=8,
    )

    assert response.object == "response"
    assert response.id.startswith("resp_")
    # ``output`` is a list of items, each holding a ``content`` list.
    assert len(response.output) == 1
    assert response.output[0].type == "message"
    assert response.output[0].role == "assistant"
    assert response.output[0].content[0].type == "output_text"
    assert response.output[0].content[0].text == "abc"
    # Usage uses input/output, not prompt/completion.
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 8
    assert response.usage.output_tokens_details == {"reasoning_tokens": 0}


@pytest.mark.parametrize(
    "knob",
    [
        {"tools": [{"type": "function", "name": "f"}]},
        {"previous_response_id": "resp_x"},
        {"store": True},
        {"background": True},
        {"reasoning": {"effort": "high"}},
    ],
)
def test_unsupported_knobs_are_refused_by_name(knob):
    """A knob the server cannot honour must be a 4xx naming it, not a silent drop.

    Returning 200 with different behaviour leaves the client unable to notice —
    the same contract the chat and completions schemas already enforce.
    """
    from pydantic import ValidationError

    schema = _schema()
    with pytest.raises(ValidationError) as excinfo:
        schema.ResponsesRequest(input="hi", **knob)

    field = next(iter(knob))
    assert field in str(excinfo.value)


def test_an_empty_input_is_rejected():
    from pydantic import ValidationError

    schema = _schema()
    with pytest.raises(ValidationError):
        schema.ResponsesRequest(input="")
    with pytest.raises(ValidationError):
        schema.ResponsesRequest(input=[])


@pytest.mark.slow
def test_the_endpoint_returns_a_response_object(client):
    """End-to-end: POST /v1/responses returns a well-formed response."""
    response = client.post(
        "/v1/responses",
        json={"model": "tiny", "input": "abc", "max_output_tokens": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "response"
    assert body["status"] in ("completed", "incomplete")
    assert len(body["output"]) == 1
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    # ``output_text`` mirrors the item text for plain JSON clients.
    assert body["output_text"] == body["output"][0]["content"][0]["text"]
    assert body["usage"]["output_tokens"] <= 3, "the output budget was not applied"


def test_instructions_reach_the_rendered_prompt():
    """``instructions`` must change the prompt text, not be dropped.

    Checked on the RENDERED string rather than on ``usage.input_tokens``: the
    serving test client uses ``StubTokenizer``, whose ``encode`` returns a fixed
    id list regardless of the text, so token counts cannot distinguish two
    prompts. The rendered prompt is the real contract — the model sees that
    string.
    """
    from llm.serving.chat_template import messages_to_prompt
    from llm.serving.routers.responses import _to_chat_messages

    schema = _schema()
    without = messages_to_prompt(_to_chat_messages(schema.ResponsesRequest(input="abc")))
    with_instructions = messages_to_prompt(
        _to_chat_messages(schema.ResponsesRequest(input="abc", instructions="be extremely brief"))
    )

    assert "be extremely brief" in with_instructions, (
        f"instructions did not reach the prompt; rendered {with_instructions!r}"
    )
    assert "be extremely brief" not in without
    assert len(with_instructions) > len(without)


@pytest.mark.slow
def test_a_tight_budget_yields_an_incomplete_response(client):
    """The budget being consumed must be visible as ``status``."""
    response = client.post(
        "/v1/responses",
        json={"model": "tiny", "input": "abc", "max_output_tokens": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"]["reason"] == "max_output_tokens"


@pytest.mark.slow
def test_streaming_emits_the_event_sequence_the_sdk_accumulates(client):
    """The SSE order is load-bearing for ``client.responses.stream()``.

    The SDK's accumulator applies a ``response.output_text.delta`` to
    ``output[i].content[j]``, so ``response.content_part.added`` MUST precede the
    first delta — measured: omitting it raises ``IndexError`` inside the SDK.
    The terminal event must also be ``completed`` or ``incomplete``, never
    missing, or ``get_final_response()`` raises.
    """
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "tiny", "input": "abc", "max_output_tokens": 3, "stream": True},
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line[len("data: ") :]) for line in response.iter_lines() if line.startswith("data: ")]

    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    assert "response.output_item.added" in types
    assert "response.content_part.added" in types, "the SDK cannot place deltas without this event"
    assert types.index("response.content_part.added") < types.index("response.output_text.delta"), (
        "content_part.added must come BEFORE the first delta"
    )
    assert types[-1] in ("response.completed", "response.incomplete"), f"terminal event was {types[-1]!r}"

    # Sequence numbers must be present and non-decreasing: the SDK relies on them
    # to order events that share a connection.
    numbers = [event["sequence_number"] for event in events]
    assert numbers == sorted(numbers)
    assert len(set(numbers)) == len(numbers), "sequence numbers must be unique"

    # The final event carries the full response object.
    final = events[-1]["response"]
    assert final["output"][0]["content"][0]["text"] is not None


# --- error messages must name the field the CLIENT sent ---------------------
#
# ``_validate_generation_bounds`` is shared by /generate, /v1/completions,
# /v1/chat/completions and /v1/responses, and it used to hard-code
# "max_new_tokens" — an INTERNAL name. A Responses client that sent
# ``max_output_tokens`` was told about a parameter it had never written, which
# is the same defect class as the dropped ``max_completion_tokens``: the error
# points at the wrong field and the user cannot act on it.


@pytest.mark.parametrize(
    ("path", "payload", "expected_field"),
    [
        (
            "/v1/responses",
            {"model": "tiny", "input": "abc", "max_output_tokens": 200},
            "max_output_tokens",
        ),
        (
            "/v1/chat/completions",
            {"model": "tiny", "messages": [{"role": "user", "content": "abc"}], "max_tokens": 200},
            "max_tokens",
        ),
        (
            "/v1/completions",
            {"model": "tiny", "prompt": "abc", "max_tokens": 200},
            "max_tokens",
        ),
        (
            "/generate",
            {"prompt": "abc", "max_new_tokens": 200},
            "max_new_tokens",
        ),
    ],
)
def test_an_over_budget_error_names_the_clients_own_field(client, path, payload, expected_field):
    """Each endpoint's 400 must use that endpoint's parameter name.

    ``/generate`` legitimately calls it ``max_new_tokens``; the OpenAI-shaped
    endpoints do not, and telling those clients about an internal name sends them
    looking for a field that does not exist in their request.
    """
    response = client.post(path, json=payload)
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]

    assert expected_field in message, (
        f"{path} reported {message!r}; it should name {expected_field!r}, the field the client sent"
    )
    # The internal name must not leak into an endpoint whose client cannot send it.
    if expected_field != "max_new_tokens":
        assert "max_new_tokens" not in message, (
            f"{path} leaked the internal field name into a client-facing message: {message!r}"
        )


def test_the_shared_validator_defaults_to_generate_endpoints_field_name():
    """The default keeps ``/generate`` working without every caller passing it.

    ``/generate`` is the endpoint whose request really does call the budget
    ``max_new_tokens``, so its behaviour must be the unchanged default rather
    than something that has to be opted into.
    """
    import inspect

    from llm.serving.routers.generate import _validate_generation_bounds

    parameters = inspect.signature(_validate_generation_bounds).parameters
    assert parameters["budget_field"].default == "max_new_tokens"
