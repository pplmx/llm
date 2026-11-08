"""Contract tests for the OpenAI-compatible legacy completions route.

``POST /v1/completions`` is what the official SDK's
``client.completions.create`` and LangChain's non-chat ``OpenAI`` wrapper
call. Before this slice the route did not exist, so those clients got a bare
404 from a server that generates text perfectly well — the chat route was the
only OpenAI-shaped surface.

These tests pin the response SHAPE (``text_completion`` / ``choices[].text``
/ ``usage``), the parts that are easy to get wrong and impossible to notice
from a 200 alone:

* the completion must NOT echo the prompt (the eager backend's ``generate``
  returns ``prompt + completion``),
* a list prompt must yield one choice per prompt IN ORDER, not one
  concatenated request,
* ``finish_reason`` must distinguish budget exhaustion (``length``) from a
  natural stop (``stop``),
* ``usage`` counts must be tokenizer counts, and
* the unsupported OpenAI knobs must be refused rather than silently ignored.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client(monkeypatch):
    """TestClient with a mocked generation service (no real model loaded)."""
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    fake_service = MagicMock()
    fake_service.tokenizer.encode = lambda text: list(range(len(text)))
    fake_service.tokenizer.decode = lambda ids: "x"
    # The eager backend's ``generate`` returns ``prompt + completion``;
    # mirror that here so the route's prompt-echo strip is actually exercised.
    fake_service.generate.side_effect = lambda prompt, **kw: prompt + " completed"
    fake_service.batch_generate.side_effect = lambda prompts, **kw: [p + " completed" for p in prompts]
    fake_service.stream.return_value = iter(["a", "b"])

    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: fake_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: MagicMock()),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", fake_service)
        yield c


def _payload(**overrides):
    payload = {"model": "llm", "prompt": "hello", "max_tokens": 50}
    payload.update(overrides)
    return payload


# --- Response shape ---------------------------------------------------------


def test_completion_response_is_text_completion_shaped(client):
    """The top-level shape matches OpenAI's ``text_completion`` object."""
    response = client.post("/v1/completions", json=_payload())

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["id"].startswith("cmpl-")
    assert isinstance(data["created"], int)
    assert data["model"] == "llm"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["text"] == " completed"
    assert choice["logprobs"] is None


def test_completion_usage_is_reported(client):
    """``usage`` carries prompt / completion / total token counts."""
    data = client.post("/v1/completions", json=_payload()).json()

    usage = data["usage"]
    assert usage["prompt_tokens"] == len("hello")
    assert usage["completion_tokens"] == len(" completed")
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_completion_does_not_echo_the_prompt(client):
    """The completion is the generated suffix only.

    The eager backend's ``generate()`` returns ``prompt + completion`` (this
    route's ``/generate`` sibling strips the same echo). Reporting the raw
    string would put the prompt in ``choices[].text`` and count it as
    generated tokens — a wrong number that still looks like a valid 200.
    """
    data = client.post("/v1/completions", json=_payload(prompt="hello")).json()

    assert "hello" not in data["choices"][0]["text"]
    assert data["choices"][0]["text"] == " completed"


# --- List prompts -----------------------------------------------------------


def test_list_prompt_yields_one_choice_per_prompt_in_order(client):
    """A list prompt is a batch: one choice per prompt, in request order.

    Joining the prompts into one request would answer with a single choice
    whose text is a completion of the concatenation — the client has no way
    to detect that its per-prompt results are missing.
    """
    data = client.post("/v1/completions", json=_payload(prompt=["aaa", "bb"])).json()

    assert len(data["choices"]) == 2
    assert [c["index"] for c in data["choices"]] == [0, 1]
    assert [c["text"] for c in data["choices"]] == [" completed", " completed"]


def test_list_prompt_usage_sums_prompt_tokens(client):
    """``prompt_tokens`` covers every prompt in a batch, not just the first."""
    data = client.post("/v1/completions", json=_payload(prompt=["aaa", "bb"])).json()

    assert data["usage"]["prompt_tokens"] == len("aaa") + len("bb")


# --- finish_reason ----------------------------------------------------------


def test_finish_reason_length_when_budget_exhausted(client, monkeypatch):
    """A completion that consumed the full ``max_tokens`` budget is ``length``."""
    service = MagicMock()
    service.tokenizer.encode = lambda text: list(range(len(text)))
    service.tokenizer.decode = lambda ids: "x"
    service.generate.side_effect = lambda prompt, **kw: prompt + "abcd"
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", service)

    data = client.post("/v1/completions", json=_payload(max_tokens=4)).json()
    assert data["choices"][0]["finish_reason"] == "length"


def test_finish_reason_stop_when_natural_stop(client):
    """A short completion well under the budget keeps ``stop``."""
    data = client.post("/v1/completions", json=_payload(max_tokens=50)).json()
    assert data["choices"][0]["finish_reason"] == "stop"


# --- Streaming --------------------------------------------------------------


def test_completion_stream_emits_sse_frames_and_done(client):
    """Streaming emits ``data:`` frames terminated by ``data: [DONE]``."""
    with client.stream("POST", "/v1/completions", json=_payload(stream=True)) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = [line for line in response.iter_lines() if line and line.startswith("data: ")]

    assert chunks[-1] == "data: [DONE]"
    payloads = [json.loads(line[6:]) for line in chunks[:-1]]
    assert all(p["object"] == "text_completion" for p in payloads)
    # The generated tokens are streamed verbatim, in order.
    streamed = "".join(p["choices"][0]["text"] for p in payloads if p["choices"][0]["text"])
    assert streamed == "ab"
    # The last frame carries the finish reason.
    assert payloads[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


def test_completion_stream_rejects_list_prompts(client):
    """Streaming a batch is refused instead of emitting ambiguous indices."""
    response = client.post("/v1/completions", json=_payload(prompt=["a", "b"], stream=True))

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "list of prompts" in body["error"]["message"]


# --- Unsupported OpenAI knobs ----------------------------------------------


@pytest.mark.parametrize(("field", "value"), [("n", 3), ("seed", 42), ("stream_options", {"include_usage": True})])
def test_unsupported_completion_knobs_are_refused(client, field, value):
    """Declared-but-unsupported knobs must 422, not be silently dropped.

    Returning 200 while ignoring ``n=3`` gives the client a single completion
    it did not ask for and no signal that it was ignored.
    """
    response = client.post("/v1/completions", json=_payload(**{field: value}))

    assert response.status_code == 422, response.text
    assert field in response.text


def test_n_equals_one_is_accepted(client):
    """``n=1`` is the OpenAI default ("one completion") — it must not be refused."""
    response = client.post("/v1/completions", json=_payload(n=1))
    assert response.status_code == 200


# --- Validation -------------------------------------------------------------


@pytest.mark.parametrize("prompt", ["", []])
def test_empty_prompt_is_rejected(client, prompt):
    """An empty prompt (or empty list) is a 422, not a crash downstream."""
    response = client.post("/v1/completions", json=_payload(prompt=prompt))
    assert response.status_code == 422


def test_oversized_prompt_list_is_rejected(client):
    """The prompt list is bounded so one request cannot fan out unboundedly."""
    from llm.serving.schemas import MAX_BATCH_PROMPTS

    response = client.post("/v1/completions", json=_payload(prompt=["x"] * (MAX_BATCH_PROMPTS + 1)))
    assert response.status_code == 422


# --- Auth and error mapping -------------------------------------------------


def test_completions_route_requires_the_api_key(monkeypatch):
    """With ``api_key`` configured, the route is guarded like its siblings.

    A new OpenAI-shaped route that skipped the auth dependency would be an
    anonymous inference bypass next to guarded ``/generate`` and
    ``/v1/chat/completions``.
    """
    from llm.serving.api import config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    fake_service = MagicMock()
    fake_service.tokenizer.encode = lambda text: [1]
    fake_service.generate.return_value = "ok"
    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: fake_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: MagicMock()),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    original_key = config.api_key
    config.api_key = "test-key"
    try:
        with TestClient(app) as c:
            # No key -> refused.
            assert c.post("/v1/completions", json=_payload()).status_code == 403
            # Correct key -> served.
            c.headers["X-API-Key"] = "test-key"
            assert c.post("/v1/completions", json=_payload()).status_code == 200
    finally:
        config.api_key = original_key


def test_completions_maps_value_error_to_400(client, monkeypatch):
    """A client-caused backend ``ValueError`` is a 400, not a 500."""
    service = MagicMock()
    service.tokenizer.encode = lambda text: [1]
    service.generate.side_effect = ValueError("bad prompt")
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", service)

    response = client.post("/v1/completions", json=_payload())
    assert response.status_code == 400


def test_completions_maps_runtime_error_to_503(client, monkeypatch):
    """A backend ``RuntimeError`` is reported as "model unavailable" (503)."""
    service = MagicMock()
    service.tokenizer.encode = lambda text: [1]
    service.generate.side_effect = RuntimeError("model exploded")
    monkeypatch.setattr("llm.serving.routers.generate.generation_service", service)

    response = client.post("/v1/completions", json=_payload())
    assert response.status_code == 503
