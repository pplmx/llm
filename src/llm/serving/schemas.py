import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Request-input bounds — defense-in-depth against client memory-exhaustion
# DoS. The schema caps are a fast 422 fail BEFORE encoding;
# the transport-level body-size middleware (serving/middleware.py) is the
# hard cap on the whole request, so every layer is bounded.
MAX_PROMPT_CHARS = 256_000  # ~64K tokens of prompt text — generous but bounded
MAX_BATCH_PROMPTS = 32
MAX_MESSAGES = 128
MAX_STOP_STRINGS = 4
MAX_STOP_LEN = 512  # per stop string


def _validate_stop(value: str | list[str] | None) -> str | list[str] | None:
    """Enforce the documented ``stop`` contract: a single string OR a list of
    up to 4 strings.

    ``Field(max_length=4)`` on a ``str | list[str]`` union applies the limit
    to *both* branches, so a legitimate stop sequence like ``"\\n\\n###"`` or
    ``"Human:"`` (longer than 4 characters) was 422-rejected even though the
    documented contract only caps the *list cardinality*.

    Also bounds each stop string's length (``MAX_STOP_LEN``) so an unbounded
    per-string stop cannot contribute to request-body unboundedness.
    """
    if value is None:
        return value
    if isinstance(value, str):
        if len(value) > MAX_STOP_LEN:
            raise ValueError(f"stop string may be at most {MAX_STOP_LEN} characters")
        return value
    if isinstance(value, list):
        if len(value) > MAX_STOP_STRINGS:
            raise ValueError(f"stop list may contain at most {MAX_STOP_STRINGS} strings")
        for s in value:
            if not isinstance(s, str) or len(s) > MAX_STOP_LEN:
                raise ValueError(f"each stop string must be a str of at most {MAX_STOP_LEN} characters")
        return value
    raise TypeError(f"stop must be a string or a list of strings, got {type(value).__name__}")


class RequestState(Enum):
    """Internal state of a request in the scheduler."""

    WAITING = auto()
    RUNNING = auto()
    PENDING = auto()  # Waiting for preemption to complete
    FINISHED = auto()


@dataclass
class Sequence:
    """
    Internal representation of a sequence state for the engine.
    """

    request_id: str
    prompt: str
    input_ids: list[int]

    status: RequestState = RequestState.WAITING
    generated_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    stop: str | list[str] | None = None

    def __post_init__(self):
        self._prompt_len = len(self.input_ids)

    @property
    def total_len(self) -> int:
        return self._prompt_len + len(self.generated_ids)

    def is_finished(self) -> bool:
        return self.status == RequestState.FINISHED

    def append_token_id(self, token_id: int):
        self.generated_ids.append(token_id)


class GenerationRequest(BaseModel):
    """Generation request model."""

    request_id: str | None = Field(None, description="Client-provided request ID.")
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
        description=(
            f"Input prompt text (1..{MAX_PROMPT_CHARS} characters). An empty prompt is "
            "rejected: the eager backend cannot decode a zero-length context."
        ),
    )
    max_new_tokens: int = Field(50, ge=1, le=4096, description="Maximum number of tokens to generate.")
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness. 0 for Greedy Search. Capped at 2.0 like /v1/chat/completions.",
    )
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter. None to disable.")
    top_p: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling (top-p) parameter. 1.0 means no truncation (OpenAI default).",
    )
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty. 1.0 means no penalty.")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="OpenAI-compatible per-frequency penalty.")
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="OpenAI-compatible per-presence penalty.",
    )
    logit_bias: dict[str, float] | None = Field(
        None,
        description="OpenAI-compatible per-token additive bias. JSON keys are token ids (as strings); values are added to the affected logits before sampling.",
    )
    stream: bool = Field(False, description="Whether to use streaming output (SSE).")
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequence(s). Generation halts the moment the generated text contains any of these as a suffix; the stop string itself is excluded from the response. Accepts a single string or a list of up to 4 strings. None means no stop.",
    )

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v):
        return _validate_stop(v)


class GenerationResponse(BaseModel):
    """Generation response model."""

    generated_text: str = Field(..., description="Generated text.")
    token_count: int | None = Field(None, description="Number of generated tokens.")


class BatchGenerationRequest(BaseModel):
    """Batch generation request model."""

    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_PROMPTS,
        description=f"List of input prompts (1..{MAX_BATCH_PROMPTS} entries, each at most {MAX_PROMPT_CHARS} chars).",
    )

    @field_validator("prompts")
    @classmethod
    def _check_prompts(cls, v):
        for s in v:
            if not s:
                raise ValueError("each prompt must be a non-empty string")
            if len(s) > MAX_PROMPT_CHARS:
                raise ValueError(f"each prompt may be at most {MAX_PROMPT_CHARS} characters")
        return v

    max_new_tokens: int = Field(50, ge=1, le=4096, description="Maximum tokens to generate per prompt.")
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 for greedy. Capped at 2.0 like /generate.",
    )
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter.")
    top_p: float | None = Field(
        None, gt=0.0, le=1.0, description="Nucleus sampling parameter. 1.0 means no truncation."
    )
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty.")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="OpenAI-compatible per-frequency penalty.")
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="OpenAI-compatible per-presence penalty.",
    )
    logit_bias: dict[str, float] | None = Field(
        None,
        description="Per-token additive logit bias. Keys are token ids (as strings); values are added to the affected logits before sampling.",
    )
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequence(s). Generation halts the moment the generated text contains any of these as a suffix; the stop string itself is excluded from the response. Accepts a single string or a list of up to 4 strings. None means no stop.",
    )

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v):
        return _validate_stop(v)


class BatchGenerationResponse(BaseModel):
    """Batch generation response model."""

    results: list[GenerationResponse] = Field(..., description="List of generation results.")


# OpenAI Chat Completions API Compatible Schemas


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message author.")
    content: str = Field(
        ...,
        max_length=MAX_PROMPT_CHARS,
        description=f"Content of the message (max {MAX_PROMPT_CHARS} characters).",
    )


class ChatJSONSchemaSpec(BaseModel):
    """The inner ``json_schema`` object of a chat ``response_format``.

    Note the shape difference from the Responses API: chat NESTS these under
    ``json_schema``, while ``text.format`` puts ``name``/``schema`` at the top
    level. Both are OpenAI's own spellings; a shared model would have to pretend
    one of them is wrong.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, max_length=64, description="Name of the response format.")
    schema_: dict[str, Any] = Field(..., alias="schema", description="JSON Schema the reply must match.")
    description: str | None = Field(None, description="Guidance for the model.")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ChatResponseFormatJSONSchema(BaseModel):
    """``response_format`` in schema mode."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["json_schema"] = "json_schema"
    json_schema: ChatJSONSchemaSpec = Field(..., description="The schema specification.")


class ChatResponseFormatJSONObject(BaseModel):
    """``response_format`` in plain-JSON mode."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["json_object"] = "json_object"


class ChatResponseFormatText(BaseModel):
    """``response_format`` in plain-text mode (the default)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = "text"


class ChatCompletionStreamOptions(BaseModel):
    """``stream_options`` — which extras to include on a streamed response."""

    model_config = ConfigDict(extra="ignore")

    include_usage: bool = Field(
        False,
        description=(
            "When true, an extra chunk carrying token usage follows the chunk with "
            "finish_reason; its choices list is empty and it precedes [DONE]."
        ),
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    # ``extra="ignore"`` is pydantic's default and is what this model used
    # implicitly: ANY unrecognised OpenAI field was accepted and dropped. For a
    # compatibility surface that is the worst option — a client sending
    # ``n=3`` / ``seed=42`` / ``stream_options={"include_usage": true}`` got a
    # 200 with silently different behaviour. The common unsupported knobs are
    # now DECLARED below so they can be refused with a clear message; fields we
    # have never heard of still pass through (clients send plenty of harmless
    # extras).
    model_config = ConfigDict(extra="ignore")

    model: str = Field("llm", description="Model ID (ignored, for compatibility).")
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGES,
        description=f"List of messages (1..{MAX_MESSAGES}).",
    )
    max_tokens: int = Field(
        50,
        ge=1,
        le=4096,
        description="Maximum tokens to generate (the deprecated OpenAI spelling; see max_completion_tokens).",
    )
    # OpenAI deprecated ``max_tokens`` in favour of ``max_completion_tokens``: the
    # SDK says so verbatim — "This value is now deprecated in favor of
    # max_completion_tokens". Modern clients send the NEW name only; measured,
    # ``langchain_openai.ChatOpenAI(max_tokens=8)`` puts
    # ``max_completion_tokens: 8`` on the wire. With just ``max_tokens`` declared
    # and ``extra="ignore"``, that field was dropped and the request ran with the
    # default 50 — a client asking for 8 tokens got 50, or on a short context a
    # 400 naming a max_new_tokens it never sent. Accept both spellings.
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        le=4096,
        description="Maximum tokens to generate. Preferred OpenAI spelling; overrides max_tokens when set.",
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float | None = Field(
        None, gt=0.0, le=1.0, description="Nucleus sampling parameter. 1.0 means no truncation."
    )
    stream: bool = Field(False, description="Whether to stream responses.")
    stop: list[str] | str | None = Field(
        None,
        description="Stop sequence(s). Generation halts the moment the generated text contains any of these as a suffix; the stop string itself is excluded from the response. Accepts a single string or a list of up to 4 strings. None means no stop.",
    )

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v):
        return _validate_stop(v)

    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="OpenAI-compatible per-presence penalty.",
    )
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="OpenAI-compatible per-frequency penalty.")
    logit_bias: dict[str, float] | None = Field(
        None,
        description="OpenAI-compatible per-token additive bias. JSON keys are token ids (as strings); values are added to the affected logits before sampling.",
    )

    # --- Declared-but-unsupported OpenAI knobs -----------------------------
    # Each defaults to the value that means "not requested", and a validator
    # rejects any other value. Without this they were silently dropped (see the
    # ``model_config`` note above).
    n: int | None = Field(None, description="UNSUPPORTED: multiple completions per request.")
    seed: int | None = Field(None, description="UNSUPPORTED: sampling reproducibility.")
    response_format: ChatResponseFormatText | ChatResponseFormatJSONObject | ChatResponseFormatJSONSchema | None = (
        Field(
            None,
            discriminator="type",
            description=(
                "Output format: text (default), json_object, or json_schema. The logits are masked by a "
                "JSON grammar when either of the last two is requested."
            ),
        )
    )
    stream_options: ChatCompletionStreamOptions | None = Field(
        None, description="Which extras to include on a streamed response."
    )
    tools: list[Any] | None = Field(None, description="UNSUPPORTED: tool/function calling.")

    @model_validator(mode="after")
    def _prefer_completion_tokens(self) -> ChatCompletionRequest:
        """Fold the two max-token spellings into ``max_tokens``.

        ``max_completion_tokens`` wins when both are present: it is the field
        OpenAI still supports, so a client that sets it means it. Downstream code
        keeps reading ``max_tokens`` alone, leaving exactly one place to get right.
        """
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _reject_unsupported_knobs(self) -> ChatCompletionRequest:
        """Refuse knobs this server cannot honour, instead of ignoring them.

        Silently accepting them gives a 200 whose behaviour differs from what
        the client asked for — the client has no way to notice. Answering with a
        4xx that names the field is the honest contract.
        """
        unsupported = {
            "n": self.n,
            "seed": self.seed,
            "tools": self.tools,
        }
        requested = [name for name, value in unsupported.items() if value is not None and value != 1]
        if requested:
            raise ValueError(
                f"unsupported chat-completion parameter(s): {', '.join(sorted(requested))}. "
                "This server generates a single completion and does not implement structured "
                "output, tool calling, seeding, or streaming usage reporting; remove the "
                "parameter instead of having it silently ignored."
            )
        return self


# OpenAI legacy Completions API (`POST /v1/completions`).
#
# This is the pre-chat OpenAI surface. It is still what a lot of tooling
# speaks: the official SDK exposes `client.completions.create`, and
# LangChain's `OpenAI` (as opposed to `ChatOpenAI`) / text-davinci-era
# prompt templates, plus a long tail of evaluation harnesses, call it
# directly. Without the route those clients got a bare 404 even though the
# server can generate text perfectly well.
#
# OpenAI accepts either a single `prompt` string or a list of them. The list
# form is supported natively: the router emits one `CompletionChoice` per
# prompt, in order, matching OpenAI's contract rather than joining the
# prompts into one request.


class CompletionRequest(BaseModel):
    """OpenAI-compatible legacy completions request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field("llm", description="Model ID (ignored, for compatibility).")
    prompt: str | list[str] = Field(
        ...,
        description=(
            "Prompt(s) to complete. A single string yields one choice; a list of up to "
            f"{MAX_BATCH_PROMPTS} strings yields one choice per prompt, in order."
        ),
    )
    max_tokens: int = Field(50, ge=1, le=4096, description="Maximum tokens to generate per prompt.")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float | None = Field(
        None, gt=0.0, le=1.0, description="Nucleus sampling parameter. 1.0 means no truncation."
    )
    stream: bool = Field(False, description="Whether to stream responses as server-sent events.")
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequence(s). Accepts a single string or a list of up to 4 strings.",
    )
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="OpenAI-compatible per-presence penalty.")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="OpenAI-compatible per-frequency penalty.")
    logit_bias: dict[str, float] | None = Field(
        None,
        description="OpenAI-compatible per-token additive bias. JSON keys are token ids (as strings).",
    )

    # Declared-but-unsupported knobs, mirroring ``ChatCompletionRequest``:
    # silently ignoring them yields a 200 whose behaviour differs from what the
    # client asked for.
    n: int | None = Field(None, description="UNSUPPORTED: multiple completions per request.")
    seed: int | None = Field(None, description="UNSUPPORTED: sampling reproducibility.")
    stream_options: ChatCompletionStreamOptions | None = Field(
        None, description="Which extras to include on a streamed response."
    )

    @field_validator("prompt")
    @classmethod
    def _check_prompt(cls, v: str | list[str]) -> str | list[str]:
        prompts = [v] if isinstance(v, str) else v
        if not prompts:
            raise ValueError("prompt must be a non-empty string or a non-empty list of strings")
        if len(prompts) > MAX_BATCH_PROMPTS:
            raise ValueError(f"prompt list may contain at most {MAX_BATCH_PROMPTS} entries")
        for s in prompts:
            if not isinstance(s, str):
                raise TypeError(f"each prompt must be a str, got {type(s).__name__}")
            if not s:
                raise ValueError("each prompt must be a non-empty string")
            if len(s) > MAX_PROMPT_CHARS:
                raise ValueError(f"each prompt may be at most {MAX_PROMPT_CHARS} characters")
        return v

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v):
        return _validate_stop(v)

    @model_validator(mode="after")
    def _reject_unsupported_knobs(self) -> CompletionRequest:
        """Refuse knobs this server cannot honour, instead of ignoring them.

        Same contract as :meth:`ChatCompletionRequest._reject_unsupported_knobs`.
        ``n=1`` is the OpenAI default and means "one completion", which is
        exactly what this server does, so it is accepted.
        """
        unsupported = {
            "n": self.n,
            "seed": self.seed,
        }
        requested = [name for name, value in unsupported.items() if value is not None and value != 1]
        if requested:
            raise ValueError(
                f"unsupported completion parameter(s): {', '.join(sorted(requested))}. "
                "This server generates a single completion per prompt and does not implement "
                "seeding or streaming usage reporting; remove the parameter instead of having "
                "it silently ignored."
            )
        return self


class CompletionChoice(BaseModel):
    """A single legacy-completions choice."""

    text: str = Field(..., description="Generated text (prompt is NOT echoed).")
    index: int = Field(..., description="Index of this choice.")
    finish_reason: Literal["stop", "length"] | None = Field(None, description="Reason for stopping.")
    logprobs: None = Field(None, description="Unsupported; always null.")


class CompletionUsage(BaseModel):
    """Token usage statistics for legacy completions."""

    prompt_tokens: int = Field(..., description="Number of tokens across all prompts.")
    completion_tokens: int = Field(..., description="Number of generated tokens across all choices.")
    total_tokens: int = Field(..., description="Total tokens used.")


class CompletionResponse(BaseModel):
    """OpenAI-compatible legacy completions response."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field("llm", description="Model used.")
    choices: list[CompletionChoice] = Field(..., description="One choice per prompt, in order.")
    usage: CompletionUsage = Field(..., description="Token usage.")


class CompletionChunkChoice(BaseModel):
    """A choice in a streaming legacy-completions chunk."""

    text: str = Field("", description="Delta text for this chunk.")
    index: int = Field(0, description="Index of the prompt this delta belongs to.")
    finish_reason: Literal["stop", "length"] | None = None
    logprobs: None = None


class CompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk for legacy completions."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llm"
    choices: list[CompletionChunkChoice] = Field(..., description="Chunk choices.")
    usage: ChatCompletionUsage | None = Field(
        None, description="Token usage; present on the final chunk only when requested."
    )


class ChatCompletionChoiceMessage(BaseModel):
    """Message in a chat completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str = Field(..., description="Generated content.")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(..., description="Index of this choice.")
    message: ChatCompletionChoiceMessage = Field(..., description="Generated message.")
    finish_reason: Literal["stop", "length"] | None = Field(None, description="Reason for stopping.")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field("llm", description="Model used.")
    choices: list[ChatCompletionChoice] = Field(..., description="List of choices.")
    usage: ChatCompletionUsage = Field(..., description="Token usage.")


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta = Field(..., description="Delta content.")
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    completion_tokens: int = Field(..., description="Number of tokens in the completion.")
    total_tokens: int = Field(..., description="Total tokens used.")


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llm"
    choices: list[ChatCompletionChunkChoice] = Field(..., description="List of chunk choices.")
    # Populated ONLY on the extra trailing chunk emitted when the request set
    # ``stream_options.include_usage``. OpenAI's contract: that chunk carries an
    # EMPTY ``choices`` list and the totals, and it arrives after the chunk with
    # ``finish_reason`` and before ``[DONE]``.
    usage: ChatCompletionUsage | None = Field(
        None, description="Token usage; present on the final chunk only when requested."
    )


# --- OpenAI Responses API (``POST /v1/responses``) -------------------------
#
# OpenAI's current recommendation: the Responses API supersedes chat
# completions for new integrations. The shape differs in ways that matter, so
# this is modelled on the SDK's own types rather than reusing the chat models:
#
#   * ``input`` (string or message list) replaces ``messages``;
#   * ``instructions`` replaces the ``system`` role message;
#   * ``max_output_tokens`` replaces ``max_tokens`` / ``max_completion_tokens``;
#   * the assistant text comes back in ``output`` as a list of items, each with
#     a ``content`` list — not in ``choices[0].message.content``;
#   * the response carries ``status`` and, when truncated, ``incomplete_details``
#     with a ``reason`` of ``"max_output_tokens"`` instead of ``finish_reason``.
#
# Reference: ``openai.types.responses`` in the installed SDK.


class ResponsesInputMessage(BaseModel):
    """One input item for the Responses API."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant"] = Field(..., description="Role of the message author.")
    content: str = Field(
        ...,
        max_length=MAX_PROMPT_CHARS,
        description=f"Text content of the item (max {MAX_PROMPT_CHARS} characters).",
    )


class ResponsesUsage(BaseModel):
    """Token usage, in the Responses API's field names."""

    input_tokens: int = Field(..., description="Tokens in the prompt.")
    input_tokens_details: dict[str, int] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens: int = Field(..., description="Tokens generated.")
    output_tokens_details: dict[str, int] = Field(default_factory=lambda: {"reasoning_tokens": 0})
    total_tokens: int = Field(..., description="Sum of input and output tokens.")


class ResponsesOutputText(BaseModel):
    """A text part inside an output message item."""

    type: Literal["output_text"] = "output_text"
    text: str = Field(..., description="The generated text.")
    annotations: list[Any] = Field(default_factory=list, description="Always empty (no file/web search).")
    logprobs: list[Any] = Field(default_factory=list, description="Always empty (not implemented).")


class ResponsesOutputMessage(BaseModel):
    """One message item in ``response.output``."""

    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    status: Literal["completed", "incomplete"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[ResponsesOutputText] = Field(..., description="The message's content parts.")


class ResponsesIncompleteDetails(BaseModel):
    """Why a response stopped before finishing."""

    reason: Literal["max_output_tokens", "content_filter"] = Field(
        ..., description="``max_output_tokens`` when the token budget ran out."
    )


class ResponsesJSONSchemaFormat(BaseModel):
    """``text.format`` for structured output: a JSON Schema the reply must match."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["json_schema"] = "json_schema"
    name: str = Field(..., min_length=1, description="Schema name.")
    schema_: dict[str, Any] = Field(
        ...,
        alias="schema",
        description="The JSON Schema the output must validate against.",
    )
    description: str | None = Field(None, description="Optional guidance for the model.")
    strict: bool | None = Field(None, description="Accepted for compatibility; always enforced.")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ResponsesTextFormatJSONObject(BaseModel):
    """``text.format`` for plain JSON mode (valid JSON, no schema)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["json_object"] = "json_object"


class ResponsesTextFormatText(BaseModel):
    """``text.format`` for the default plain-text mode."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = "text"


class ResponsesTextConfig(BaseModel):
    """``text`` on a Responses request."""

    model_config = ConfigDict(extra="ignore")

    format: ResponsesTextFormatText | ResponsesTextFormatJSONObject | ResponsesJSONSchemaFormat = Field(
        default_factory=ResponsesTextFormatText,
        discriminator="type",
        description="Output format: text (default), json_object, or json_schema.",
    )


class ResponsesRequest(BaseModel):
    """OpenAI-compatible Responses API request."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field("llm", description="Model ID (ignored, for compatibility).")
    input: str | list[ResponsesInputMessage] = Field(
        ...,
        description=(
            "The prompt: a plain string, or a list of message items. A string is treated as a single user message."
        ),
    )
    instructions: str | None = Field(
        None,
        max_length=MAX_PROMPT_CHARS,
        description="System-level instructions; equivalent to a leading system message.",
    )
    max_output_tokens: int | None = Field(
        None,
        ge=1,
        le=4096,
        description="Upper bound on generated tokens. None means the server default.",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float | None = Field(None, gt=0.0, le=1.0, description="Nucleus sampling parameter.")
    stream: bool = Field(False, description="Whether to stream the response as SSE events.")

    # Declared-but-unsupported knobs. Silently dropping them would return a 200
    # whose behaviour differs from what the client asked for, so they are
    # refused by name — the same contract as the chat/completions schemas.
    tools: list[Any] | None = Field(None, description="UNSUPPORTED: tool calling.")
    tool_choice: Any | None = Field(None, description="UNSUPPORTED: tool choice.")
    previous_response_id: str | None = Field(None, description="UNSUPPORTED: response chaining.")
    store: bool | None = Field(None, description="UNSUPPORTED: server-side response storage.")
    background: bool | None = Field(None, description="UNSUPPORTED: background execution.")
    include: list[str] | None = Field(None, description="UNSUPPORTED: extra output fields.")
    reasoning: Any | None = Field(None, description="UNSUPPORTED: reasoning configuration.")
    text: ResponsesTextConfig | None = Field(
        None,
        description=(
            "Output format. Set ``text.format`` to ``{type: json_schema, name, schema}`` for schema-constrained JSON."
        ),
    )
    truncation: str | None = Field(None, description="UNSUPPORTED: context truncation strategies.")

    @field_validator("input")
    @classmethod
    def _check_input(cls, v: str | list[ResponsesInputMessage]) -> str | list[ResponsesInputMessage]:
        if isinstance(v, str):
            if not v:
                raise ValueError("input must be a non-empty string")
            if len(v) > MAX_PROMPT_CHARS:
                raise ValueError(f"input string may be at most {MAX_PROMPT_CHARS} characters")
            return v
        if not v:
            raise ValueError("input must be a non-empty string or a non-empty list of messages")
        if len(v) > MAX_MESSAGES:
            raise ValueError(f"input may contain at most {MAX_MESSAGES} items")
        return v

    @model_validator(mode="after")
    def _reject_unsupported_knobs(self) -> ResponsesRequest:
        """Refuse knobs this server cannot honour, instead of ignoring them."""
        unsupported = {
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "previous_response_id": self.previous_response_id,
            "store": self.store,
            "background": self.background,
            "include": self.include,
            "reasoning": self.reasoning,
            "truncation": self.truncation,
        }
        requested = [name for name, value in unsupported.items() if value is not None and value is not False]
        if requested:
            raise ValueError(
                f"unsupported Responses API parameter(s): {', '.join(sorted(requested))}. "
                "This server generates plain text and does not implement tool calling, response "
                "storage, background execution, reasoning controls, or structured output; remove "
                "the parameter instead of having it silently ignored."
            )
        return self


class ResponsesResponse(BaseModel):
    """OpenAI-compatible Responses API response object."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:16]}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llm"
    status: Literal["completed", "incomplete", "failed"] = "completed"
    output: list[ResponsesOutputMessage] = Field(..., description="The generated items.")
    output_text: str = Field(
        "",
        description=(
            "Convenience mirror of the generated text. The SDK exposes this as a computed "
            "property; the server includes it so that plain JSON clients (and ``jq``) can read "
            "the text without walking the output items."
        ),
    )
    usage: ResponsesUsage = Field(..., description="Token usage.")
    incomplete_details: ResponsesIncompleteDetails | None = Field(
        None, description="Present only when ``status`` is ``incomplete``."
    )
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    instructions: str | None = None
    error: Any | None = Field(None, description="Always null: failures are HTTP error responses.")
