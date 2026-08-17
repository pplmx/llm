import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Request-input bounds — defense-in-depth against client memory-exhaustion
# DoS (RIL ISS-171). The schema caps are a fast 422 fail BEFORE encoding;
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
    temperature: float = Field(1.0, ge=0.0, description="Controls randomness. 0 for Greedy Search.")
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
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature. 0 for greedy.")
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


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field("llm", description="Model ID (ignored, for compatibility).")
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGES,
        description=f"List of messages (1..{MAX_MESSAGES}).",
    )
    max_tokens: int = Field(50, ge=1, le=4096, description="Maximum tokens to generate.")
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


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    completion_tokens: int = Field(..., description="Number of tokens in the completion.")
    total_tokens: int = Field(..., description="Total tokens used.")


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


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "llm"
    choices: list[ChatCompletionChunkChoice] = Field(..., description="List of chunk choices.")
