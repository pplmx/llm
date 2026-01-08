import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel, Field


class RequestState(Enum):
    """Internal state of a request in the scheduler."""

    WAITING = auto()
    RUNNING = auto()
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
    prompt: str = Field(..., description="Input prompt text.")
    max_new_tokens: int = Field(50, ge=1, le=4096, description="Maximum number of tokens to generate.")
    temperature: float = Field(1.0, ge=0.0, description="Controls randomness. 0 for Greedy Search.")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter. None to disable.")
    top_p: float | None = Field(None, gt=0.0, lt=1.0, description="Nucleus sampling (top-p) parameter.")
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty. 1.0 means no penalty.")
    stream: bool = Field(False, description="Whether to use streaming output (SSE).")


class GenerationResponse(BaseModel):
    """Generation response model."""

    generated_text: str = Field(..., description="Generated text.")
    token_count: int | None = Field(None, description="Number of generated tokens.")


class BatchGenerationRequest(BaseModel):
    """Batch generation request model."""

    prompts: list[str] = Field(..., min_length=1, max_length=32, description="List of input prompts.")
    max_new_tokens: int = Field(50, ge=1, le=4096, description="Maximum tokens to generate per prompt.")
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature. 0 for greedy.")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter.")
    top_p: float | None = Field(None, gt=0.0, lt=1.0, description="Nucleus sampling parameter.")
    repetition_penalty: float = Field(1.0, ge=1.0, description="Repetition penalty.")


class BatchGenerationResponse(BaseModel):
    """Batch generation response model."""

    results: list[GenerationResponse] = Field(..., description="List of generation results.")


# OpenAI Chat Completions API Compatible Schemas


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message author.")
    content: str = Field(..., description="Content of the message.")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field("llm", description="Model ID (ignored, for compatibility).")
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of messages.")
    max_tokens: int = Field(50, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float | None = Field(None, gt=0.0, lt=1.0, description="Nucleus sampling parameter.")
    stream: bool = Field(False, description="Whether to stream responses.")
    stop: list[str] | str | None = Field(None, description="Stop sequences (not implemented).")
    presence_penalty: float = Field(0.0, description="Presence penalty (mapped to repetition_penalty).")
    frequency_penalty: float = Field(0.0, description="Frequency penalty (not implemented).")


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
