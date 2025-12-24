from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Generation request model."""

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
