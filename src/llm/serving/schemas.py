from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """生成请求模型."""

    prompt: str = Field(..., description="输入的提示文本.")
    max_new_tokens: int = Field(50, ge=1, le=4096, description="生成的最大 token 数量.")
    temperature: float = Field(1.0, ge=0.0, description="控制生成文本的随机性. 0 为 Greedy Search.")
    top_k: int | None = Field(None, ge=1, description="Top-k 采样参数. 如果为 None 则不使用.")
    top_p: float | None = Field(None, gt=0.0, lt=1.0, description="Nucleus Sampling (Top-p) 参数.")
    repetition_penalty: float = Field(1.0, ge=1.0, description="重复惩罚参数. 1.0 表示不惩罚.")
    stream: bool = Field(False, description="是否使用流式输出 (SSE).")


class GenerationResponse(BaseModel):
    """生成响应模型."""

    generated_text: str = Field(..., description="生成的文本.")
    token_count: int | None = Field(None, description="生成的 token 数量.")
