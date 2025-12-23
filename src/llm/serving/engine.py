import threading
from collections.abc import Iterator

import torch

from llm.inference import generate, stream_generate
from llm.models.decoder import DecoderModel
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


class LLMEngine:
    """
    LLM 推理引擎.
    封装模型加载、状态管理和生成逻辑.
    """

    def __init__(self, config: ServingConfig | None = None) -> None:
        self.config = config or ServingConfig()
        self.model: DecoderModel | None = None
        self.tokenizer: SimpleCharacterTokenizer | None = None
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """加载模型和分词器."""
        print(f"Loading model with config: {self.config}")
        # TODO: 支持从 self.config.model_path 加载真实权重

        corpus = [
            "hello world",
            "this is a test",
            "the quick brown fox jumps over the lazy dog",
        ]
        self.tokenizer = SimpleCharacterTokenizer(corpus)
        self.model = DecoderModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
        )

        if self.config.device != "auto":
            device = self.config.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.model.to(device) # Keeping on CPU for simplicity in this demo environment

        self.model.eval()
        print("Model loaded successfully.")

    def unload_model(self) -> None:
        """卸载模型以释放资源."""
        self.model = None
        self.tokenizer = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        """非流式生成."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model explicitly not loaded")

        with self._lock:  # 简单并发控制
            return generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        """流式生成."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model explicitly not loaded")

        with self._lock:
            yield from stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
