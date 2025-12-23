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
        # 1. Load Tokenizer
        if self.config.tokenizer_path:
            print(f"Loading tokenizer from {self.config.tokenizer_path}...")
            try:
                # We need weights_only=False to load the pickled tokenizer object
                self.tokenizer = torch.load(self.config.tokenizer_path, weights_only=False)
                print("Tokenizer loaded successfully.")
            except Exception as e:
                print(f"Failed to load tokenizer from {self.config.tokenizer_path}: {e}")
                raise
        else:
            # Use a richer corpus for dummy tokenizer to support more characters
            import string

            corpus = [
                "hello world",
                "this is a test",
                "the quick brown fox jumps over the lazy dog",
                string.printable,
            ]
            self.tokenizer = SimpleCharacterTokenizer(corpus)

        # 2. Initialize Model
        self.model = DecoderModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
        )

        # 3. Load Model Weights
        if self.config.model_path:
            print(f"Loading model weights from {self.config.model_path}...")
            try:
                checkpoint = torch.load(self.config.model_path, map_location="cpu")

                # Handle both raw state_dict and checkpoint dict (from training engine)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                # Handle DDP prefix if present (e.g. "module.")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Failed to load model from {self.config.model_path}: {e}")
                raise

        # self.model.to(device) # Keeping on CPU for simplicity in this demo environment

        self.model.eval()

        if self.config.compile_model:
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

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
