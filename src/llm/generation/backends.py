"""Generation backend abstractions."""

from __future__ import annotations

import abc
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from llm.models.decoder import DecoderModel


@dataclass(frozen=True)
class GenerationConfig:
    """Shared generation hyperparameters across inference backends."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    use_cache: bool = True


class GenerationBackend(abc.ABC):
    """Backend protocol for text generation."""

    @abc.abstractmethod
    def stream(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> Generator[str]:
        pass

    def generate(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        chunks = list(self.stream(model, tokenizer, prompt, config))
        return prompt + "".join(chunks)


class EagerGenerationBackend(GenerationBackend):
    """Default in-process generation using the library stream_generate path."""

    def stream(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> Generator[str]:
        from llm.generation.eager import stream_generate

        yield from stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            use_cache=config.use_cache,
        )


def get_generation_backend(name: str = "eager") -> GenerationBackend:
    """Resolve a generation backend by name."""
    backends: dict[str, GenerationBackend] = {
        "eager": EagerGenerationBackend(),
    }
    if name not in backends:
        available = ", ".join(sorted(backends))
        raise ValueError(f"Unknown generation backend '{name}'. Available: {available}")
    return backends[name]
