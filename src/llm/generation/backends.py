"""Generation backend abstractions."""

from __future__ import annotations

import abc
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm.models.decoder import DecoderModel

if TYPE_CHECKING:
    from llm.serving.batch_engine import ContinuousBatchingEngine


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

    def batch_generate(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        return [self.generate(model, tokenizer, prompt, config) for prompt in prompts]


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

    def batch_generate(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        from llm.generation.eager import batch_generate

        return batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )


class BatchedGenerationBackend(GenerationBackend):
    """Generation via ContinuousBatchingEngine (iteration-level scheduling)."""

    def __init__(self, engine: ContinuousBatchingEngine):
        self.engine = engine

    def stream(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> Generator[str]:
        from llm.serving.schemas import GenerationRequest

        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )
        yield from self.engine.stream_request(request)

    def batch_generate(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        from llm.serving.schemas import GenerationRequest

        requests = [
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
            )
            for prompt in prompts
        ]
        return self.engine.batch_generate_requests(requests)


def get_generation_backend(
    name: str = "eager",
    *,
    engine: ContinuousBatchingEngine | None = None,
) -> GenerationBackend:
    """Resolve a generation backend by name."""
    if name == "eager":
        return EagerGenerationBackend()
    if name == "batched":
        if engine is None:
            raise ValueError("batched generation backend requires a ContinuousBatchingEngine instance")
        return BatchedGenerationBackend(engine)
    raise ValueError(f"Unknown generation backend '{name}'. Available: batched, eager")
