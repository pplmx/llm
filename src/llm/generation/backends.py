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
    frequency_penalty: float = 0.0
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
            frequency_penalty=config.frequency_penalty,
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
            frequency_penalty=config.frequency_penalty,
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
            frequency_penalty=config.frequency_penalty,
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
                frequency_penalty=config.frequency_penalty,
            )
            for prompt in prompts
        ]
        return self.engine.batch_generate_requests(requests)


class SpeculativeDecodingBackend(GenerationBackend):
    """Speculative decoding: small draft model proposes, large target verifies.

    Implements Leviathan et al. 2023 — the draft model speculates
    ``gamma`` tokens ahead; the target scores them in a single
    forward pass and accepts each with probability
    ``min(1, q_target / q_draft)``. On rejection, sample a
    correction token from ``(q_target - q_draft)+``. The output
    distribution exactly matches the target distribution under the
    same sampling parameters.

    The ``model`` argument to :meth:`stream` / :meth:`batch_generate`
    is **ignored** — the target and draft models are bound at
    construction time. ``tokenizer`` must be the shared tokenizer
    used by both models (same vocab, pad id, eos id).

    Args:
        target_model: The "expensive" model whose distribution is
            the canonical output distribution.
        draft_model: The "cheap" model used for speculation. Must
            share vocabulary with ``target_model``.
        gamma: Number of speculative tokens per round (default 5).
            Typical values: 4–8.
    """

    def __init__(
        self,
        target_model: DecoderModel,
        draft_model: DecoderModel,
        *,
        gamma: int = 5,
    ) -> None:
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {gamma}")
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    def stream(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompt: str,
        config: GenerationConfig,
    ) -> Generator[str]:
        from llm.generation.speculative import speculative_generate

        yield from speculative_generate(
            target=self.target_model,
            draft=self.draft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            gamma=self.gamma,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            frequency_penalty=config.frequency_penalty,
        )

    def batch_generate(
        self,
        model: DecoderModel,
        tokenizer: Any,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        return [self.generate(model, tokenizer, prompt, config) for prompt in prompts]
