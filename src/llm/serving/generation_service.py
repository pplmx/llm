"""Serving-side generation service backed by GenerationBackend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from llm.generation.backends import GenerationBackend, GenerationConfig
from llm.generation.registry import get_generation_backend
from llm.models.decoder import DecoderModel
from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.config import ServingConfig
from llm.serving.loader import load_model_and_tokenizer


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class ServingGenerationService:
    """Shared generation entry point for REST and chat APIs."""

    model: DecoderModel
    tokenizer: Any
    backend: GenerationBackend
    device: torch.device

    @classmethod
    def from_config(
        cls,
        config: ServingConfig,
        *,
        engine: ContinuousBatchingEngine | None = None,
    ) -> ServingGenerationService:
        device = _resolve_device(config.device)
        model, tokenizer = load_model_and_tokenizer(config)
        if engine is None:
            model.to(device)
            model.eval()
        else:
            model = engine.model
            device = engine.device
        backend = get_generation_backend(config.generation_backend, engine=engine)
        return cls(model=model, tokenizer=tokenizer, backend=backend, device=device)

    def _generation_config(
        self,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
    ) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            use_cache=True,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
    ) -> str:
        gen_config = self._generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
        )
        return self.backend.generate(self.model, self.tokenizer, prompt, gen_config)

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
    ):
        gen_config = self._generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
        )
        yield from self.backend.stream(self.model, self.tokenizer, prompt, gen_config)

    def batch_generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
    ) -> list[str]:
        gen_config = self._generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
        )
        return self.backend.batch_generate(self.model, self.tokenizer, prompts, gen_config)
