"""Tests for ServingGenerationService."""

import pytest

from llm.generation.backends import EagerGenerationBackend
from llm.serving.generation_service import ServingGenerationService


@pytest.mark.quick
def test_generation_service_generate(tiny_model, device, stub_tokenizer):
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    output = service.generate(prompt="hi", max_new_tokens=2, temperature=0.0)
    assert output.startswith("hi")
    assert len(output) > len("hi")


@pytest.mark.quick
def test_generation_service_stream(tiny_model, device, stub_tokenizer):
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    chunks = list(service.stream(prompt="hi", max_new_tokens=2, temperature=0.0))
    assert len(chunks) == 2


@pytest.mark.quick
def test_generation_service_rejects_max_new_tokens_above_context(tiny_model, device, stub_tokenizer):
    """max_new_tokens >= the model's max_seq_len leaves no room for a prompt.

    The eager backend used to slice the input to an *empty* tensor in that
    case and 500 on the forward; the service now rejects it up front with a
    clear ValueError so the API can surface a 4xx instead.
    """
    service = ServingGenerationService(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )
    max_seq = getattr(tiny_model, "max_seq_len", None)
    assert max_seq is not None, "tiny_model must expose max_seq_len"

    with pytest.raises(ValueError, match="max_new_tokens"):
        service.generate(prompt="hi", max_new_tokens=max_seq, temperature=0.0)


class TestTrueBatchPathMatchesPerPromptGeneration:
    """The padded, single-forward-pass batcher must equal per-prompt generation.

    Important distinction this pins down: the DEFAULT ``eager`` backend's
    ``batch_generate`` is a plain loop over ``generate``
    (``backends.py``), so comparing it to independent calls is tautological — it
    cannot fail. The genuine batching implementation is
    ``llm.generation.eager.batch_generate``, which pads every prompt to a common
    length and runs one forward pass; that is where a padding or KV-slot bug would
    actually appear.

    Greedy decoding on purpose: with sampling the two runs consume the RNG in a
    different order and legitimately differ.
    """

    @staticmethod
    def _model_and_tokenizer():
        import torch

        from llm.models.decoder import DecoderModel
        from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

        torch.manual_seed(0)
        model = DecoderModel(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
            intermediate_size=64,
        ).eval()
        return model, SimpleCharacterTokenizer(["hello world abc"])

    def test_mixed_length_prompts_match_per_prompt_generation(self):
        """Different lengths are padded together — the real risk case."""
        from llm.generation.eager import batch_generate, generate

        model, tokenizer = self._model_and_tokenizer()
        prompts = ["h", "hello", "hello wor"]

        batched = batch_generate(model=model, tokenizer=tokenizer, prompts=prompts, max_new_tokens=4, temperature=0.0)
        singles = [generate(model, tokenizer, prompt, max_new_tokens=4, temperature=0.0) for prompt in prompts]

        assert batched == singles, (
            f"true batching diverged from per-prompt generation: {batched} != {singles} — "
            "padding leaked into a prompt's attention context"
        )

    def test_same_length_prompts_match_per_prompt_generation(self):
        from llm.generation.eager import batch_generate, generate

        model, tokenizer = self._model_and_tokenizer()
        prompts = ["hello", "world", "abcba"]

        batched = batch_generate(model=model, tokenizer=tokenizer, prompts=prompts, max_new_tokens=4, temperature=0.0)
        singles = [generate(model, tokenizer, prompt, max_new_tokens=4, temperature=0.0) for prompt in prompts]

        assert batched == singles, f"batched {batched} != singles {singles}"
