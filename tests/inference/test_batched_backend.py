"""Tests for BatchedGenerationBackend."""

from unittest.mock import patch

import pytest
import torch

from llm.generation.backends import BatchedGenerationBackend, GenerationConfig
from llm.models.decoder import DecoderModel
from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.schemas import GenerationRequest
from tests.support.tokenizers import StubTokenizer


@pytest.mark.quick
def test_batched_backend_generate(tiny_model, device, stub_tokenizer):
    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        device=device,
        max_batch_size=2,
    )
    backend = BatchedGenerationBackend(engine)
    output = backend.generate(
        model=engine.model,
        tokenizer=stub_tokenizer,
        prompt="hi",
        config=GenerationConfig(max_new_tokens=2, temperature=0.0),
    )
    assert output.startswith("hi")


@pytest.mark.quick
def test_engine_stream_request_respects_max_new_tokens(tiny_model, device, stub_tokenizer):
    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        device=device,
        max_batch_size=2,
    )
    request = GenerationRequest(prompt="test", max_new_tokens=2)
    chunks = list(engine.stream_request(request))
    assert len(chunks) == 2


class _TinyVocabTokenizer:
    """Tokenizer whose ``decode`` rejects any id >= its ``vocab_size``.

    Simulates a padded model vocab (or a BPE/HF model served with a char
    tokenizer): the model may emit ids up to its own vocab, but only the
    first ``vocab_size`` are decodable — the ISS-125 regression shape.
    """

    vocab_size: int = 5
    pad_token_id: int = 0

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i >= self.vocab_size:
                raise KeyError(f"Token ID '{i}' not found in tokenizer vocabulary")
            out.append(chr(ord("a") + i))
        return "".join(out)


@pytest.mark.quick
def test_engine_stream_masks_undecodable_tail_vocab(tiny_model, device):
    """Batched engine must not sample a token the tokenizer cannot decode.

    Mirror of the eager/speculative ISS-125 regression (test_eager.py): the
    engine's ``_forward_and_sample`` used to mask only the pad token, so a
    padded-vocab model served with a smaller-vocab tokenizer sampled the tail
    id and ``_emit_tokens`` crashed with ``tokenizer.decode([id])`` ->
    KeyError mid-stream.
    """
    bs, seq, vocab = 1, 8, 100  # 100-wide model vocab, tokenizer only decodes 5
    logits = torch.full((bs, seq, vocab), -1.0)
    logits[0, :, 99] = 10.0  # greedy argmax lands on id 99 (>= tokenizer vocab)

    def _fixed_forward(*_args, **_kwargs):
        return logits, None

    engine = ContinuousBatchingEngine(
        model=tiny_model.to(device),
        tokenizer=_TinyVocabTokenizer(),
        device=device,
        max_batch_size=2,
    )
    with patch.object(engine.model, "forward", side_effect=_fixed_forward):
        chunks = list(engine.stream_request(GenerationRequest(prompt="p", max_new_tokens=2, temperature=0.0)))

    # Without the mask the first sampled id (99) would raise KeyError in
    # decode; with it the tail ids are -inf and the sampler picks a decodable
    # one, so the stream completes with text-only chunks.
    assert chunks, "batched stream must not crash on a padded-vocab model"
    for text in chunks:
        assert isinstance(text, str)


@pytest.mark.quick
def test_engine_rejects_prompt_exceeding_model_capacity(device):
    """The engine validates against the MODEL's positional capacity.

    A configured engine ``max_seq_len`` larger than the checkpoint's
    ``max_seq_len`` used to let a too-long request through ``add_request`` and
    crash in the positional-encoding range check mid-forward (round-73 serving
    deep-dive). The context bound must be ``min(engine window, model
    capacity)``.
    """
    model = DecoderModel(
        vocab_size=64,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,  # tiny positional capacity
        device=device,
    )
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=StubTokenizer(token_ids=[1, 2, 3]),  # 3-token prompts
        device=device,
        max_batch_size=2,
        max_seq_len=32,  # engine window is larger than the model's capacity
    )

    # 3 prompt tokens + 6 generated fit the window (32) but not the model (8).
    with pytest.raises(ValueError, match=r"model's context window is 8"):
        engine.add_request(GenerationRequest(prompt="x", max_new_tokens=6))
