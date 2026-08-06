"""Tests for prompt truncation in ``eager.batch_generate``.

When a prompt exceeds ``max_seq_len - max_new_tokens``, the input must be
truncated **before** it is used to seed ``generated_ids``.  Otherwise the
repetition penalty (and any penalty helper) operates on token ids the model
never actually attended to — a correctness bug that silently distorts the
sampling distribution.

These tests verify that ``batch_generate``:

1. Truncates each prompt to the last ``max_seq_len - max_new_tokens`` tokens
   before initialising ``generated_ids``.
2. Keeps ``generated_ids`` consistent with the (possibly truncated) tensor
   the model forward pass receives.
3. Still produces correct output when no truncation is needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from llm.generation.eager import batch_generate
from llm.models.decoder import DecoderModel


class _LongPromptTokenizer:
    """Tokenizer whose ``encode`` returns a configurable number of token ids.

    Each id ``i`` decodes to ``chr(ord('a') + i % 26)`` so we can inspect the
    generated text without a real vocabulary.
    """

    pad_token_id: int = 0

    def __init__(self, prompt_ids: list[int]) -> None:
        self._prompt_ids = list(prompt_ids)

    def encode(self, text: str) -> list[int]:
        return list(self._prompt_ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


class _RecordingModel(DecoderModel):
    """DecoderModel wrapper that records every ``input_ids`` seen.

    This lets tests assert which tokens the model actually attended to,
    independent of the (random) logit values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self.seen_inputs.append(input_ids.detach().clone())
        return super().forward(*args, **kwargs)


@pytest.fixture
def recording_model(device):
    """Tiny model that records the input_ids of every forward call."""
    model = _RecordingModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        device=device,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Truncation correctness
# ---------------------------------------------------------------------------


def test_batch_generate_truncates_generated_ids_to_match_input(recording_model):
    """``generated_ids`` must be seeded from the truncated prompt, not the full one.

    Prompt has 20 tokens, ``max_seq_len=16``, ``max_new_tokens=4`` → truncate to
    last 12 tokens.  The model's prefill forward must receive exactly those 12
    tokens (plus padding to align the batch), and ``generated_ids`` started
    from the same 12 tokens.

    We verify by checking the prefill forward input: after left-padding the
    12-token prompt, the non-pad portion is the last 12 ids.
    """
    prompt_ids = list(range(1, 21))  # 20 tokens
    tokenizer = _LongPromptTokenizer(prompt_ids)

    batch_generate(
        model=recording_model,
        tokenizer=tokenizer,
        prompts=["long prompt"],
        max_new_tokens=4,
        temperature=0.0,  # greedy for determinism
    )

    # First forward is the prefill — it should have exactly 12 non-pad tokens.
    prefill_input = recording_model.seen_inputs[0]
    assert prefill_input.shape[0] == 1
    # Left-padded: the last 12 columns should be the truncated prompt ids.
    truncated_ids = prompt_ids[-12:]  # [9, 10, ..., 20]
    row = prefill_input[0]
    # Find where the actual tokens start (after padding zeros).
    non_pad = row[row != 0]
    assert non_pad.tolist() == truncated_ids


def test_batch_generate_no_truncation_when_prompt_fits(recording_model):
    """When the prompt fits within ``max_seq_len - max_new_tokens``, no
    truncation occurs and ``generated_ids`` starts with the full prompt."""
    prompt_ids = list(range(1, 11))  # 10 tokens, max_seq_len=16, max_new_tokens=4 → 14
    tokenizer = _LongPromptTokenizer(prompt_ids)

    batch_generate(
        model=recording_model,
        tokenizer=tokenizer,
        prompts=["short prompt"],
        max_new_tokens=4,
        temperature=0.0,
    )

    prefill_input = recording_model.seen_inputs[0]
    row = prefill_input[0]
    non_pad = row[row != 0]
    assert non_pad.tolist() == prompt_ids


def test_batch_generate_repetition_penalty_uses_truncated_context(recording_model):
    """With ``repetition_penalty != 1.0``, the penalty must be applied only to
    tokens the model actually attended to (the truncated prompt), not the full
    prompt.

    We verify indirectly: the prefill forward receives the truncated prompt,
    so if the model predicts a token that appears in the truncated prompt
    but not in the full-but-truncated part, the penalty correctly applies.
    """
    prompt_ids = list(range(1, 21))  # 20 tokens
    tokenizer = _LongPromptTokenizer(prompt_ids)

    # The model should see only the last 12 tokens [9..20] in the prefill.
    batch_generate(
        model=recording_model,
        tokenizer=tokenizer,
        prompts=["long prompt"],
        max_new_tokens=4,
        temperature=0.0,
        repetition_penalty=2.0,
    )

    prefill_input = recording_model.seen_inputs[0]
    row = prefill_input[0]
    non_pad = row[row != 0]
    # Only the last 12 tokens should be visible.
    assert non_pad.tolist() == prompt_ids[-12:]


# ---------------------------------------------------------------------------
# Multi-batch edge cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug-specific test: repetition penalty context must match truncated input
# ---------------------------------------------------------------------------


def test_batch_generate_repetition_penalty_context_matches_truncated_input(recording_model):
    """BUG: ``generated_ids`` is seeded from the full prompt, but the model
    only sees the truncated prompt in the prefill forward.

    This means ``apply_repetition_penalty`` receives token ids the model never
    attended to.  The fix must truncate ``generated_ids`` to match the
    truncated ``input_tensor``.

    We patch ``apply_repetition_penalty`` and assert that the ``token_ids``
    it receives are a subset of (truncatable to) the tokens the model
    actually saw — i.e. the last ``truncate_len`` prompt tokens plus any
    generated tokens.
    """

    prompt_ids = list(range(1, 21))  # 20 tokens
    tokenizer = _LongPromptTokenizer(prompt_ids)
    max_new_tokens = 4
    truncate_len = 16 - max_new_tokens  # 12

    with patch("llm.generation.eager.apply_repetition_penalty", wraps=_passthrough) as spy:
        batch_generate(
            model=recording_model,
            tokenizer=tokenizer,
            prompts=["long prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=2.0,
        )

    # Spy should have been called with token_ids that start from the truncated
    # prompt, NOT the full 20-token prompt.
    assert spy.call_count > 0, "apply_repetition_penalty should have been called"
    for call in spy.call_args_list:
        context_ids = call.args[1]  # (logits, token_ids)
        # The prompt portion of the context (first truncate_len tokens) must
        # only contain the truncated prompt tokens, NOT the truncated-away ones.
        # Generated tokens may coincidentally match a truncated-away id, so
        # we only check the input prefix.
        prompt_context = context_ids[:truncate_len]
        truncated_away = set(prompt_ids[: len(prompt_ids) - truncate_len])
        prompt_context_set = set(prompt_context)
        assert not (truncated_away & prompt_context_set), (
            f"apply_repetition_penalty received token_ids from the truncated-away "
            f"prompt portion: {truncated_away & prompt_context_set}. "
            f"Context: {context_ids}"
        )


def _passthrough(logits, _token_ids, _repetition_penalty):
    """Identity wrapper — return logits unchanged."""
    return logits


def test_batch_generate_different_length_prompts(recording_model):
    """Mixed-length prompts where only the long one is truncated."""
    # Use a tokenizer that returns different ids per prompt by encoding text.
    prompt_long_ids = list(range(1, 21))
    prompt_short_ids = list(range(1, 11))

    class _MixedTokenizer:
        pad_token_id = 0

        def encode(self, text):
            if text == "short":
                return list(prompt_short_ids)
            return list(prompt_long_ids)

        def decode(self, ids):
            return "".join(chr(ord("a") + i % 26) for i in ids)

    batch_generate(
        model=recording_model,
        tokenizer=_MixedTokenizer(),
        prompts=["short", "long"],
        max_new_tokens=4,
        temperature=0.0,
    )

    # Prefill forward: both rows should have their non-pad tokens truncated
    # to at most 12 (max_seq_len - max_new_tokens = 16 - 4 = 12).
    prefill_input = recording_model.seen_inputs[0]
    assert prefill_input.shape[0] == 2
    for i in range(2):
        row = prefill_input[i]
        non_pad = row[row != 0]
        assert len(non_pad) <= 12
