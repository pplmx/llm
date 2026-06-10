"""Tests for shared generation sampling helpers."""

import torch

from llm.generation.sampling import apply_repetition_penalty, sample_next_token


def test_greedy_sampling():
    logits = torch.tensor([0.1, 2.0, 0.5])
    assert sample_next_token(logits, temperature=0.0) == 1


def test_repetition_penalty_changes_logits():
    logits = torch.tensor([1.0, 2.0, 3.0])
    adjusted = apply_repetition_penalty(logits, [1, 2], repetition_penalty=2.0)
    assert adjusted[1].item() != logits[1].item()
