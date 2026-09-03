"""Tests for the E2E pipeline utilities (``llm.utils.e2e``).

Focus on the empty-input guards: with zero samples (or a validation
split that truncates to zero) the helpers used to raise a bare
``ZeroDivisionError`` where they now fail fast with a clear message
(RIL ISS-331).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.utils.e2e import E2EConfig, evaluate_model, run_e2e_pipeline, train_model


def _tiny_model(vocab: int = 16) -> nn.Module:
    """A minimal model the helpers can forward over a 16-token vocab."""
    return nn.Sequential(
        nn.Embedding(vocab, 8),
        nn.Linear(8, vocab),
        nn.Softmax(dim=-1),
    )


def test_train_model_empty_inputs_raises_clear_error():
    """A zero-row input tensor must raise ValueError (not ZeroDivisionError)."""
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    empty = torch.zeros(0, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="empty"):
        train_model(model, opt, criterion, empty, empty, batch_size=4, vocab_size=16, epochs=1)


def test_evaluate_model_empty_inputs_raises_clear_error():
    """A zero-row validation tensor must raise ValueError (not ZeroDivisionError)."""
    model = _tiny_model()
    criterion = nn.CrossEntropyLoss()

    empty = torch.zeros(0, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(model, criterion, empty, empty, batch_size=4, vocab_size=16)


@pytest.mark.parametrize("num_samples", [0, 4])
def test_run_e2e_pipeline_rejects_too_few_samples(num_samples: int):
    """``num_samples < 5`` makes the validation split (``num_samples // 5``)
    empty — reject it up front instead of dying mid-pipeline on the divide."""
    with pytest.raises(ValueError, match="num_samples"):
        run_e2e_pipeline(E2EConfig(num_samples=num_samples), device=torch.device("cpu"))


def test_evaluate_model_returns_loss_for_nonempty():
    """Sanity: the non-empty path still returns (loss, perplexity)."""
    model = _tiny_model()
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randint(0, 16, (4, 4))
    loss, ppl = evaluate_model(model, criterion, inputs, inputs, batch_size=2, vocab_size=16)
    assert loss > 0
    assert ppl > 0
