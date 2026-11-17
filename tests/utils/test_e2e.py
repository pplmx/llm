"""Tests for the E2E pipeline utilities (``llm.utils.e2e``).

Focus on the empty-input guards: with zero samples (or a validation
split that truncates to zero) the helpers used to raise a bare
``ZeroDivisionError`` where they now fail fast with a clear message.
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


class TestTrainModelActuallyTrains:
    """The non-empty path must reduce the loss — nothing asserted that.

    ``tests/utils/test_e2e.py`` covered only the empty-input guards and one
    ``evaluate_model`` sanity call, so ``train_model``'s whole loop ran with no
    test checking its result: the backward pass, the gradient clip, the
    per-epoch mean, or the scheduler step could have been wrong and every test
    still passed. Coverage showed it (61.5% of the module).

    These tests use a plain ``Embedding -> Linear`` head with
    ``CrossEntropyLoss`` — the standard pairing. The existing helper ends in
    ``Softmax``, which double-applies the normalisation; it is fine for a shape
    sanity check but cannot demonstrate learning.
    """

    @staticmethod
    def _learnable_model(vocab: int = 16) -> nn.Module:
        return nn.Sequential(nn.Embedding(vocab, 16), nn.Linear(16, vocab))

    def _fixed_batch(self, vocab: int = 16, samples: int = 32, seq: int = 4):
        torch.manual_seed(0)
        inputs = torch.randint(0, vocab, (samples, seq))
        # A deterministic target the model CAN fit: predict token (i+1) % vocab.
        labels = (inputs + 1) % vocab
        return inputs, labels

    def test_loss_decreases_over_epochs_on_a_fittable_batch(self):
        """The single property that makes ``train_model`` useful."""
        torch.manual_seed(0)
        vocab = 16
        model = self._learnable_model(vocab)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        inputs, labels = self._fixed_batch(vocab)

        losses = train_model(
            model,
            optimizer,
            criterion,
            inputs,
            labels,
            batch_size=8,
            vocab_size=vocab,
            epochs=30,
        )

        assert len(losses) == 30, f"expected one loss per epoch, got {len(losses)}"
        assert all(isinstance(v, float) for v in losses)
        assert losses[-1] < losses[0] * 0.7, (
            f"loss did not meaningfully fall on a fittable batch: first={losses[0]:.4f} last={losses[-1]:.4f}"
        )

    def test_each_epoch_loss_is_a_mean_over_batches_not_a_sum(self):
        """``epoch_loss / num_batches`` — a sum would grow by the batch count.

        With a fixed model and a raised LR the exact value is noisy, so this
        asserts the SCALE instead: the reported loss must stay in the same
        range as one forward's loss, not ``num_batches`` times larger.
        """
        torch.manual_seed(0)
        vocab = 16
        model = self._learnable_model(vocab)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # frozen: loss is constant
        inputs, labels = self._fixed_batch(vocab, samples=32)

        with torch.no_grad():
            # ``train_model`` flattens logits to [B*T, vocab] and labels to [B*T];
            # the reference value must use the same shape or CrossEntropyLoss
            # compares a 3-D logit against 2-D targets and raises.
            logits = model(inputs)
            single = criterion(logits.view(-1, vocab), labels.view(-1)).item()

        batch_size = 8  # -> 4 batches per epoch
        losses = train_model(model, optimizer, criterion, inputs, labels, batch_size, vocab, epochs=2)

        assert losses[0] == pytest.approx(single, rel=0.05), (
            f"epoch loss {losses[0]:.4f} is not the per-batch MEAN of {single:.4f}"
        )
        assert losses[0] < single * 4, "epoch loss looks like a SUM over batches"

    def test_scheduler_steps_once_per_epoch(self):
        """A scheduler must be advanced once per epoch, not per batch."""
        torch.manual_seed(0)
        vocab = 16
        model = self._learnable_model(vocab)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        inputs, labels = self._fixed_batch(vocab)

        lrs = []
        for _ in range(2):
            train_model(model, optimizer, criterion, inputs, labels, 8, vocab, epochs=1, scheduler=scheduler)
            lrs.append(optimizer.param_groups[0]["lr"])

        assert lrs[1] < lrs[0], f"scheduler did not advance across epochs: {lrs}"

    def test_no_scheduler_is_accepted(self):
        """``scheduler=None`` must be a valid call, not an AttributeError."""
        vocab = 8
        model = nn.Sequential(nn.Embedding(vocab, 8), nn.Linear(8, vocab))
        losses = train_model(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
            torch.randint(0, vocab, (8, 3)),
            torch.randint(0, vocab, (8, 3)),
            4,
            vocab,
            2,
            scheduler=None,
        )
        assert len(losses) == 2
