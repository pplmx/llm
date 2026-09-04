from unittest.mock import patch

import pytest
import torch

from llm.training.tasks.sft_task import SFTTask


def test_sft_task_train_step(tiny_config, tiny_model, device):
    # Use fixtures for config and model
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    # Batch: 2 sequences, len 4 (on same device as model)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 4), device=device),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]], device=device),
        "attention_mask": torch.ones(2, 4, device=device),
    }

    loss, metrics = task.train_step(batch, tiny_model, criterion)

    assert not torch.isnan(loss)
    assert metrics["loss"] > 0
    assert metrics["ppl"] >= 1.0


def test_sft_task_train_step_nan_loss_returns_zero(tiny_config, tiny_model, device):
    """NaN loss from criterion triggers fallback to zero loss with requires_grad=True."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (1, 2), device=device),
        "labels": torch.randint(0, 99, (1, 2), device=device),
    }

    # Patch criterion.forward so the NaN guard is exercised without
    # needing extreme numeric inputs that crash CUDA assertion kernels.
    nan_loss = torch.tensor(float("nan"), device=device, requires_grad=True)
    with patch.object(criterion, "forward", return_value=nan_loss):
        loss, metrics = task.train_step(batch, tiny_model, criterion)

    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad
    assert metrics["loss"] == pytest.approx(0.0)


def test_sft_task_train_step_no_attention_mask(tiny_config, tiny_model, device):
    """train_step works without attention_mask in batch."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (2, 4), device=device),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]], device=device),
    }

    loss, metrics = task.train_step(batch, tiny_model, criterion)

    assert not torch.isnan(loss)
    assert metrics["loss"] > 0


def test_sft_task_validation_step(tiny_config, tiny_model, device):
    """validation_step computes loss and val metrics."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (2, 4), device=device),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]], device=device),
        "attention_mask": torch.ones(2, 4, device=device),
    }

    loss, metrics = task.validation_step(batch, tiny_model, criterion)

    assert not torch.isnan(loss)
    assert metrics["val_loss"] > 0
    assert metrics["val_ppl"] >= 1.0


def test_sft_task_validation_step_no_attention_mask(tiny_config, tiny_model, device):
    """validation_step works without attention_mask."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (2, 4), device=device),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]], device=device),
    }

    loss, metrics = task.validation_step(batch, tiny_model, criterion)

    assert not torch.isnan(loss)
    assert metrics["val_loss"] > 0


def test_sft_task_build_criterion(tiny_config):
    """build_criterion returns CrossEntropyLoss with ignore_index=-100."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.ignore_index == -100


def test_sft_task_is_pure_alias_of_language_modeling_task(tiny_config, tiny_model, device):
    """SFTTask must produce EXACTLY the parent's loss/metrics on the same
    dict batch (RIL ISS-339): the old copies threaded attention_mask into
    the model forward — a mask the flash_attn backend ignores and the PP
    scheduler drops — silently diverging by backend/strategy. With pure
    delegation the two classes are bit-for-bit identical.
    """
    from llm.training.tasks.lm_task import LanguageModelingTask

    # The alias contract is "no overrides": the resolved step methods ARE the
    # parent's objects, not copies.
    assert SFTTask.train_step is LanguageModelingTask.train_step
    assert SFTTask.validation_step is LanguageModelingTask.validation_step
    assert SFTTask.build_criterion is LanguageModelingTask.build_criterion

    sft_task = SFTTask(tiny_config, data_module=None)
    lm_task = LanguageModelingTask(tiny_config, data_module=None)
    criterion = sft_task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (2, 4), device=device),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]], device=device),
        "attention_mask": torch.ones(2, 4, device=device),
    }

    # eval mode disables dropout, so both forwards are deterministic and the
    # bit-for-bit comparison is meaningful (two train-mode forwards differ by
    # dropout noise — RIL ISS-174).
    tiny_model.eval()
    sft_loss, sft_metrics = sft_task.train_step(batch, tiny_model, criterion)
    lm_loss, lm_metrics = lm_task.train_step(batch, tiny_model, criterion)
    assert torch.equal(sft_loss, lm_loss), "SFTTask.train_step diverged from LanguageModelingTask"
    assert set(sft_metrics) == set(lm_metrics)
    assert sft_metrics["loss"] == lm_metrics["loss"]
    assert sft_metrics["ppl"] == lm_metrics["ppl"]

    sft_vloss, sft_vmetrics = sft_task.validation_step(batch, tiny_model, criterion)
    lm_vloss, lm_vmetrics = lm_task.validation_step(batch, tiny_model, criterion)
    assert torch.equal(sft_vloss, lm_vloss), "SFTTask.validation_step diverged from LanguageModelingTask"
    assert set(sft_vmetrics) == set(lm_vmetrics)


def test_sft_task_rejects_length_one_sequences(tiny_config, tiny_model, device):
    """A degenerate (length-1) row must fail loudly like the parent, not be
    silently swallowed as a 0.0 loss via the NaN guard (RIL ISS-339)."""
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    batch = {
        "input_ids": torch.randint(0, 100, (1, 1), device=device),
        "labels": torch.randint(0, 99, (1, 1), device=device),
    }
    with pytest.raises(ValueError, match="sequence length must be > 1"):
        task.train_step(batch, tiny_model, criterion)
