"""SimPO task tests: reference-free loss, finite, and directionally correct."""

import torch

from llm.training.tasks.simpo_task import SimPOTask


def _pref_batch(seq_len: int, chosen: tuple, rejected: tuple) -> dict:
    """Build a chosen/rejected preference batch with the DPO task's keys."""
    chosen_ids = torch.as_tensor([list(chosen)])
    rejected_ids = torch.as_tensor([list(rejected)])
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, 0] = -100
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "chosen_attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "rejected_attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }


def test_simpo_is_reference_free(tiny_config):
    task = SimPOTask(tiny_config, data_module=None)
    assert not hasattr(task, "ref_model")
    model = task.build_model()
    assert any(p.requires_grad for p in model.parameters())


def test_simpo_train_step_finite_and_backprops(tiny_config, device):
    task = SimPOTask(tiny_config, data_module=None)
    model = task.build_model().to(device).train()
    vocab = tiny_config.model.vocab_size
    chosen = [(i % vocab) + 1 for i in range(6)]
    rejected = [((i + 3) % vocab) + 1 for i in range(6)]
    batch = _pref_batch(6, tuple(chosen), tuple(rejected))
    batch = {k: v.to(device) for k, v in batch.items()}

    loss, metrics = task.train_step(batch, model, criterion=None)
    assert not torch.isnan(loss)
    assert 0.0 <= metrics["reward_acc"] <= 1.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "SimPO loss must produce gradients"


def test_simpo_prefers_chosen_after_training(tiny_config, device):
    task = SimPOTask(tiny_config, data_module=None)
    model = task.build_model().to(device).train()
    vocab = tiny_config.model.vocab_size
    chosen = tuple((i % vocab) + 1 for i in range(6))
    rejected = tuple(((i + 3) % vocab) + 1 for i in range(6))
    batch = _pref_batch(6, chosen, rejected)
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(120):
        optimizer.zero_grad()
        loss, metrics = task.train_step(batch, model, criterion=None)
        loss.backward()
        optimizer.step()

    assert metrics["reward_acc"] == 1.0, "SimPO failed to prefer the chosen response"
    assert metrics["reward_margin"] > 0.0
