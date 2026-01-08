from unittest.mock import MagicMock

import torch
import torch.nn as nn

from llm.training.core.config import Config, ModelConfig
from llm.training.tasks.dpo_task import DPOTask


class MockModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=1, num_heads=2)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        return self.head(x)


def test_dpo_task_init_and_build():
    config = Config()
    # Ensure build_model can run - it calls super().build_model() which creates DecoderModel
    # We can mock build_model or config.
    # To avoid creating heavy DecoderModel, testing logic we can subclass DPOTask and mock super().build_model?
    # Or just use tiny config.

    config.model = ModelConfig(hidden_size=16, num_layers=1, num_heads=2, intermediate_size=32, vocab_size=100)

    task = DPOTask(config, data_module=MagicMock())
    model = task.build_model()

    assert task.ref_model is not None
    assert task.ref_model is not model

    # Check freezing
    for p in task.ref_model.parameters():
        assert not p.requires_grad

    # Check policy model is trainable (at least has grad)
    # Embedding might be fixed if configured, but generally trainable.
    assert any(p.requires_grad for p in model.parameters())


def test_dpo_task_train_step():
    config = Config()
    task = DPOTask(config, data_module=MagicMock())

    # Mock models manually
    policy_model = MockModel()
    # Use same weights for ref to start
    import copy

    task.ref_model = copy.deepcopy(policy_model)
    task.ref_model.eval()

    # Batch
    # Chosen: [1, 2, 3]
    # Rejected: [1, 2, 4]
    # Labels: [-100, 2, 3] vs [-100, 2, 4] (Prompt len 1)

    B, S, V = 2, 4, 100
    chosen_ids = torch.randint(0, V, (B, S))
    rejected_ids = torch.randint(0, V, (B, S))
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, 0] = -100

    batch = {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
    }

    loss, metrics = task.train_step(batch, policy_model, criterion=None)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "reward_acc" in metrics
    assert "reward_margin" in metrics
