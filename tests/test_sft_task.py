from unittest.mock import MagicMock

import torch
import torch.nn as nn

from llm.training.core.config import Config
from llm.training.tasks.sft_task import SFTTask


class MockModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attn_mask=None, **kwargs):
        # Return random logits: (B, S, V)
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.linear.out_features)


def test_sft_task_train_step():
    config = Config()
    task = SFTTask(config, data_module=MagicMock())
    model = MockModel()
    criterion = task.build_criterion()

    # Batch: 2 sequences, len 4
    # Seq 1: [P, P, R, R] -> Labels [-100, -100, R, R] (P=Prompt, R=Response)
    # Seq 2: [P, R, R, R]

    batch = {
        "input_ids": torch.randint(0, 100, (2, 4)),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]]),  # 2 examples
        "attention_mask": torch.ones(2, 4),
    }

    loss, metrics = task.train_step(batch, model, criterion)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss" in metrics
    assert "ppl" in metrics
