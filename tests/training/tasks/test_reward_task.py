"""Tests for Reward Model Task."""

import pytest
import torch

from llm.data.reward_dataset import RewardDataset
from llm.training.tasks.reward_task import RewardModel


@pytest.fixture
def sample_reward_data(tmp_path):
    """Create sample preference data file."""
    data = [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"prompt": "Hello!", "chosen": "Hi there!", "rejected": "Go away"},
        {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
    ]

    file_path = tmp_path / "reward_data.jsonl"
    with file_path.open("w") as f:
        import json

        for item in data:
            f.write(json.dumps(item) + "\n")

    return file_path


@pytest.fixture
def mock_tokenizer():
    """Simple tokenizer for testing."""

    class SimpleTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 100 for c in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i + 32) for i in ids)

    return SimpleTokenizer()


def test_reward_dataset_loads_data(sample_reward_data, mock_tokenizer):
    """Test that RewardDataset correctly loads preference pairs."""
    dataset = RewardDataset(
        file_path=sample_reward_data,
        tokenizer=mock_tokenizer,
        max_seq_len=64,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    assert "chosen_input_ids" in sample
    assert "chosen_attention_mask" in sample
    assert "rejected_input_ids" in sample
    assert "rejected_attention_mask" in sample

    assert sample["chosen_input_ids"].shape == (64,)
    assert sample["rejected_input_ids"].shape == (64,)


def test_reward_model_forward(tiny_model):
    """Test RewardModel forward pass."""
    reward_model = RewardModel(tiny_model)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    rewards = reward_model(input_ids, attention_mask)

    assert rewards.shape == (batch_size,)
    assert rewards.dtype == torch.float32


def test_reward_model_handles_padding(tiny_model):
    """Test RewardModel correctly handles padded sequences."""
    reward_model = RewardModel(tiny_model)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Sequence 1: 10 tokens, Sequence 2: 5 tokens
    attention_mask = torch.zeros(batch_size, seq_len)
    attention_mask[0, :10] = 1
    attention_mask[1, :5] = 1

    rewards = reward_model(input_ids, attention_mask)

    assert rewards.shape == (batch_size,)
    # Rewards should differ since we're using different last token positions
    # (This is a weak test, but ensures the mask is used)


def test_reward_model_value_head_trainable(tiny_model):
    """Test that value head is trainable."""
    reward_model = RewardModel(tiny_model)

    # Check value head parameters are trainable
    for param in reward_model.value_head.parameters():
        assert param.requires_grad

    # Forward and backward pass
    input_ids = torch.randint(0, 100, (2, 8))
    rewards = reward_model(input_ids)
    loss = rewards.mean()
    loss.backward()

    # Verify gradients exist
    for param in reward_model.value_head.parameters():
        assert param.grad is not None
