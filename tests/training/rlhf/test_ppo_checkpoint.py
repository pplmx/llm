"""Tests for PPO trainer checkpoint state."""

import pytest
import torch

from llm.training.core.config import PPOConfig
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.rlhf.value_model import ValueModel


@pytest.fixture
def tiny_setup(tiny_model):
    from llm.training.tasks.reward_task import RewardModel

    class SimpleTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 100 for c in text[:10]]

        def decode(self, ids: list[int]) -> str:
            return "x"

        eos_id = 0

    policy = tiny_model
    reward_model = RewardModel(tiny_model)
    value_model = ValueModel(tiny_model)
    config = PPOConfig(value_coef=0.5, ppo_epochs=1, mini_batch_size=1, response_max_len=2)

    trainer = PPOTrainer(
        policy_model=policy,
        reward_model=reward_model,
        tokenizer=SimpleTokenizer(),
        config=config,
        value_model=value_model,
        device="cpu",
    )
    return trainer


def test_ppo_trainer_checkpoint_roundtrip(tiny_setup):
    trainer = tiny_setup
    trainer.global_step = 3

    state = trainer.get_checkpoint_state()
    assert state["global_step"] == 3
    assert "value_model" in state
    assert "value_optimizer" in state

    before = trainer.value_model.value_head.weight.detach().clone()
    trainer.value_model.value_head.weight.data.fill_(0.0)
    # Saved snapshot must not alias live parameters.
    assert not torch.allclose(state["value_model"]["value_head.weight"], torch.zeros_like(before))

    trainer.load_checkpoint_state(state)
    assert trainer.global_step == 3
    assert torch.allclose(trainer.value_model.value_head.weight, before)
