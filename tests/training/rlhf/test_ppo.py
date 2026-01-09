"""Tests for RLHF PPO Trainer."""

import pytest
import torch

from llm.training.rlhf.config import PPOConfig
from llm.training.rlhf.rollout_buffer import RolloutBuffer


class TestPPOConfig:
    """Tests for PPOConfig."""

    def test_default_config(self):
        """Test default PPO config values."""
        config = PPOConfig()

        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.1
        assert config.ppo_epochs == 4
        assert config.gae_lambda == 0.95

    def test_custom_config(self):
        """Test custom PPO config."""
        config = PPOConfig(
            clip_epsilon=0.1,
            ppo_epochs=2,
            mini_batch_size=32,
        )

        assert config.clip_epsilon == 0.1
        assert config.ppo_epochs == 2
        assert config.mini_batch_size == 32


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_add_sample(self):
        """Test adding samples to buffer."""
        buffer = RolloutBuffer()

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        assert len(buffer) == 1

    def test_compute_advantages(self):
        """Test advantage computation."""
        buffer = RolloutBuffer(normalize_advantages=False)

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        buffer.compute_advantages()

        assert buffer.samples[0].advantages is not None
        assert len(buffer.samples[0].advantages) == 2

    def test_get_batches(self):
        """Test mini-batch generation."""
        buffer = RolloutBuffer()

        # Add multiple samples
        for i in range(4):
            buffer.add(
                prompt_ids=torch.tensor([1, 2, 3]),
                response_ids=torch.tensor([4, 5, 6]),
                rewards=torch.tensor(float(i)),
                old_log_probs=torch.tensor([-0.5, -0.3, -0.2]),
            )

        buffer.compute_advantages()

        batches = list(buffer.get_batches(mini_batch_size=2))

        assert len(batches) == 2
        assert batches[0].input_ids.shape[0] == 2
        assert batches[0].rewards.shape[0] == 2

    def test_clear_buffer(self):
        """Test buffer clearing."""
        buffer = RolloutBuffer()

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        assert len(buffer) == 1
        buffer.clear()
        assert len(buffer) == 0


class TestPPOTrainer:
    """Tests for PPOTrainer."""

    @pytest.fixture
    def tiny_setup(self, tiny_model):
        """Create minimal setup for PPO trainer tests."""
        from llm.training.tasks.reward_task import RewardModel

        class SimpleTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 100 for c in text[:10]]

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(i + 32) for i in ids)

            eos_id = 0

        policy = tiny_model
        reward_model = RewardModel(tiny_model)
        tokenizer = SimpleTokenizer()
        config = PPOConfig(
            ppo_epochs=1,
            mini_batch_size=2,
            response_max_len=5,
        )

        return policy, reward_model, tokenizer, config

    def test_ppo_trainer_init(self, tiny_setup):
        """Test PPO trainer initialization."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device="cpu",
        )

        assert trainer.policy is policy
        assert trainer.reward_model is reward_model
        assert trainer.ref_model is not None  # Should create ref model

    def test_generate_responses(self, tiny_setup):
        """Test response generation."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device="cpu",
        )

        prompts = ["Hello", "Hi"]
        prompt_ids, response_ids, log_probs = trainer.generate_responses(prompts)

        assert len(prompt_ids) == 2
        assert len(response_ids) == 2
        assert len(log_probs) == 2

        # Each response should have tokens
        assert len(response_ids[0]) > 0
        assert len(log_probs[0]) == len(response_ids[0])

    def test_compute_rewards(self, tiny_setup):
        """Test reward computation."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device="cpu",
        )

        prompt_ids = [torch.tensor([1, 2, 3])]
        response_ids = [torch.tensor([4, 5])]

        rewards = trainer.compute_rewards(prompt_ids, response_ids)

        assert len(rewards) == 1
        assert rewards[0].dim() == 0  # Scalar
