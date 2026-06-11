"""Tests for unified PPOConfig in training config."""

from llm.training.core.config import Config, PPOConfig


def test_ppo_config_defaults():
    config = PPOConfig()
    assert config.clip_epsilon == 0.2
    assert config.ppo_epochs == 4


def test_config_embeds_ppo_config():
    config = Config()
    assert isinstance(config.ppo, PPOConfig)
