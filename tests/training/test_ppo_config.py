"""Tests for unified PPOConfig in training config."""

from llm.training.core.config import Config, PPOConfig, PPOSettings
from llm.training.rlhf.config import PPOConfig as RLHFPPOConfig


def test_ppo_config_defaults():
    config = PPOConfig()
    assert config.clip_epsilon == 0.2
    assert config.ppo_epochs == 4


def test_ppo_settings_alias():
    assert PPOSettings is PPOConfig


def test_rlhf_reexport_matches_core():
    assert RLHFPPOConfig is PPOConfig


def test_config_embeds_ppo_config():
    config = Config()
    assert isinstance(config.ppo, PPOConfig)
