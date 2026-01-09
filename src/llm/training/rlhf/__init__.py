"""
RLHF (Reinforcement Learning from Human Feedback) module.

Contains PPO trainer and supporting utilities for RLHF training.
"""

from llm.training.rlhf.config import PPOConfig
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.rlhf.rollout_buffer import RolloutBatch, RolloutBuffer, RolloutSample

__all__ = ["PPOConfig", "PPOTrainer", "RolloutBuffer", "RolloutBatch", "RolloutSample"]
