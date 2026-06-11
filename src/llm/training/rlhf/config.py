"""Backward-compatible re-export; prefer ``llm.training.core.config.PPOConfig``."""

from llm.training.core.config import PPOConfig, PPOSettings

__all__ = ["PPOConfig", "PPOSettings"]
