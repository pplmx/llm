"""
PPO Configuration for RLHF training.
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    Configuration for Proximal Policy Optimization (PPO) training.

    Attributes:
        clip_epsilon: PPO clipping parameter for policy ratio.
        kl_coef: Coefficient for KL divergence penalty.
        value_coef: Coefficient for value function loss.
        entropy_coef: Coefficient for entropy bonus.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        ppo_epochs: Number of PPO epochs per rollout.
        mini_batch_size: Mini-batch size for PPO updates.
        max_grad_norm: Maximum gradient norm for clipping.
        target_kl: Target KL divergence for early stopping.
        normalize_advantages: Whether to normalize advantages.
        rollout_batch_size: Number of prompts per rollout.
        response_max_len: Maximum response length during generation.
    """

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 1.0  # Discount factor (1.0 for language modeling)

    # Training hyperparameters
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    max_grad_norm: float = 1.0
    target_kl: float | None = None  # Early stopping if KL exceeds this

    # Generation hyperparameters
    rollout_batch_size: int = 16
    response_max_len: int = 256
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    # Normalization
    normalize_advantages: bool = True
    normalize_rewards: bool = False

    # Learning rates (can override base config)
    policy_lr: float | None = None
    value_lr: float | None = None

    # Reference model
    use_ref_model: bool = True
    ref_model_update_freq: int = 0  # 0 = never update, >0 = update every N steps
