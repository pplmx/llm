"""DPO-vs-PPO alignment benchmark (TASK-232 / ROADMAP 11.3).

Unit-tests the shared judge-reward + preference metric, then a CPU e2e showing
DPO *converges* on the shared preference signal (preference fraction 0 -> ~1),
and an integration test showing the PPO leg captures a finite mean-reward
trajectory end-to-end through the real PPOTrainer.  The on-policy PPO rollout
path is intentionally NOT asserted to converge here: with a sparse end-token
reward and a tiny CPU budget it is noisy/slower, which the docs guide reports
as the benchmark's honest finding (see docs/guides/dpo_vs_ppo.md).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _config(
    vocab_size: int = 128,
    num_samples: int = 32,
    batch_size: int = 8,
    lr: float = 1e-3,
    grpo_group_size: int = 4,
):
    from llm.training.core.config import (
        Config,
        DistributedConfig,
        ModelConfig,
        OptimizationConfig,
        TrainingConfig,
    )

    return Config(
        model=ModelConfig(vocab_size=vocab_size, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24),
        training=TrainingConfig(
            batch_size=batch_size,
            epochs=1,
            num_samples=num_samples,
            lr=lr,
            warmup_epochs=0,
            grpo_group_size=grpo_group_size,
            grpo_clip_eps=0.2,
            grpo_kl_beta=0.01,
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_target_token_reward_respects_target_and_mask():
    from llm.training.rlhf.aligner_benchmark import TargetTokenReward

    reward = TargetTokenReward(target_token=3)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6], [3, 3, 3]])
    mask = torch.ones_like(ids)
    assert reward(ids, mask).tolist() == [1.0, 0.0, 1.0]

    # Padding beyond the last real token must not change the reward: the last
    # REAL token of the row is still id 3 (index 0).
    padded = torch.tensor([[3, 0, 0]])
    padded_mask = torch.tensor([[1, 0, 0]])
    assert reward(padded, padded_mask).item() == 1.0

    # No mask: the final position is used verbatim.
    assert reward(torch.tensor([[7, 7, 2]])).item() == 0.0


def test_preference_fraction_orders_pairs_by_log_prob():
    from llm.training.rlhf.aligner_benchmark import preference_fraction

    class _BiasToken5(nn.Module):
        """Logits strongly favor token id 5 at every position."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq = x.shape
            out = torch.zeros(batch, seq, 8)
            out[..., 5] = 10.0
            return out

    model = _BiasToken5()
    # Length-2 responses: the SECOND token is the predicted position (the
    # first token is a prompt/conditioning prefix with no own log-prob).
    chosen = torch.tensor([[2, 5], [3, 5], [4, 5]])
    rejected = torch.tensor([[2, 3], [3, 1], [4, 2]])
    assert preference_fraction(model, chosen, rejected) == 1.0

    # Swap two rejected to chosen-preference rows: fraction drops to 1/3.
    rejected2 = torch.tensor([[2, 3], [3, 5], [4, 5]])
    assert preference_fraction(model, chosen, rejected2) == pytest.approx(1.0 / 3.0)

    with pytest.raises(ValueError, match="equal"):
        preference_fraction(model, torch.zeros(3, 4), torch.zeros(3, 5))


def test_reference_kl_tracks_divergence_from_reference():
    """Unit check of the reward-over-optimization diagnostic: identical
    policy/reference -> ~0 KL; a diverged policy -> positive KL."""
    from llm.training.rlhf.aligner_benchmark import _reference_kl

    class _BiasFull(nn.Module):
        def __init__(self, token: int) -> None:
            super().__init__()
            self.token = token

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq = x.shape
            out = torch.zeros(batch, seq, 8)
            out[..., self.token] = 10.0
            return out

    seqs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    reference = _BiasFull(token=5)
    assert _reference_kl(_BiasFull(token=5), reference, seqs) < 1e-6
    assert _reference_kl(_BiasFull(token=3), reference, seqs) > 0.05


def test_dpo_benchmark_preference_fraction_rises():
    """CPU e2e: on the shared judge-labeled preference set, DPO (off-policy)
    reliably converges — preference fraction 0 -> ~1."""
    from llm.training.rlhf.aligner_benchmark import run_dpo

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_dpo(_config(), epochs=40)
        traj = result["preference_fraction_trajectory"]
        assert len(traj) == 40
        assert traj[0] < 0.5, f"expected near-random initial preference, got {traj[0]:.3f}"
        assert result["final_preference_fraction"] > 0.9
        assert all(loss == loss for loss in result["dpo_loss_trajectory"])  # finite
    finally:
        torch.set_num_threads(prev)


def test_grpo_benchmark_group_reward_fraction_rises():
    """CPU e2e: the group-relative GRPO path converges — its group-reward
    fraction (fraction of groups whose most-likely response is the rewarded
    target) rises on CPU."""
    from llm.training.rlhf.aligner_benchmark import run_grpo

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_grpo(_config(), epochs=40)
        traj = result["group_reward_fraction_trajectory"]
        assert len(traj) == 40
        assert traj[0] < 0.5, f"expected poor initial group reward, got {traj[0]:.3f}"
        assert result["final_group_reward_fraction"] > 0.9
        assert all(loss == loss for loss in result["grpo_loss_trajectory"])  # finite
    finally:
        torch.set_num_threads(prev)


def test_ppo_benchmark_reports_finite_reward_trajectory():
    """The on-policy PPO leg runs the real PPOTrainer and captures a finite
    mean-reward trajectory.  No convergence asserted (sparse reward + tiny CPU
    budget is noisy — the documented finding), only that the harness reports
    valid metrics end-to-end."""
    from llm.training.rlhf.aligner_benchmark import run_ppo

    config = _config(num_samples=8)
    config.ppo.ppo_epochs = 2
    config.ppo.response_max_len = 3
    config.ppo.mini_batch_size = 1
    config.ppo.temperature = 0.5

    prompts = ["hello", "world", "prompt", "test", "alpha", "beta", "gamma", "delta"]
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        result = run_ppo(config, steps=3, prompts=prompts, target_token=1)
        traj = result["mean_reward_trajectory"]
        assert len(traj) == 3
        assert all(t == t for t in traj)  # finite (no NaN)
        assert all(0.0 <= t <= 1.0 for t in traj)
    finally:
        torch.set_num_threads(prev)


def test_compare_dpo_vs_ppo_returns_summary():
    """The comparison entry point runs both aligners and returns a summary."""
    from llm.training.rlhf.aligner_benchmark import compare_dpo_vs_ppo

    prompts = ["hello", "world", "prompt", "test"]
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        out = compare_dpo_vs_ppo(
            _config(num_samples=16),
            dpo_epochs=3,
            ppo_steps=1,
            grpo_epochs=20,
            prompts=prompts,
        )
        assert set(out) == {"dpo", "ppo", "grpo", "summary"}
        assert isinstance(out["summary"], str)
        assert out["summary"]
        assert out["dpo"]["final_preference_fraction"] > out["dpo"]["preference_fraction_trajectory"][0]
        assert 0.0 <= out["ppo"]["final_mean_reward"] <= 1.0
        assert out["grpo"]["final_group_reward_fraction"] > out["grpo"]["group_reward_fraction_trajectory"][0]
        # Reference-KL (reward-over-optimization diagnostic) is finite and >= 0
        # for every aligner (reporting it is the deliverable, not its size).
        for align in ("dpo", "ppo", "grpo"):
            kl = out[align]["final_reference_kl"]
            assert isinstance(kl, float)
            assert kl >= 0.0
            assert kl == kl  # no NaN
    finally:
        torch.set_num_threads(prev)
