"""Tests for GRPO (TASK-229): group advantages, GRPO loss, and a CPU e2e."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import llm.training.tasks.builtin  # noqa: F401 - registers built-in tasks
from llm.training.task_registry import TASK_REGISTRY


def _config(**training):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    defaults = {
        "batch_size": 8,
        "epochs": 1,
        "num_samples": 8,
        "lr": 3e-3,
        "warmup_epochs": 0,
        "grpo_group_size": 4,
        "grpo_kl_beta": 0.01,
        "grpo_clip_eps": 0.2,
    }
    defaults.update(training)
    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24),
        training=TrainingConfig(**defaults),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_group_advantages_normalizes_within_group():
    from llm.training.rlhf.grpo import group_advantages

    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    adv = group_advantages(rewards)
    assert adv.shape == rewards.shape
    # Each row is zero-mean (z-scores).
    assert torch.allclose(adv.mean(dim=-1), torch.zeros(2), atol=1e-5)
    # The rewarded response has the largest advantage within its group.
    assert torch.all(adv.argmax(dim=-1) == 0)


def test_grpo_loss_validates_config():
    from llm.training.rlhf.grpo import GRPOLoss

    with pytest.raises(ValueError, match="clip_eps"):
        GRPOLoss(clip_eps=1.5)
    with pytest.raises(ValueError, match="kl_beta"):
        GRPOLoss(kl_beta=-1.0)


def test_grpo_loss_zero_kl_and_clip_boundaries():
    from llm.training.rlhf.grpo import GRPOLoss

    # identical old/cur -> ratio 1 -> clip is a no-op; with kl_beta=0 the loss
    # is -mean(A)  (advantage already detached, so a positive advantage lowers loss).
    loss = GRPOLoss(clip_eps=0.0, kl_beta=0.0)
    lp = torch.zeros(4, 3)
    old = torch.zeros(4, 3)
    ref = torch.zeros(4, 3)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    out = loss(lp, old, ref, adv)
    assert torch.allclose(out, -adv.mean(), atol=1e-5)


def test_grpo_loss_ratio_clipping_limits_negative_effect():
    from llm.training.rlhf.grpo import GRPOLoss

    # A big ratio with a NEGATIVE advantage is clipped to (1 - eps), so the
    # surrogate stays bounded (no extreme negative push from one token).
    loss = GRPOLoss(clip_eps=0.2, kl_beta=0.0)
    cur = torch.tensor([[5.0, 5.0]])
    old = torch.tensor([[-5.0, -5.0]])  # ratio = e^10 huge
    ref = torch.zeros_like(cur)
    adv = torch.tensor([-1.0])
    clipped_surrogate = -torch.clamp(torch.exp(cur - old), 0.8, 1.2).mean() * adv
    grpo_loss = loss(cur, old, ref, adv)
    assert grpo_loss == pytest.approx(clipped_surrogate.item(), abs=1e-4)


def test_grpo_task_registered_and_builds():
    from llm.data.modules.grpo import GRPODataModule
    from llm.training.rlhf.grpo import GRPOLoss
    from llm.training.tasks.grpo_task import GRPOTask

    spec = TASK_REGISTRY.get("grpo")
    assert spec is not None
    assert spec.task_cls is GRPOTask
    config = _config()
    module = GRPODataModule(config)
    task = GRPOTask(config, module)
    assert isinstance(task.build_criterion(), GRPOLoss)


def test_grpo_datamodule_batch_and_group_rewards():
    from llm.data.modules.grpo import GRPODataModule

    config = _config()
    module = GRPODataModule(config)
    module.prepare_data()
    module.setup()
    assert module.response_tokens.shape == (8, 24)
    assert module.group_rewards.shape == (2, 4)
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels", "rewards"}
    assert batch["input_ids"].shape == (8, 24)
    # One reward-1 response per group (the target).
    assert batch["rewards"].reshape(2, 4).sum(dim=-1).tolist() == [1.0, 1.0]


def test_grpo_datamodule_rejects_bad_batch_group_alignment():
    from llm.data.modules.grpo import GRPODataModule

    config = _config(batch_size=10, grpo_group_size=4)  # 10 % 4 != 0
    with pytest.raises(ValueError, match="multiple of grpo_group_size"):
        GRPODataModule(config)


def _group_reward_hits(model, module):
    model.eval()
    # Probe on the model's own device: the engine moved the model to CUDA but
    # this standalone probe feeds data-module tensors (CPU) directly — a
    # device-mismatch on GPU machines without the .to().
    response_tokens = module.response_tokens.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(response_tokens)
        lp = functional.log_softmax(logits, -1).gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1).sum(-1)
    lp = lp.reshape(-1, module.group_size)
    return (lp.argmax(-1) == 0).float().mean().item()


def test_grpo_engine_improves_group_reward():
    """CPU e2e: GRPO raises the fraction of groups that favor the rewarded
    response (0 -> ~1), proving the group-relative objective works end-to-end."""
    from llm.data.modules.grpo import GRPODataModule
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.grpo_task import GRPOTask

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config()
        module = GRPODataModule(config)
        module.prepare_data()
        module.setup()
        task = GRPOTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        before = _group_reward_hits(engine.model, module)
        losses = [engine._run_epoch(epoch) for epoch in range(40)]
        after = _group_reward_hits(engine.model, module)

        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0], f"GRPO loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert before < 0.5, f"expected poor initial group reward, got {before:.3f}"
        assert after > 0.9, f"GRPO did not improve group reward: {before:.3f} -> {after:.3f}"
    finally:
        torch.set_num_threads(prev)
