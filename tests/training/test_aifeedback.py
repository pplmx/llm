"""Tests for AI-feedback preference labeling -> DPO (TASK-231 / ROADMAP 11.4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import llm.training.tasks.builtin  # noqa: F401 - registers built-in tasks


def _config(batch_size: int = 8, num_samples: int = 32):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24),
        training=TrainingConfig(batch_size=batch_size, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_target_token_judge_scores():
    from llm.training.rlhf.aifeedback import TargetTokenJudge

    judge = TargetTokenJudge(target_token=0)
    responses = torch.tensor([[1, 2, 0], [3, 4, 5], [0, 0, 0]])
    assert judge.score_batch(responses).tolist() == [1.0, 0.0, 1.0]


def test_prefer_batch_labels_chosen_by_judge():
    from llm.training.rlhf.aifeedback import TargetTokenJudge, prefer_batch

    judge = TargetTokenJudge(target_token=0)
    a = torch.tensor([[1, 2, 0], [9, 8, 7], [0, 0, 0]])  # rows 0,2 good
    b = torch.tensor([[3, 4, 5], [6, 0, 0], [1, 2, 3]])  # row 1 good
    chosen, rejected = prefer_batch(a, b, judge)
    # chosen always end with the target; rejected never do (ties prefer a).
    assert (chosen[:, -1] == 0).all()
    assert (rejected[:, -1] != 0).all()


def test_prefer_batch_validates_shapes():
    from llm.training.rlhf.aifeedback import TargetTokenJudge, prefer_batch

    judge = TargetTokenJudge()
    with pytest.raises(ValueError, match="equal"):
        prefer_batch(torch.zeros(3, 4), torch.zeros(3, 5), judge)


def test_aifeedback_datamodule_yields_dpo_batch():
    from llm.data.modules.aifeedback import AIFeedbackDataModule

    config = _config()
    module = AIFeedbackDataModule(config)
    module.prepare_data()
    module.setup()
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    expected = {
        "chosen_input_ids",
        "chosen_labels",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_labels",
        "rejected_attention_mask",
    }
    assert set(batch) == expected
    assert batch["chosen_input_ids"].shape == (8, 24)
    assert (batch["chosen_attention_mask"] == 1).all()


def _seq_log_probs(model, responses):
    model.eval()
    # Probe on the model's own device: the engine moved the model to CUDA but
    # this standalone probe feeds data-module tensors (CPU) directly — a
    # device-mismatch on GPU machines without the .to().
    responses = responses.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(responses)  # channels-last: [B, L, V]
    lp = functional.log_softmax(logits, -1)
    return lp[:, :-1].gather(-1, responses[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)  # [B]


def _preference_fraction(model, module):
    chosen_lp = _seq_log_probs(model, module.chosen)
    rejected_lp = _seq_log_probs(model, module.rejected)
    return (chosen_lp > rejected_lp).float().mean().item()


def test_aifeedback_dpo_prefers_judge_chosen():
    """CPU e2e: DPO on AI-judge-labeled pairs shifts the policy so it prefers
    the judge's chosen response (preference fraction 0 -> ~1)."""
    from llm.data.modules.aifeedback import AIFeedbackDataModule
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.dpo_task import DPOTask

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(batch_size=8, num_samples=32)
        module = AIFeedbackDataModule(config)
        module.prepare_data()
        module.setup()
        task = DPOTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        before = _preference_fraction(engine.model, module)
        losses = [engine._run_epoch(epoch) for epoch in range(40)]
        after = _preference_fraction(engine.model, module)

        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0], f"DPO loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert before < 0.5, f"expected near-random initial preference, got {before:.3f}"
        assert after > 0.9, f"DPO did not learn to prefer judge-chosen: {before:.3f} -> {after:.3f}"
    finally:
        torch.set_num_threads(prev)
