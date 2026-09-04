"""Tests for rejection sampling (TASK-230 / ROADMAP 11.4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import llm.training.tasks.builtin  # noqa: F401 - registers built-in tasks


def _config(num_samples: int = 64):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_select_top_k_keeps_highest_scores():
    from llm.training.rlhf.rejection_sampling import select_top_k

    scores = torch.tensor([0.1, 5.0, 3.0, 9.0, 2.0])
    mask = select_top_k(scores, 2)
    assert mask.tolist() == [False, True, False, True, False]


def test_select_top_k_clamps_large_k_and_validates():
    from llm.training.rlhf.rejection_sampling import select_top_k

    scores = torch.tensor([1.0, 2.0, 3.0])
    assert select_top_k(scores, 100).sum().item() == 3  # k > N -> keep all
    with pytest.raises(ValueError, match="k must be"):
        select_top_k(scores, 0)
    with pytest.raises(ValueError, match="non-empty"):
        select_top_k(torch.tensor([]), 1)


def test_select_above_threshold():
    from llm.training.rlhf.rejection_sampling import select_above_threshold

    scores = torch.tensor([0.0, 0.5, 1.0, -1.0])
    assert select_above_threshold(scores, 0.5).tolist() == [False, True, True, False]


def test_rejection_sample_requires_exactly_one_mode():
    from llm.training.rlhf.rejection_sampling import rejection_sample

    scores = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="exactly one of k or threshold"):
        rejection_sample(scores)  # neither
    with pytest.raises(ValueError, match="exactly one of k or threshold"):
        rejection_sample(scores, k=2, threshold=1.0)  # both


def test_rejection_sample_reports_kept_over_base_reward():
    from llm.training.rlhf.rejection_sampling import rejection_sample

    # 100 responses: half reward 1, half reward 0; keep top-25% (all reward-1).
    scores = torch.tensor([1.0] * 50 + [0.0] * 50)
    _mask, stats = rejection_sample(scores, k=25)
    assert stats["kept_mean_reward"] > stats["base_mean_reward"]
    assert stats["kept_fraction"] == pytest.approx(0.25)


def _kept_log_likelihood(model, responses):
    model.eval()
    # Probe on the model's own device (engine moved it to CUDA on GPU machines;
    # this standalone probe feeds raw tensors — device-mismatch without .to()).
    responses = responses.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(responses)
    lp = functional.log_softmax(logits, -1)
    # next-token: logits[:,:-1] vs tokens[:,1:] (LM shift)
    seq_lp = lp[:, :-1].gather(-1, responses[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
    return seq_lp.mean().item()


def test_rejection_sample_sft_onboarding():
    """CPU e2e: reject low-reward responses (kept mean reward > base), then
    SFT-on-selected via the standard LM task and verify the model becomes much
    more likely to reproduce the kept (high-reward) responses."""
    from llm.data.modules.rejection_sample import RejectionSampleDataModule
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(num_samples=64)
        module = RejectionSampleDataModule(config, mode="top_k", k=16)
        module.prepare_data()
        module.setup()

        # Filtering property: the kept responses have higher mean reward.
        assert module.stats is not None
        assert module.stats["kept_mean_reward"] > module.stats["base_mean_reward"]

        task = LanguageModelingTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        before = _kept_log_likelihood(engine.model, module.train_dataset.tensors[0])
        losses = [engine._run_epoch(epoch) for epoch in range(25)]
        after = _kept_log_likelihood(engine.model, module.train_dataset.tensors[0])

        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0], f"SFT-on-selected loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert after > before, f"kept-response likelihood did not improve: {before:.3f} -> {after:.3f}"
    finally:
        torch.set_num_threads(prev)


def test_rejection_sample_datamodule_rejects_bad_mode():
    from llm.data.modules.rejection_sample import RejectionSampleDataModule

    with pytest.raises(ValueError, match="mode must be"):
        RejectionSampleDataModule(_config(), mode="med-sam")


def test_rejection_sample_datamodule_refuses_empty_keep():
    """When no response clears the filter, the kept TensorDataset would be
    length 0 → an empty epoch → the engine averages over zero batches. The
    module must fail at setup with a clear error (RIL ISS-336)."""
    from llm.data.modules.rejection_sample import RejectionSampleDataModule

    config = _config(num_samples=8)
    module = RejectionSampleDataModule(config, mode="threshold", threshold=2.0)  # rewards are 0/1
    with pytest.raises(ValueError, match="kept ZERO"):
        module.setup()
