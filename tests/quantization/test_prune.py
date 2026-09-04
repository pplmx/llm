"""Tests for weight pruning (ROADMAP 13.4, RIL TASK-224)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.base import BaseDataModule


class TwoLinearMLP(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_pruning_config_validates_ratio_and_method():
    from llm.quantization.prune import PruningConfig

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ratio must be in"):
            PruningConfig(ratio=bad)
    with pytest.raises(ValueError, match="unknown prune method"):
        PruningConfig(method="sensor")


def test_prune_model_replaces_linears_and_reports_sparsity():
    from llm.quantization.prune import PrunedLinear, PruningConfig, compute_sparsity, prune_model

    model = TwoLinearMLP(hidden=16)
    sparsity = prune_model(model, PruningConfig(ratio=0.5))

    assert sum(1 for m in model.modules() if isinstance(m, nn.Linear)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, PrunedLinear)) == 2
    assert 0.5 <= sparsity <= 0.5 + 1e-6
    assert compute_sparsity(model) == pytest.approx(sparsity)
    # Magnitude policy: the kept entries are exactly the largest magnitudes.
    kept = model.fc1.weight_mask.bool()
    kept_mags = model.fc1.weight.abs()[kept]
    dropped_mags = model.fc1.weight.abs()[~kept]
    assert kept_mags.min() >= dropped_mags.max()


def test_magnitude_prune_with_ties_achieves_exact_ratio():
    """Regression (RIL TASK-309): an ``abs >= threshold`` mask keeps EVERY
    boundary tie, so a weight whose zero-mass exceeds the drop quota prunes
    NOTHING (e.g. a 20%-nonzero layer at ratio 0.5 achieved 0.0 sparsity —
    silent wrong result on already-pruned / dead-channel weights). The
    topk-index mask prunes exactly ``(1 - ratio)`` regardless of ties.
    """
    from llm.quantization.prune import PruningConfig, compute_sparsity, prune_model

    model = TwoLinearMLP(hidden=16)
    # 80% of every Linear's entries are exactly 0.0 -> the keep boundary
    # lands on a huge tied mass (0.0), the regime that used to no-op.
    with torch.no_grad():
        for lin in (model.fc1, model.fc2):
            w = lin.weight
            keep_idx = torch.randperm(w.numel())[: round(0.2 * w.numel())]
            flat = torch.zeros(w.numel())
            flat[keep_idx] = 1.0
            w.copy_(flat.view_as(w))

    sparsity = prune_model(model, PruningConfig(ratio=0.5))
    assert sparsity == pytest.approx(0.5, abs=1e-6)
    assert compute_sparsity(model) == pytest.approx(0.5, abs=1e-6)
    for lin in (model.fc1, model.fc2):
        assert lin.weight_mask.float().mean().item() == pytest.approx(0.5, abs=1e-6)


def test_magnitude_mask_all_zero_weight_still_reaches_target():
    """A fully-dead (all-zero) linear still prunes to the target fraction.

    ``round((1 - ratio) * 100)`` is integral for these ratios, so each mask
    keeps exactly the requested fraction of a 100-entry all-zero weight.
    """
    from llm.quantization.prune import _magnitude_mask

    for ratio in (0.1, 0.5, 0.9):
        mask = _magnitude_mask(torch.zeros(100), ratio)
        assert mask.float().mean().item() == pytest.approx(1.0 - ratio, abs=1e-6)


def test_prune_model_respects_target_modules():
    from llm.quantization.prune import PrunedLinear, PruningConfig, prune_model

    model = TwoLinearMLP(hidden=16)
    prune_model(model, PruningConfig(ratio=0.5, target_modules=["fc1"]))
    assert sum(1 for m in model.modules() if isinstance(m, PrunedLinear)) == 1
    assert isinstance(model.fc1, PrunedLinear)
    assert isinstance(model.fc2, nn.Linear)


def test_random_pruning_is_reproducible_with_seed():
    from llm.quantization.prune import PruningConfig, prune_model

    a = TwoLinearMLP(hidden=16)
    b = TwoLinearMLP(hidden=16)
    prune_model(a, PruningConfig(ratio=0.5, method="random", random_seed=7))
    prune_model(b, PruningConfig(ratio=0.5, method="random", random_seed=7))
    assert torch.equal(a.fc1.weight_mask, b.fc1.weight_mask)
    assert torch.equal(a.fc2.weight_mask, b.fc2.weight_mask)


def test_prune_no_linear_raises():
    from llm.quantization.prune import PruningConfig, prune_model

    with pytest.raises(ValueError, match=r"no nn\.Linear"):
        prune_model(nn.Sequential(nn.ReLU()), PruningConfig(ratio=0.5))


def test_prune_unmatched_target_modules_raises_not_silent_noop():
    """Regression (TASK-228): a target_modules filter that matches nothing must
    fail loudly instead of silently returning 0% sparsity."""
    from llm.quantization.prune import PrunedLinear, PruningConfig, prune_model

    model = TwoLinearMLP(hidden=16)
    with pytest.raises(ValueError, match=r"matched no nn\.Linear"):
        prune_model(model, PruningConfig(ratio=0.5, target_modules=["does-not-exist"]))
    # The model is untouched (no silent partial mutation).
    assert sum(1 for m in model.modules() if isinstance(m, PrunedLinear)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, nn.Linear)) == 2


def test_pruned_forward_is_finite_and_shape_preserving():
    from llm.quantization.prune import PruningConfig, prune_model

    model = TwoLinearMLP(hidden=16)
    before = model(torch.randn(4, 16))
    prune_model(model, PruningConfig(ratio=0.7))
    after = model(torch.randn(4, 16))
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_public_exports():
    import llm.quantization as q

    for name in ("PrunedLinear", "PruningConfig", "prune_model", "compute_sparsity"):
        assert hasattr(q, name)


class _Cyclic(BaseDataModule):
    """Fixed deterministic next-token corpus: token_{j+1} = (token_j + 1) % V.

    Gives the model a real, learnable function so pruning accuracy is
    measurable (unlike the fully random DummyLMDataModule).
    """

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    def _data(self, n: int) -> TensorDataset:
        vocab = self.config.model.vocab_size
        seq_len = self.config.model.max_seq_len
        rows = [(torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len] + i).fmod(vocab).long() for i in range(n)]
        x = torch.stack(rows)
        return TensorDataset(x, x)

    def train_dataloader(self, rank, world_size, device=None):
        ds = self._data(self.config.training.num_samples)
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        return DataLoader(ds, batch_size=self.config.training.batch_size, sampler=sampler, drop_last=True), sampler

    def val_dataloader(self, rank, world_size):
        return None, None


def _tiny_decoder_config():
    from llm.training.core.config import ModelConfig

    return ModelConfig(vocab_size=32, hidden_size=24, num_layers=3, num_heads=3, max_seq_len=24)


def _cyclic_accuracy(model):
    model.eval()
    vocab, seq_len = 32, 24
    # Probe on the model's own device — the engine moves the model to CUDA when
    # available but this standalone probe feeds raw tensors (device-mismatch on
    # GPU machines).
    device = next(model.parameters()).device
    x = torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(-1)
    target = torch.roll(x, -1, dims=1)
    return (pred[:, :-1] == target[:, :-1]).float().mean().item()


def test_pruned_decoder_light_prune_keeps_accuracy_heavy_degrades():
    """Overfit a tiny decoder, prune lightly (accuracy preserved) and heavily
    (reported sparsity, accuracy collapses) — proving the pass is real."""
    from llm.quantization.prune import PruningConfig, compute_sparsity, prune_model
    from llm.runtime import ModelFactory
    from llm.training.core.config import Config, DistributedConfig, OptimizationConfig, TrainingConfig
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = Config(
            model=_tiny_decoder_config(),
            training=TrainingConfig(batch_size=8, epochs=1, num_samples=32, lr=1e-3, warmup_epochs=0),
            optimization=OptimizationConfig(use_compile=False, use_amp=False),
            distributed=DistributedConfig(parallel_strategy="ddp"),
        )
        torch.manual_seed(3)
        module = _Cyclic(cfg)
        task = LanguageModelingTask(cfg, module)
        engine = TrainingEngine(config=cfg, task=task, rank=0, world_size=1, data_module=module)
        for epoch in range(30):
            engine._run_epoch(epoch)

        base_acc = _cyclic_accuracy(engine.model)
        assert base_acc > 0.9, f"model did not learn the cyclic rule: {base_acc:.3f}"

        light = ModelFactory.from_config(cfg.model)
        light.load_state_dict(engine.model.state_dict())
        s_light = prune_model(light, PruningConfig(ratio=0.1))
        assert compute_sparsity(light) == pytest.approx(s_light, abs=1e-3)
        assert _cyclic_accuracy(light) >= base_acc - 0.15, "light pruning unexpectedly hurt accuracy"

        heavy = ModelFactory.from_config(cfg.model)
        heavy.load_state_dict(engine.model.state_dict())
        s_heavy = prune_model(heavy, PruningConfig(ratio=0.95))
        assert s_heavy >= 0.9
        assert _cyclic_accuracy(heavy) < 0.5, "heavy pruning did not degrade the model"
    finally:
        torch.set_num_threads(prev)
