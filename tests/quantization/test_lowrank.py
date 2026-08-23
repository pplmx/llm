"""Tests for low-rank (SVD U-V) decomposition (ROADMAP 13.4, RIL TASK-225)."""

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


def test_lowrank_config_requires_exactly_one_rank_knob():
    from llm.quantization.lowrank import LowRankConfig

    with pytest.raises(ValueError, match="exactly one of rank or rank_ratio"):
        LowRankConfig()  # neither
    with pytest.raises(ValueError, match="exactly one of rank or rank_ratio"):
        LowRankConfig(rank=8, rank_ratio=0.5)  # both
    with pytest.raises(ValueError, match="rank must be > 0"):
        LowRankConfig(rank=0)
    with pytest.raises(ValueError, match="rank_ratio must be in"):
        LowRankConfig(rank_ratio=1.5)


def test_full_rank_decomposition_is_exact():
    from llm.quantization.lowrank import _relative_error, decompose_layer

    torch.manual_seed(0)
    lin = nn.Linear(12, 8)
    out = decompose_layer(lin, rank=8)  # full rank (min(8,12))
    assert out.rank == 8
    err = _relative_error(out.reconstruct(), lin.weight.detach())
    assert err < 1e-4
    x = torch.randn(4, 12)
    assert torch.allclose(out(x), lin(x), atol=1e-3)


def test_low_rank_has_lower_rank_and_reports_compression():
    from llm.quantization.lowrank import LowRankLinear, decompose_layer

    lin = nn.Linear(64, 32)
    out = decompose_layer(lin, rank=4)
    assert isinstance(out, LowRankLinear)
    assert out.rank == 4
    assert out.compression_ratio() > 1.0  # (32*64) / (32*4 + 4*64)
    assert out.u.shape == (32, 4)
    assert out.v.shape == (4, 64)


def test_decompose_model_replaces_linears_and_reports_stats():
    from llm.quantization.lowrank import LowRankConfig, LowRankLinear, compute_compression, decompose_model

    model = TwoLinearMLP(hidden=16)
    stats = decompose_model(model, LowRankConfig(rank=2))
    assert sum(1 for m in model.modules() if isinstance(m, nn.Linear)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, LowRankLinear)) == 2
    assert stats["compression_ratio"] > 1.0
    assert stats["compression_ratio"] == pytest.approx(compute_compression(model))
    assert 0.0 <= stats["relative_error"] < 1.0
    assert len(stats["layers"]) == 2


def test_decompose_model_rank_ratio_auto():
    from llm.quantization.lowrank import LowRankConfig, decompose_model

    model = TwoLinearMLP(hidden=16)
    stats = decompose_model(model, LowRankConfig(rank_ratio=1.0))  # full rank
    assert stats["relative_error"] < 1e-3
    # 16->32 fc1: min=16 -> rank 16; 32->16 fc2: min=16 -> rank 16
    assert all(rank == 16 for _, rank in stats["layers"])


def test_decompose_model_respects_target_modules():
    from llm.quantization.lowrank import LowRankConfig, LowRankLinear, decompose_model

    model = TwoLinearMLP(hidden=16)
    decompose_model(model, LowRankConfig(rank=2, target_modules=["fc1"]))
    assert sum(1 for m in model.modules() if isinstance(m, LowRankLinear)) == 1
    assert isinstance(model.fc1, LowRankLinear)
    assert isinstance(model.fc2, nn.Linear)


def test_decompose_forward_finite_and_shape_preserving():
    from llm.quantization.lowrank import LowRankConfig, decompose_model

    model = TwoLinearMLP(hidden=16)
    before = model(torch.randn(4, 16))
    decompose_model(model, LowRankConfig(rank=3))
    after = model(torch.randn(4, 16))
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_public_exports():
    import llm.quantization as q

    for name in ("LowRankLinear", "LowRankConfig", "decompose_model", "compute_compression"):
        assert hasattr(q, name)


class _Cyclic(BaseDataModule):
    """Fixed deterministic next-token corpus (token_{j+1} = (token_j + 1) % V)."""

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
    x = torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len].unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(-1)
    target = torch.roll(x, -1, dims=1)
    return (pred[:, :-1] == target[:, :-1]).float().mean().item()


def test_decomposed_decoder_full_rank_keeps_accuracy_rank1_degrades():
    """Overfit a tiny decoder, decomposing at full rank preserves accuracy while
    rank-1 collapse degrades it — proving the reconstruction is real."""
    from llm.quantization.lowrank import LowRankConfig, decompose_model
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
        torch.manual_seed(5)
        module = _Cyclic(cfg)
        task = LanguageModelingTask(cfg, module)
        engine = TrainingEngine(config=cfg, task=task, rank=0, world_size=1, data_module=module)
        for epoch in range(30):
            engine._run_epoch(epoch)

        base_acc = _cyclic_accuracy(engine.model)
        assert base_acc > 0.9, f"model did not learn the cyclic rule: {base_acc:.3f}"

        full = ModelFactory.from_config(cfg.model)
        full.load_state_dict(engine.model.state_dict())
        full_stats = decompose_model(full, LowRankConfig(rank_ratio=1.0))
        assert full_stats["relative_error"] < 1e-3
        assert _cyclic_accuracy(full) >= base_acc - 0.05, "full-rank reconstruction hurt accuracy"

        collapsed = ModelFactory.from_config(cfg.model)
        collapsed.load_state_dict(engine.model.state_dict())
        decompose_model(collapsed, LowRankConfig(rank=1))
        assert _cyclic_accuracy(collapsed) < 0.6, "rank-1 collapse did not degrade the model"
    finally:
        torch.set_num_threads(prev)
