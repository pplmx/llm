"""Tests for MultimodalDataModule (text + modality pairing, TASK-226)."""

from __future__ import annotations

import pytest
import torch

from llm.multimodal import MultimodalDataModule


def _config(num_samples: int = 32):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_multimodal_datamodule_setup_and_train_batch():
    config = _config()
    module = MultimodalDataModule(config, modality="linear", input_dim=16)
    module.prepare_data()
    module.setup()
    assert module.encoder is not None

    loader, _sampler = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels", "modal_embeds"}
    assert batch["input_ids"].shape == (8, 24)
    assert batch["labels"].shape == (8, 24)
    # modal_embeds from the registry encoder at hidden_size (default embed_dim).
    assert batch["modal_embeds"].shape == (8, 24)
    assert torch.isfinite(batch["modal_embeds"]).all()


def test_multimodal_datamodule_uses_requested_embed_dim():
    config = _config()
    module = MultimodalDataModule(config, modality="linear", input_dim=8, embed_dim=17)
    module.setup()
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert batch["modal_embeds"].shape == (8, 17)


def test_multimodal_datamodule_val_dataloader():
    config = _config()
    module = MultimodalDataModule(config, modality="linear")
    module.setup()
    loader, _sampler = module.val_dataloader(rank=0, world_size=1)
    assert loader is not None
    batch = next(iter(loader))
    assert "modal_embeds" in batch


def test_multimodal_datamodule_unknown_modality_raises():

    config = _config()
    module = MultimodalDataModule(config, modality="does-not-exist")
    module.prepare_data()
    with pytest.raises(ValueError, match="not found"):
        module.setup()
