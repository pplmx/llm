"""Tests for the Whisper-style audio encoder + raw-spectrogram data path (ROADMAP 12.2).

Covers registry registration, shape/determinism invariants, gradient flow, the
frozen-tower option, and the ``MultimodalDataModule``/e2e wiring (raw
spectrograms -> audio tokens -> fused text prefix) on CPU.
"""

from __future__ import annotations

import pytest
import torch

from llm.multimodal import MODALITY_ENCODER_REGISTRY, AudioSpectrogramEncoder, MultimodalDataModule


def _audio(batch: int = 2, frames: int = 32, mels: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 1, frames, mels)


def test_audio_registered_in_modality_registry():
    encoder_cls = MODALITY_ENCODER_REGISTRY.get("audio")
    assert encoder_cls is AudioSpectrogramEncoder
    enc = encoder_cls(patch_size=8, embed_dim=8, layers=1, num_heads=2, n_frames=32, n_mels=32)
    assert enc.modality == "audio"


def test_audio_encode_shape_and_deterministic():
    torch.manual_seed(1)
    enc = AudioSpectrogramEncoder(patch_size=8, embed_dim=8, layers=2, num_heads=2, n_frames=32, n_mels=32)
    audio = _audio(3, 32, 32)
    out = enc.encode(audio)
    assert out.shape == (3, 17, 8)  # (32/8)^2 = 16 patches + CLS
    assert torch.equal(out, enc.encode(audio))
    assert torch.isfinite(out).all()
    assert enc.num_tokens == 17


def test_audio_gradient_flows_to_all_trainable_parts():
    torch.manual_seed(3)
    enc = AudioSpectrogramEncoder(patch_size=8, embed_dim=8, layers=2, num_heads=2, n_frames=32, n_mels=32)
    enc.train()
    audio = _audio(2, 32, 32)
    enc(audio).sum().backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"no gradient on {name}"
        assert bool(torch.isfinite(param.grad).all()), f"non-finite gradient on {name}"


def test_audio_freeze_encoder_disables_grads():
    enc = AudioSpectrogramEncoder(
        patch_size=8, embed_dim=8, layers=1, num_heads=2, n_frames=32, n_mels=32, freeze_encoder=True
    )
    assert all(not param.requires_grad for param in enc.parameters())


def test_audio_bad_args_raise():
    with pytest.raises(ValueError, match="layers"):
        AudioSpectrogramEncoder(patch_size=8, embed_dim=8, layers=0, num_heads=2, n_frames=32, n_mels=32)
    with pytest.raises(ValueError, match="divisible"):
        AudioSpectrogramEncoder(patch_size=8, embed_dim=8, layers=1, num_heads=2, n_frames=33, n_mels=32)
    with pytest.raises(ValueError, match="not divisible"):
        AudioSpectrogramEncoder(patch_size=8, embed_dim=8, layers=1, num_heads=3, n_frames=32, n_mels=32)


def _config(num_samples: int = 32, use_rope: bool = True):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24, use_rope=use_rope),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_datamodule_audio_frozen_path_produces_audio_token_batch():
    """modality='audio' precomputes frozen audio-token features [B,17,24]."""
    config = _config()
    module = MultimodalDataModule(
        config, modality="audio", audio_frames=32, audio_mels=32, patch_size=8, vit_layers=2, vit_heads=4
    )
    module.prepare_data()
    module.setup()
    assert module.num_modal_tokens == 17
    assert all(not p.requires_grad for p in module.encoder.parameters())  # type: ignore[union-attr]
    loader, _sampler = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert batch["modal_embeds"].shape == (8, 17, 24)
    assert torch.isfinite(batch["modal_embeds"]).all()


def test_datamodule_audio_trainable_e2e_converges():
    """CPU e2e with train_encoder=True: raw spectrograms -> audio tower ->
    fused prefix -> decoder; loss drops AND gradients reach the audio tower."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(num_samples=32)
        module = MultimodalDataModule(
            config,
            modality="audio",
            audio_frames=32,
            audio_mels=32,
            patch_size=8,
            vit_layers=2,
            vit_heads=4,
            train_encoder=True,
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
        assert engine.model.encoder is not None

        losses = [engine._run_epoch(epoch) for epoch in range(20)]
        assert all(loss == loss for loss in losses)
        assert losses[-1] < losses[0] * 0.5, f"audio-fused loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert losses[-1] < 1.7

        # Gradient reaches the audio tower after a probe backward.
        device = next(engine.model.parameters()).device
        loader, _ = module.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        engine.model.zero_grad(set_to_none=True)
        engine.model.train()
        engine.model(batch["input_ids"], modal_samples=batch["modal_samples"]).float().sum().backward()
        encoder = engine.model.encoder
        grads = [(n, p.grad) for n, p in encoder.named_parameters()]
        assert all(g is not None for _, g in grads), [n for n, g in grads if g is None]
        assert all(bool(torch.isfinite(g).all()) for _, g in grads)
    finally:
        torch.set_num_threads(prev)
