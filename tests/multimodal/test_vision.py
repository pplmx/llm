"""Tests for the CLIP/SigLIP-style vision encoder + raw-image data path (ROADMAP 12.1).

Covers registry registration, shape/determinism invariants, exact-math parity
against a handwritten reference, gradient flow, the frozen-tower option, and the
``MultimodalDataModule``/``MultimodalTask`` wiring (raw images -> image tokens ->
fused decoder prefix) end-to-end on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from llm.multimodal import (
    MODALITY_ENCODER_REGISTRY,
    MultimodalDataModule,
    MultimodalModel,
    VisionTransformerEncoder,
)
from llm.multimodal.preprocess import patchify


def _image(batch: int = 2, size: int = 32) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 3, size, size)


def test_vit_registered_in_modality_registry():
    encoder_cls = MODALITY_ENCODER_REGISTRY.get("vit")
    assert encoder_cls is VisionTransformerEncoder
    enc = encoder_cls(embed_dim=8, layers=1, num_heads=2, patch_size=16, image_h=32, image_w=32)
    assert enc.modality == "vit"


def test_vit_encode_shape_with_and_without_cls():
    torch.manual_seed(1)
    with_cls = VisionTransformerEncoder(
        embed_dim=8, layers=2, num_heads=2, patch_size=16, image_h=32, image_w=32, with_cls=True
    )
    without_cls = VisionTransformerEncoder(
        embed_dim=8, layers=2, num_heads=2, patch_size=16, image_h=32, image_w=32, with_cls=False
    )
    img = _image(3)
    assert with_cls.encode(img).shape == (3, 5, 8)  # 4 patches + CLS
    assert without_cls.encode(img).shape == (3, 4, 8)
    assert with_cls.num_tokens == 5
    assert without_cls.num_tokens == 4
    assert "models.decoder" not in VisionTransformerEncoder.__module__  # ADR-013: standalone


def test_vit_encode_deterministic_and_pos_sensitive():
    torch.manual_seed(2)
    enc = VisionTransformerEncoder(embed_dim=8, layers=2, num_heads=2, patch_size=16, image_h=32, image_w=32)
    enc.eval()
    img = _image(2)
    out = enc.encode(img)
    assert torch.equal(out, enc.encode(img))  # deterministic
    assert torch.isfinite(out).all()
    # Rolling the image circularly must change the output: patches land at
    # different positions, so the learned positional embedding is not degenerate.
    shifted = torch.roll(img, shifts=(8, 8), dims=(2, 3))
    assert not torch.allclose(out, enc.encode(shifted), atol=1e-5)


def test_vit_matches_manual_reference():
    """Exact-math parity: a 1-block/2-head encoder equals a step-by-step reference
    re-derived from its own parameters (patchify+proj+pos+CLS, then a pre-norm
    SDPA block, then final LayerNorm)."""
    torch.manual_seed(0)
    enc = VisionTransformerEncoder(
        embed_dim=8,
        layers=1,
        num_heads=2,
        patch_size=16,
        image_h=32,
        image_w=32,
        mlp_ratio=2.0,
        with_cls=True,
    )
    enc.eval()
    img = _image(2)

    # --- reference: patch embedding + CLS (read params from the encoder) ---
    patches = patchify(img, 16)
    tokens = enc.patch_embed.proj(patches) + enc.patch_embed.pos_embed
    cls = enc.cls_token.expand(img.shape[0], -1, -1)
    x = torch.cat([cls, tokens], dim=1)

    # --- reference: pre-norm block ---
    blk = enc.blocks[0]
    residual = x
    h = blk.norm1(x)
    batch, length, dim = h.shape
    head_dim = dim // 2
    q = blk.q(h).view(batch, length, 2, head_dim).transpose(1, 2)
    k = blk.k(h).view(batch, length, 2, head_dim).transpose(1, 2)
    v = blk.v(h).view(batch, length, 2, head_dim).transpose(1, 2)
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = functional.softmax(attn, dim=-1)
    ctx = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, dim)
    h1 = residual + blk.out_proj(ctx)

    residual = h1
    h = blk.norm2(h1)
    h2 = residual + blk.fc2(blk.act(blk.fc1(h)))

    expected = enc.norm(h2)
    assert torch.allclose(enc.encode(img), expected, atol=1e-4)


def test_vit_gradient_flows_to_all_trainable_parts():
    torch.manual_seed(3)
    enc = VisionTransformerEncoder(
        embed_dim=8,
        layers=2,
        num_heads=2,
        patch_size=16,
        image_h=32,
        image_w=32,
        mlp_ratio=2.0,
        with_cls=True,
    )
    enc.train()
    img = _image(2)
    enc(img).sum().backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"no gradient on {name}"
        assert bool(torch.isfinite(param.grad).all()), f"non-finite gradient on {name}"


def test_vit_freeze_encoder_disables_grads():
    enc = VisionTransformerEncoder(
        embed_dim=8, layers=1, num_heads=2, patch_size=16, image_h=32, image_w=32, freeze_encoder=True
    )
    assert all(not param.requires_grad for param in enc.parameters())


def test_vit_bad_args_raise():
    with pytest.raises(ValueError, match="layers"):
        VisionTransformerEncoder(embed_dim=8, layers=0, num_heads=2, patch_size=16, image_h=32, image_w=32)
    with pytest.raises(ValueError, match="not divisible"):
        VisionTransformerEncoder(embed_dim=8, layers=1, num_heads=3, patch_size=16, image_h=32, image_w=32)
    with pytest.raises(ValueError, match="divisible"):
        VisionTransformerEncoder(embed_dim=8, layers=1, num_heads=2, patch_size=16, image_h=30, image_w=30)


def test_vit_encoder_rejects_nontrainable_autograd_path():
    """The encoder must be usable under ``torch.no_grad()`` (data-module setup
    precomputes features) without raising or leaking train-state."""
    enc = VisionTransformerEncoder(embed_dim=8, layers=1, num_heads=2, patch_size=16, image_h=32, image_w=32)
    img = _image(2)
    with torch.no_grad():
        out = enc(img)
    assert out.shape == (2, 5, 8)


def _config(num_samples: int = 32, use_rope: bool = True):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24, use_rope=use_rope),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_datamodule_vit_path_produces_image_token_batch():
    """modal_embeds is [B, num_tokens, embed_dim] (16 patches + CLS) — 3D, unlike
    the 2D linear path — and feeds MultimodalModel's prefix fusion directly."""
    config = _config()
    module = MultimodalDataModule(
        config, modality="vit", image_h=64, image_w=64, patch_size=16, vit_layers=4, vit_heads=4, with_cls=True
    )
    module.prepare_data()
    module.setup()
    assert module.num_modal_tokens == 17  # 16 patches (+ CLS)
    # Vision tower is frozen (CLIP-style features precomputed at setup).
    assert all(not p.requires_grad for p in module.encoder.parameters())  # type: ignore[union-attr]

    loader, _sampler = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels", "modal_embeds"}
    assert batch["input_ids"].shape == (8, 24)
    embeds = batch["modal_embeds"]
    assert embeds.shape == (8, 17, 24)
    assert torch.isfinite(embeds).all()


def test_datamodule_vit_instruction_masks_prompt():
    """vit_instruction_len>0: batch is [instruction | response] with -100 labels
    on the instruction prefix, so the shift-based CE only supervises response."""
    config = _config()
    module = MultimodalDataModule(
        config,
        modality="vit",
        image_h=64,
        image_w=64,
        patch_size=16,
        vit_layers=2,
        vit_heads=4,
        vit_instruction_len=8,
    )
    module.prepare_data()
    module.setup()
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (8, 24)
    assert batch["labels"].shape == (8, 24)
    inst = 8
    assert (batch["labels"][:, :inst] == -100).all()
    assert (batch["labels"][:, inst:] != -100).all()
    assert (batch["input_ids"][:, :inst] >= 1).all()  # instruction tokens
    assert (batch["input_ids"][:, inst:] >= 0).all()


def test_datamodule_vit_instruction_e2e_learns_response():
    """Visual Instruction Tuning CPU e2e (frozen-tower): the masked response
    objective learns — loss drops and next-token accuracy on the response
    region (excluding the -100 prompt) rises, proving VIT-style conditioning."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    inst = 8
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(num_samples=32)
        module = MultimodalDataModule(
            config,
            modality="vit",
            image_h=64,
            image_w=64,
            patch_size=16,
            vit_layers=2,
            vit_heads=4,
            with_cls=True,
            vit_instruction_len=inst,
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        losses = [engine._run_epoch(epoch) for epoch in range(20)]
        assert all(loss == loss for loss in losses)
        assert losses[-1] < losses[0] * 0.5, f"VIT loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert losses[-1] < 1.7

        # Response-region next-token accuracy (instruction positions masked).
        device = next(engine.model.parameters()).device
        loader, _ = module.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        engine.model.eval()
        with torch.no_grad():
            logits = engine.model(batch["input_ids"], batch["modal_embeds"])
        pred = logits.argmax(-1)  # [B, T, V] -> [B, T]
        shift_pred = pred[:, :-1]
        shift_lab = batch["labels"][:, 1:]
        mask = shift_lab != -100
        assert mask.any()
        acc = (shift_pred[mask] == shift_lab[mask]).float().mean().item()
        assert acc > 0.7, f"VIT response accuracy too low: {acc:.3f}"
    finally:
        torch.set_num_threads(prev)


def test_datamodule_vit_trainable_batch_has_raw_images():
    """train_encoder=True: the batch carries RAW images [B,3,H,W] (not
    precomputed modal_embeds) and the built encoder is NOT frozen."""
    config = _config()
    module = MultimodalDataModule(
        config,
        modality="vit",
        image_h=64,
        image_w=64,
        patch_size=16,
        vit_layers=2,
        vit_heads=4,
        with_cls=True,
        train_encoder=True,
    )
    module.prepare_data()
    module.setup()
    assert module.num_modal_tokens is not None
    # Trainable tower: requires_grad stays on.
    assert any(p.requires_grad for p in module.encoder.parameters())  # type: ignore[union-attr]
    loader, _sampler = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels", "images"}
    assert batch["images"].shape == (8, 3, 64, 64)
    assert torch.isfinite(batch["images"]).all()


def test_datamodule_vit_trainable_e2e_trains_vision_tower():
    """CPU e2e with train_encoder=True: loss drops AND gradients flow through
    the vision tower (image-text alignment is learnable, not frozen-feature
    only)."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(num_samples=32)
        module = MultimodalDataModule(
            config,
            modality="vit",
            image_h=64,
            image_w=64,
            patch_size=16,
            vit_layers=2,
            vit_heads=4,
            with_cls=True,
            train_encoder=True,
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
        assert engine.model.encoder is not None

        losses = [engine._run_epoch(epoch) for epoch in range(20)]
        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0] * 0.5, f"image-fused loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert losses[-1] < 1.5

        # Gradient reaches the vision tower: after a probe backward, patch proj
        # / pos embed / CLS / block weights all carry gradients.
        device = next(engine.model.parameters()).device
        loader, _ = module.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        engine.model.zero_grad(set_to_none=True)
        engine.model.train()
        engine.model(batch["input_ids"], images=batch["images"]).float().sum().backward()
        encoder = engine.model.encoder
        grads = [(n, p.grad) for n, p in encoder.named_parameters()]
        assert all(g is not None for _, g in grads), [n for n, g in grads if g is None]
        assert all(bool(torch.isfinite(g).all()) for _, g in grads)
    finally:
        torch.set_num_threads(prev)


def test_datamodule_vit_e2e_fused_training_converges():
    """CPU e2e: images -> vision tower -> fused prefix -> decoder; loss drops and
    text next-token accuracy improves, proving the raw-image path is wired through
    the real engine."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config(num_samples=32)
        module = MultimodalDataModule(
            config, modality="vit", image_h=64, image_w=64, patch_size=16, vit_layers=2, vit_heads=4, with_cls=True
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        losses = [engine._run_epoch(epoch) for epoch in range(20)]
        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0] * 0.5, f"image-fused loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert losses[-1] < 1.5

        model = engine.model
        assert isinstance(model, MultimodalModel)
        # Accuracy probe must move the loader batch to the model's device — the
        # engine does that inside train_step, but this standalone eval does not
        # (CPU batch vs CUDA model would raise, e.g. on GPU machines).
        device = next(model.parameters()).device
        loader, _ = module.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model.eval()
        with torch.no_grad():
            logits = model(batch["input_ids"], batch["modal_embeds"])
        pred = logits.argmax(-1)
        acc = (pred[:, :-1] == batch["labels"][:, 1:]).float().mean().item()
        assert acc > 0.7, f"image-fused text accuracy too low: {acc:.3f}"
    finally:
        torch.set_num_threads(prev)
