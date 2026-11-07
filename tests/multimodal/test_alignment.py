"""Tests for the CLIP/SigLIP-style contrastive image-text alignment module (ROADMAP 12.1).

Covers shape/validation invariants, the initial ``log(B)`` loss, and a CPU
training e2e where symmetric InfoNCE (and its SigLIP variant) align paired
image/text embeddings and retrieval accuracy reaches 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm.multimodal import ContrastiveAligner, VisionTransformerEncoder, contrastive_loss


def _paired(batch: int = 8, tokens: int = 5, dim: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic paired image tokens / text tokens with a shared latent per sample.

    Each sample's image and text rows both get a deterministic per-sample mean
    so retrieval is learnable and unambiguous.
    """
    torch.manual_seed(seed)
    idx = torch.arange(batch)
    image = torch.randn(batch, tokens, dim)
    text = torch.randn(batch, tokens, dim)
    # Add a shared per-sample signal on top of random noise.
    image = image + (idx.float()[:, None, None] * 0.5)
    text = text + (idx.float()[:, None, None] * 0.5)
    return image, text


def test_contrastive_loss_logits_square_and_uniform_baseline():
    torch.manual_seed(0)
    aligner = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8)
    image, text = _paired()
    logits = aligner(image, text)
    assert logits.shape == (8, 8)
    loss = contrastive_loss(logits)
    assert torch.isfinite(loss)
    # Random, unaligned input near the default temperature gives the uniform
    # baseline near log(B) (batch entropy): a wide projection + gentle scale
    # keeps random cosine logits near 0 so no class dominates.
    torch.manual_seed(1)
    flat = ContrastiveAligner(image_dim=16, text_dim=16, init_logit_scale=1.0)
    random_img = torch.randn(8, 5, 16)
    random_txt = torch.randn(8, 5, 16)
    baseline = contrastive_loss(flat(random_img, random_txt)).detach().item()
    assert baseline == pytest.approx(math.log(8), abs=0.15)


def test_contrastive_loss_validation():
    with pytest.raises(ValueError, match="square"):
        contrastive_loss(torch.randn(3, 4))
    with pytest.raises(ValueError, match="image_pool"):
        ContrastiveAligner(image_dim=16, text_dim=16, image_pool="nope")
    with pytest.raises(ValueError, match="must all be > 0"):
        ContrastiveAligner(image_dim=0, text_dim=16)


def test_aligner_cls_pool_uses_first_row():
    torch.manual_seed(0)
    mean = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8, image_pool="mean")
    cls = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8, image_pool="cls")
    image, text = _paired()
    # CLS pool reads only row 0; mean pool averages all tokens — for random
    # token embeddings those differ, and both pooling modes stay callable.
    with torch.no_grad():
        pooled_mean = mean._pool(image, image=True)
        pooled_cls = cls._pool(image, image=True)
    assert tuple(pooled_mean.shape) == (8, 16)
    assert tuple(pooled_cls.shape) == (8, 16)
    assert not torch.allclose(pooled_mean, pooled_cls)
    assert mean(image, text).shape == (8, 8)
    assert cls(image, text).shape == (8, 8)


def _train_aligner(aligner, steps: int = 200, lr: float = 1e-2, seed: int = 0) -> tuple[float, float]:
    optimizer = torch.optim.Adam(aligner.parameters(), lr=lr)
    loss = math.inf
    for _step in range(steps):
        image, text = _paired(seed=seed)
        optimizer.zero_grad()
        loss = aligner.loss(image, text)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = aligner(image, text)
        hits = (logits.argmax(-1) == torch.arange(logits.size(0))).float().mean().item()
    return loss.detach().item(), hits


def test_aligner_info_nce_training_retrieval_reaches_one():
    """Symmetric InfoNCE over random paired data: loss drops and image->text
    top-1 retrieval rises to 1 — the alignment is actually learned."""
    torch.manual_seed(0)
    aligner = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8)
    loss, acc = _train_aligner(aligner)
    assert loss < math.log(8) * 0.6, f"InfoNCE loss did not drop: {loss:.3f}"
    assert acc > 0.9, f"retrieval accuracy too low: {acc:.3f}"


def test_aligner_siglip_variant_training_retrieval_reaches_one():
    torch.manual_seed(1)
    aligner = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8, sigmoid=True)
    loss, acc = _train_aligner(aligner)
    assert torch.isfinite(torch.tensor(loss))
    assert acc > 0.9, f"SigLIP retrieval accuracy too low: {acc:.3f}"


def test_aligner_composes_with_vision_tower_output():
    """The aligner consumes VisionTransformerEncoder output directly: raw
    images -> pooled image tokens [B, N, D] + text embs [B, T, D] align."""
    torch.manual_seed(2)
    enc = VisionTransformerEncoder(embed_dim=8, layers=1, num_heads=2, patch_size=16, image_h=32, image_w=32)
    aligner = ContrastiveAligner(image_dim=8, text_dim=8, proj_dim=8)
    img = torch.randn(4, 3, 32, 32)  # real [B, C, H, W] image for the tower
    text, _ = _paired(batch=4, tokens=5, dim=8)
    image_tokens = enc(img)  # [4, 5, 8]
    assert image_tokens.shape == (4, 5, 8)
    loss = aligner.loss(image_tokens, text)
    assert torch.isfinite(loss)
    assert loss.item() > 0


# --- Numerical stability of the learnable temperature ---


def test_logit_scale_clamps_and_stays_finite_at_extremes():
    """A trained/desynced temperature must never produce NaN or inf.

    The scale is a LEARNABLE parameter, so nothing bounds it during training:
    an unpinned ``exp(logit_scale)`` can overflow to inf (logits inf, CE NaN) or
    underflow to 0 (all logits zero). The module clamps the scale at 100; this
    pins the behaviour at both ends, including the uniform-loss baseline that
    the underflow case must degrade to gracefully.
    """
    import math

    from llm.multimodal import ContrastiveAligner, contrastive_loss

    aligner = ContrastiveAligner(image_dim=16, text_dim=16, proj_dim=8)
    images, texts = torch.randn(4, 16), torch.randn(4, 16)

    for logit_scale in (-20.0, -100.0, 100.0, 1e6):
        with torch.no_grad():
            aligner.logit_scale.fill_(logit_scale)
            logits = aligner(images, texts)
        assert torch.isfinite(logits).all(), f"logits not finite at logit_scale={logit_scale}"
        loss = contrastive_loss(logits)
        assert torch.isfinite(loss), f"loss not finite at logit_scale={logit_scale}"

    # Underflow to scale==0 gives a uniform similarity matrix, whose InfoNCE is
    # exactly ln(B) — a useful invariant, and the graceful degradation path.
    with torch.no_grad():
        aligner.logit_scale.fill_(-100.0)
        uniform_loss = contrastive_loss(aligner(images, texts))
    assert math.isclose(float(uniform_loss), math.log(4), rel_tol=1e-5)


def test_siglip_loss_prefers_matched_pairs_at_realistic_temperature():
    """Matched pairs must score lower than random ones at a usable scale.

    SigLIP is a per-pair binary loss, so it lacks InfoNCE's sharp diagonal
    contrast; the property only holds once the similarity scale is sane. At an
    extreme scale (e.g. 1/0.07 on unnormalized random features) the negative
    pairs dominate and the ordering can invert — documenting the realistic
    regime here keeps the assertion honest.
    """
    from llm.multimodal import contrastive_loss

    def logits(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return a @ b.T / temperature

    torch.manual_seed(0)
    random_a, random_b = torch.randn(8, 16), torch.randn(8, 16)
    matched = torch.randn(8, 16)

    for temperature in (1.0, 4.0):
        random_loss = float(contrastive_loss(logits(random_a, random_b, temperature), sigmoid=True))
        matched_loss = float(contrastive_loss(logits(matched, matched.clone(), temperature), sigmoid=True))
        assert matched_loss < random_loss, (
            f"at temperature={temperature} SigLIP scored matched pairs ({matched_loss:.4f}) "
            f"worse than random ones ({random_loss:.4f})"
        )
