"""Tests for the image-to-token preprocessor slice (ROADMAP 12.3)."""

from __future__ import annotations

import pytest
import torch

from llm.multimodal.preprocess import ImagePatchPreprocessor, patchify


def test_patchify_matches_manual_reference():
    torch.manual_seed(0)
    batch, channels, height, width, patch = 2, 3, 32, 32, 16
    image = torch.randn(batch, channels, height, width)
    patches = patchify(image, patch)
    assert patches.shape == (batch, 4, channels * patch * patch)

    manual = torch.stack(
        [
            torch.stack(
                [
                    image[b, :, i : i + patch, j : j + patch].reshape(-1)
                    for i in range(0, height, patch)
                    for j in range(0, width, patch)
                ]
            )
            for b in range(batch)
        ]
    )
    assert torch.allclose(patches, manual, atol=1e-6)


def test_preprocessor_output_shape_and_deterministic():
    torch.manual_seed(1)
    p = ImagePatchPreprocessor(in_channels=3, patch_size=16, embed_dim=8, image_h=32, image_w=32)
    image = torch.randn(2, 3, 32, 32)
    out = p(image)
    assert out.shape == (2, 4, 8)
    assert torch.equal(out, p(image))  # deterministic, no grad required path
    assert torch.isfinite(out).all()


def test_gradient_flows_to_projection_and_pos_embed():
    torch.manual_seed(2)
    p = ImagePatchPreprocessor(in_channels=3, patch_size=16, embed_dim=8, image_h=32, image_w=32)
    image = torch.randn(2, 3, 32, 32)
    p(image).sum().backward()
    assert p.proj.weight.grad is not None
    assert p.pos_embed.grad is not None
    grads = torch.cat([p.proj.weight.grad.reshape(-1), p.pos_embed.grad.reshape(-1)])
    assert bool(torch.isfinite(grads).all())


def test_divisibility_validation():
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.randn(1, 3, 30, 30), 16)
    with pytest.raises(ValueError, match="patch_size"):
        patchify(torch.randn(1, 3, 32, 32), 0)
    with pytest.raises(ValueError, match="divisible"):
        ImagePatchPreprocessor(in_channels=3, patch_size=16, embed_dim=8, image_h=33, image_w=32)


def test_patch_count_mismatch_raises():
    p = ImagePatchPreprocessor(in_channels=3, patch_size=16, embed_dim=8, image_h=32, image_w=32)
    with pytest.raises(ValueError, match="patches"):
        p(torch.randn(1, 3, 64, 64))  # 16 patches but pos_embed has 4 slots
