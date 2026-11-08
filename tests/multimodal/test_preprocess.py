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


class TestPatchifyShapesAndRoundTrip:
    """`test_patchify_matches_manual_reference` pins the LAYOUT at one square size.

    Two properties it cannot see: (a) non-square images, where the patch grid is
    n_h != n_w and a row/column mix-up stays invisible on 32x32 input; and (b)
    LOSSLESSNESS — a patchify that dropped or duplicated a pixel would still match
    a hand-written reference only if that reference had the same bug, but the
    round trip back to the original image is an independent check.
    """

    @pytest.mark.parametrize(
        ("channels", "height", "width", "patch"),
        [
            (3, 8, 8, 4),  # square
            (2, 6, 12, 3),  # WIDE: n_h=2, n_w=4
            (1, 12, 6, 3),  # TALL: n_h=4, n_w=2
            (3, 32, 32, 16),  # the reference case
        ],
    )
    def test_patch_count_matches_the_grid(self, channels, height, width, patch):
        image = torch.randn(2, channels, height, width)
        patches = patchify(image, patch)

        n_h, n_w = height // patch, width // patch
        assert patches.shape == (2, n_h * n_w, channels * patch * patch), (
            f"grid {n_h}x{n_w} gave {tuple(patches.shape)}"
        )

    @pytest.mark.parametrize(("height", "width", "patch"), [(8, 8, 4), (6, 12, 3), (12, 6, 3)])
    def test_round_trip_recovers_every_pixel(self, height, width, patch):
        """Inverting the permutation must reproduce the image exactly.

        Independent of any hand-written reference: if patchify dropped, duplicated
        or transposed a pixel, the reconstruction cannot equal the input.
        """
        torch.manual_seed(0)
        image = torch.randn(1, 2, height, width)
        patches = patchify(image, patch)

        n_h, n_w = height // patch, width // patch
        channels = image.shape[1]
        rebuilt = patches.reshape(1, n_h, n_w, channels, patch, patch)
        rebuilt = rebuilt.permute(0, 3, 1, 4, 2, 5).reshape(1, channels, height, width)

        assert torch.equal(rebuilt, image), "patchify is not lossless — a pixel was moved, dropped or duplicated"

    def test_one_pixel_patches_cover_the_image(self):
        """`patch_size=1` is the degenerate case: C*1*1 per patch, H*W patches."""
        image = torch.randn(1, 3, 4, 5)
        patches = patchify(image, 1)

        assert patches.shape == (1, 20, 3)
        # Each patch is the per-pixel channel vector, in row-major order.
        assert torch.equal(patches[0, 0], image[0, :, 0, 0])
        assert torch.equal(patches[0, -1], image[0, :, 3, 4])

    def test_a_single_patch_equal_to_the_whole_image(self):
        """`patch_size == H == W` collapses to one patch holding the whole tensor."""
        image = torch.randn(1, 3, 4, 4)
        patches = patchify(image, 4)

        assert patches.shape == (1, 1, 3 * 4 * 4)
        assert torch.equal(patches[0, 0], image[0].reshape(-1))
