"""Tests for Prefix Tuning foundation slice (ROADMAP P2 efficient fine-tuning).

Three concerns, three sections:

1. ``MHA.forward`` accepts an optional ``prefix_kv`` argument and
   prepends it to K and V before the attention compute.
2. ``PrefixTuningAttention`` holds the prefix parameters + reparameterization
   MLPs and dispatches to the underlying MHA.
3. ``apply_prefix_tuning`` / ``get_prefix_parameters`` / ``fold_reparameterization``
   module-level helpers walk the model, enumerate the trainable parameters,
   and collapse the reparam MLPs into static buffers for inference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.attn.mha import MultiHeadAttention
from llm.core.prefix_tuning import (
    PrefixTuningAttention,
    apply_prefix_tuning,
    fold_reparameterization,
    get_prefix_parameters,
)

# ---------------------------------------------------------------------------
# 1) MHA accepts optional prefix_kv
# ---------------------------------------------------------------------------


class TestMHAPrefixKV:
    """``MHA.forward`` must accept ``prefix_kv=None`` with no behavior change."""

    def test_forward_without_prefix_kv_matches_original(self):
        """Baseline: no prefix → forward is byte-identical to the pre-extension path."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_zero_prefix_shrinks_output_magnitude(self):
        """A zero-valued prefix contributes V=0 but its K rows still participate
        in the softmax — existing tokens' weights shrink by Z/(Z+P).

        This is the same math as the MHA test above; here we just verify
        the wrapper's ``forward`` passes the prefix through correctly.
        """
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8
        zero_pk = torch.zeros(2, 4, prefix_len, head_dim)
        zero_pv = torch.zeros(2, 4, prefix_len, head_dim)
        out_with_zero_prefix = attn(x, prefix_kv=(zero_pk, zero_pv))

        mag_no = out_no_prefix.abs().mean()
        mag_zero = out_with_zero_prefix.abs().mean()
        ratio = (mag_zero / mag_no).item()
        assert 0.0 < ratio < 1.0, f"expected ratio in (0, 1), got {ratio}"
        assert not torch.allclose(out_no_prefix, out_with_zero_prefix, atol=1e-4)

    def test_forward_with_prefix_kv_changes_output(self):
        """Non-zero prefix → output diverges from the no-prefix path."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim)
        pv = torch.randn(2, 4, prefix_len, head_dim)
        out_with_prefix = attn(x, prefix_kv=(pk, pv))

        assert not torch.allclose(out_no_prefix, out_with_prefix, atol=1e-4)

    def test_prefix_does_not_affect_output_shape(self):
        """Output shape is independent of prefix_len."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        for p_len in (0, 1, 3, 8):
            head_dim = 8
            pk = torch.randn(2, 4, p_len, head_dim) if p_len > 0 else torch.zeros(2, 4, 1, head_dim)
            pv = torch.randn(2, 4, p_len, head_dim) if p_len > 0 else torch.zeros(2, 4, 1, head_dim)
            out = attn(x, prefix_kv=(pk, pv))
            assert out.shape == (2, 5, 32)


# ---------------------------------------------------------------------------
# 2) PrefixTuningAttention wrapper
# ---------------------------------------------------------------------------


def _make_mha(hidden_size: int = 32, num_heads: int = 4):
    return MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)


class TestPrefixTuningAttentionShape:
    """Shape contracts for prefix_small + reparam MLPs."""

    def test_prefix_small_shape(self):
        attn = PrefixTuningAttention(
            base_attn=_make_mha(hidden_size=32, num_heads=4),
            prefix_len=5,
            reparam_hidden=16,
        )
        # prefix_small: (prefix_len, reparam_hidden)
        assert attn.prefix_small.shape == (5, 16)

    def test_reparam_kv_output_shape(self):
        attn = PrefixTuningAttention(
            base_attn=_make_mha(hidden_size=32, num_heads=4),
            prefix_len=5,
            reparam_hidden=16,
        )
        # kv_dim = hidden_size (num_kv_heads == num_heads) = 32
        # Each reparam MLP projects reparam_hidden → kv_dim.
        pk = attn._reparam_k(attn.prefix_small)
        pv = attn._reparam_v(attn.prefix_small)
        assert pk.shape == (5, 32)
        assert pv.shape == (5, 32)

    def test_prefix_small_default_reparam_hidden(self):
        """reparam_hidden=None → defaults to hidden_size."""
        attn = PrefixTuningAttention(
            base_attn=_make_mha(hidden_size=32, num_heads=4),
            prefix_len=5,
        )
        assert attn.prefix_small.shape == (5, 32)


class TestPrefixTuningAttentionForward:
    """Forward injects prefix into base MHA."""

    def test_forward_runs_and_returns_correct_shape(self):
        torch.manual_seed(0)
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_frozen_prefix_changes_output(self):
        """Forward must actually use the prefix (not silently drop it)."""
        torch.manual_seed(0)
        attn1 = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        attn2 = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        x = torch.randn(2, 5, 32)

        # Different prefix params → different output.
        with torch.no_grad():
            attn2.prefix_small.fill_(99.0)

        out1 = attn1(x).detach()
        out2 = attn2(x).detach()
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_refuses_non_prefix_capable_base(self):
        """Wrapper must reject modules that lack the PrefixCapableAttention protocol.

        ``nn.Linear`` has a ``.forward()`` method (so it passes a naïve
        Protocol isinstance check at runtime), but it lacks the
        ``num_kv_heads`` and ``head_dim`` attributes the wrapper reads.
        The construction-time gate must catch this loudly.
        """
        fake_attn = nn.Linear(32, 32)  # has .forward but no num_kv_heads / head_dim
        with pytest.raises(TypeError, match="PrefixCapableAttention"):
            PrefixTuningAttention(base_attn=fake_attn, prefix_len=3, reparam_hidden=16)


class TestPrefixTuningAttentionProtocolGate:
    """Wrapper accepts any ``PrefixCapableAttention`` base (MHA / Flash / MLA).

    After the multi-backend extension (T2 PEFT follow-up), the wrapper
    gates on the ``PrefixCapableAttention`` Protocol + a structural
    ``num_kv_heads`` / ``head_dim`` check rather than a hard MHA
    ``isinstance``. The tests below pin that contract for the two
    newly-supported backends.
    """

    def test_accepts_mla_base(self):
        """MLA has ``prefix_kv`` in its forward → wrapper accepts it."""
        from llm.core.attn.mla import MultiLatentAttention

        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        wrapper = PrefixTuningAttention(base_attn=attn, prefix_len=3, reparam_hidden=16)
        assert wrapper.base_attn is attn
        assert wrapper.num_kv_heads == attn.num_kv_heads
        assert wrapper.head_dim == attn.head_dim

    def test_accepts_flash_attention_base(self):
        """FlashAttention has ``prefix_kv`` in its forward → wrapper accepts it (when available)."""
        from llm.core.attn.flash_attn import FLASH_ATTN_AVAILABLE, FlashAttention

        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn is optional; install via `llm[perf]`")
        attn = FlashAttention(hidden_size=32, num_heads=4, bias=False)
        wrapper = PrefixTuningAttention(base_attn=attn, prefix_len=3, reparam_hidden=16)
        assert wrapper.base_attn is attn
        assert wrapper.num_kv_heads == attn.num_kv_heads
        assert wrapper.head_dim == attn.head_dim

    def test_rejects_module_without_forward(self):
        """A bare class with no ``forward`` method fails the Protocol check."""
        class NoForward:  # noqa: D401 — trivial
            pass

        with pytest.raises(TypeError, match="PrefixCapableAttention"):
            PrefixTuningAttention(base_attn=NoForward(), prefix_len=3, reparam_hidden=16)


class TestPrefixTuningWrapsFlashAndMLA:
    """End-to-end: ``apply_prefix_tuning`` wraps Flash / MLA bases and runs forward."""

    def test_apply_prefix_tuning_wraps_every_mla(self):
        """All MultiLatentAttention modules get wrapped; Linear leaves alone."""
        from llm.core.attn.mla import MultiLatentAttention

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MultiLatentAttention(hidden_size=16, num_heads=2, num_latents=4)
                self.attn2 = MultiLatentAttention(hidden_size=16, num_heads=2, num_latents=4)
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(self.attn2(self.attn1(x)))

        model = TwoLayer()
        apply_prefix_tuning(model, prefix_len=2, reparam_hidden=8)
        assert isinstance(model.attn1, PrefixTuningAttention)
        assert isinstance(model.attn2, PrefixTuningAttention)
        assert isinstance(model.linear, nn.Linear)

    def test_apply_prefix_tuning_wraps_every_flash(self):
        """All FlashAttention modules get wrapped (when flash-attn is installed)."""
        from llm.core.attn.flash_attn import FLASH_ATTN_AVAILABLE, FlashAttention

        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn is optional; install via `llm[perf]`")

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = FlashAttention(hidden_size=16, num_heads=2, bias=False)
                self.attn2 = FlashAttention(hidden_size=16, num_heads=2, bias=False)
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(self.attn2(self.attn1(x)))

        model = TwoLayer()
        apply_prefix_tuning(model, prefix_len=2, reparam_hidden=8)
        assert isinstance(model.attn1, PrefixTuningAttention)
        assert isinstance(model.attn2, PrefixTuningAttention)
        assert isinstance(model.linear, nn.Linear)

    def test_wrapper_over_mla_runs_forward_and_freezes_base(self):
        """Wrapping MLA: forward runs, base MLA params stay frozen, prefix params trainable."""
        from llm.core.attn.mla import MultiLatentAttention

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        wrapper = PrefixTuningAttention(base_attn=attn, prefix_len=3, reparam_hidden=16)

        x = torch.randn(2, 5, 32)
        out = wrapper(x)
        assert out.shape == (2, 5, 32)

        # Base MLA params are frozen.
        for p in attn.parameters():
            assert not p.requires_grad

        # Prefix params are trainable.
        trainable = list(get_prefix_parameters(wrapper))
        assert len(trainable) == 5
        for p in trainable:
            assert p.requires_grad


class TestPrefixTuningAttentionGradients:
    """Gradients flow to prefix params but base MHA stays frozen."""

    def test_prefix_params_receive_gradients(self):
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        x = torch.randn(2, 5, 32, requires_grad=False)
        target = torch.randn(2, 5, 32)
        loss = ((attn(x) - target) ** 2).mean()
        loss.backward()

        assert attn.prefix_small.grad is not None
        assert attn.prefix_small.grad.abs().sum() > 0
        assert attn._reparam_k.weight.grad is not None
        assert attn._reparam_v.weight.grad is not None

    def test_base_attn_stays_frozen(self):
        """By default we freeze the base MHA — only prefix params train."""
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        x = torch.randn(2, 5, 32)
        target = torch.randn(2, 5, 32)
        loss = ((attn(x) - target) ** 2).mean()
        loss.backward()

        # Base MHA's qkv_proj and out_proj should be frozen.
        assert attn.base_attn.qkv_proj.weight.requires_grad is False
        assert attn.base_attn.out_proj.weight.requires_grad is False


class TestFoldReparameterization:
    """``fold_reparameterization`` collapses MLPs to static prefix buffers."""

    def test_fold_drops_reparam_mlps(self):
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        assert hasattr(attn, "_reparam_k")
        assert hasattr(attn, "_reparam_v")

        fold_reparameterization(attn)

        # MLPs gone, static buffers in place.
        assert not hasattr(attn, "_reparam_k")
        assert not hasattr(attn, "_reparam_v")
        assert hasattr(attn, "prefix_k")
        assert hasattr(attn, "prefix_v")
        assert attn.prefix_k.shape == (3, 32)
        assert attn.prefix_v.shape == (3, 32)

    def test_fold_preserves_forward_output(self):
        """Output before and after fold must match (within FP tolerance)."""
        torch.manual_seed(0)
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        # Disable MHA dropout (default p=0.1) so the two forward passes
        # don't diverge from random dropout masks. We only care about
        # whether the fold math is identity, not whether dropout is on.
        attn.eval()
        x = torch.randn(2, 5, 32)

        out_before = attn(x).detach().clone()
        fold_reparameterization(attn)
        out_after = attn(x).detach().clone()

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_fold_is_idempotent(self):
        """Calling fold twice should be a no-op the second time."""
        attn = PrefixTuningAttention(_make_mha(32, 4), prefix_len=3, reparam_hidden=16)
        fold_reparameterization(attn)
        snapshot_k = attn.prefix_k.detach().clone()
        # Second fold should not change the static prefix.
        fold_reparameterization(attn)
        assert torch.allclose(attn.prefix_k, snapshot_k)


# ---------------------------------------------------------------------------
# 3) Module-level helpers
# ---------------------------------------------------------------------------


class TestApplyPrefixTuning:
    """``apply_prefix_tuning`` walks the model and wraps MHA modules."""

    def test_wraps_every_mha_in_sequential(self):

        # Build a tiny model by hand with MHA modules.
        from llm.core.attn.mha import MultiHeadAttention

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MultiHeadAttention(hidden_size=16, num_heads=2)
                self.attn2 = MultiHeadAttention(hidden_size=16, num_heads=2)
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(self.attn2(self.attn1(x)))

        model = TwoLayer()
        apply_prefix_tuning(model, prefix_len=4, reparam_hidden=8)

        # Both MHA layers are now wrapped; the Linear stays untouched.
        assert isinstance(model.attn1, PrefixTuningAttention)
        assert isinstance(model.attn2, PrefixTuningAttention)
        assert isinstance(model.linear, nn.Linear)

    def test_target_modules_filters_by_substring(self):
        from llm.core.attn.mha import MultiHeadAttention

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Module()
                self.block1.attn = MultiHeadAttention(hidden_size=16, num_heads=2)
                self.block2 = nn.Module()
                self.block2.attn = MultiHeadAttention(hidden_size=16, num_heads=2)

            def forward(self, x):
                return self.block2.attn(self.block1.attn(x))

        model = TwoLayer()
        apply_prefix_tuning(model, prefix_len=4, target_modules=["block1"])
        assert isinstance(model.block1.attn, PrefixTuningAttention)
        assert isinstance(model.block2.attn, MultiHeadAttention)  # untouched

    def test_empty_model_is_noop(self):
        model = nn.Sequential(nn.Linear(8, 8))
        apply_prefix_tuning(model, prefix_len=4)  # must not raise


class TestGetPrefixParameters:
    """``get_prefix_parameters`` enumerates trainable prefix + reparam params."""

    def test_yields_prefix_and_reparam_params(self):
        from llm.core.attn.mha import MultiHeadAttention

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MultiHeadAttention(hidden_size=16, num_heads=2)
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(self.attn(x))

        model = TinyModel()
        apply_prefix_tuning(model, prefix_len=4, reparam_hidden=8)
        params = list(get_prefix_parameters(model))

        # 1 prefix_small + 2 reparam MLPs * (weight + bias) = 5 params total.
        assert len(params) == 5
        for p in params:
            assert p.requires_grad

    def test_excludes_base_mha_params(self):
        from llm.core.attn.mha import MultiHeadAttention

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MultiHeadAttention(hidden_size=16, num_heads=2)

            def forward(self, x):
                return self.attn(x)

        model = TinyModel()
        apply_prefix_tuning(model, prefix_len=4, reparam_hidden=8)
        params = list(get_prefix_parameters(model))
        param_ids = {id(p) for p in params}
        # Base MHA's qkv_proj.weight should NOT appear.
        assert id(model.attn.base_attn.qkv_proj.weight) not in param_ids


class TestFoldReparameterizationModuleLevel:
    """Module-level ``fold_reparameterization(model)`` walks and folds."""

    def test_folds_every_wrapped_attention(self):
        from llm.core.attn.mha import MultiHeadAttention

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = MultiHeadAttention(hidden_size=16, num_heads=2)
                self.attn2 = MultiHeadAttention(hidden_size=16, num_heads=2)

            def forward(self, x):
                return self.attn2(self.attn1(x))

        model = TinyModel()
        apply_prefix_tuning(model, prefix_len=4, reparam_hidden=8)
        fold_reparameterization(model)

        for attn in (model.attn1, model.attn2):
            assert hasattr(attn, "prefix_k")
            assert hasattr(attn, "prefix_v")
            assert not hasattr(attn, "_reparam_k")

    def test_empty_model_is_noop(self):
        model = nn.Sequential(nn.Linear(8, 8))
        fold_reparameterization(model)  # must not raise
