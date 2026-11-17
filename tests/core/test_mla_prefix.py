"""Tests for Multi-Latent Attention's ``prefix_kv`` extension (T2 PEFT multi-backend).

``MultiLatentAttention.forward`` must accept an optional ``prefix_kv``
argument and prepend it to K and V before ``_latent_attention`` runs. The
full Prefix Tuning wrapper is tested in ``test_prefix_tuning.py``.

MLA-specific concerns covered here (beyond the MHA contract):

1. ``attn_mask`` S_k axis grows by ``prefix_len`` ones — the prefix is
   unconditionally visible regardless of the input mask.
2. Paged cache + prefix runs through the same concat-after-gather path
   as the linear cache; the prefix never enters the paged cache.
3. MLA has no GQA (``num_heads == num_kv_heads``), so the GQA-repeat
   test from MHA is omitted.
"""

from __future__ import annotations

import pytest
import torch

from llm.core.attn.mla import MultiLatentAttention
from tests.support.devices import DEFAULT_DEVICE

DEVICE_STR = str(DEFAULT_DEVICE)


class TestMLAPrefixKV:
    """``MultiLatentAttention.forward`` must accept ``prefix_kv=None`` with no behavior change."""

    def test_forward_without_prefix_kv_matches_original(self):
        """Baseline: no prefix → forward shape matches input."""
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_zero_prefix_shrinks_output_magnitude(self):
        """Zero-valued V prefix → output magnitude shrinks (prefix rows participate in latent softmax).

        The latent queries attend to (prefix + dynamic) K, V. With V=0,
        the prefix rows still consume softmax mass from the dynamic
        tokens, so the latent output magnitude shrinks. Verifies the
        prefix rows are actually being concatenated into the latent
        attention.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8  # 32 / 4
        zero_pk = torch.zeros(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        zero_pv = torch.zeros(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        out_with_zero_prefix = attn(x, prefix_kv=(zero_pk, zero_pv))

        mag_no = out_no_prefix.abs().mean()
        mag_zero = out_with_zero_prefix.abs().mean()
        ratio = (mag_zero / mag_no).item()
        assert 0.0 < ratio < 1.0, f"expected ratio in (0, 1), got {ratio}"
        assert ratio < 0.99, f"zero prefix should shrink output by >=1%, got {ratio}"
        assert not torch.allclose(out_no_prefix, out_with_zero_prefix, atol=1e-4)

    def test_forward_with_prefix_kv_changes_output(self):
        """Non-zero prefix → output diverges from the no-prefix path."""
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        pv = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        out_with_prefix = attn(x, prefix_kv=(pk, pv))

        assert not torch.allclose(out_no_prefix, out_with_prefix, atol=1e-4)

    def test_prefix_does_not_affect_output_shape(self):
        """Output shape is independent of prefix_len."""
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        head_dim = 8
        for p_len in (1, 3, 8):
            pk = torch.randn(2, 4, p_len, head_dim, device=DEFAULT_DEVICE)
            pv = torch.randn(2, 4, p_len, head_dim, device=DEFAULT_DEVICE)
            out = attn(x, prefix_kv=(pk, pv))
            assert out.shape == (2, 5, 32)

    def test_prefix_kv_mismatched_shape_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        pk = torch.randn(2, 4, 3, 8, device=DEFAULT_DEVICE)
        pv = torch.randn(2, 4, 5, 8, device=DEFAULT_DEVICE)  # different seq dim
        with pytest.raises(ValueError, match="share shape"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_num_kv_heads_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        pk = torch.randn(2, 3, 3, 8, device=DEFAULT_DEVICE)  # 3 != num_kv_heads=4
        pv = torch.randn(2, 3, 3, 8, device=DEFAULT_DEVICE)
        with pytest.raises(ValueError, match="num_kv_heads"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_head_dim_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        pk = torch.randn(2, 4, 3, 16, device=DEFAULT_DEVICE)  # 16 != head_dim=8
        pv = torch.randn(2, 4, 3, 16, device=DEFAULT_DEVICE)
        with pytest.raises(ValueError, match="head_dim"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_batch_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)
        pk = torch.randn(3, 4, 3, 8, device=DEFAULT_DEVICE)  # batch=3 != 2
        pv = torch.randn(3, 4, 3, 8, device=DEFAULT_DEVICE)
        with pytest.raises(ValueError, match="batch"):
            attn(x, prefix_kv=(pk, pv))


class TestMLAPrefixWithMask:
    """MLA-specific: ``attn_mask`` S_k axis grows by ``prefix_len`` ones (prefix always visible)."""

    def test_mask_axis_grows_with_prefix(self):
        """Forward succeeds when prefix is prepended and a mask is provided.

        Without mask widening the latent attention's mask reshape at
        ``_latent_attention`` would mismatch the extended K shape and
        blow up with a shape error.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)

        # Mask covers the dynamic tokens: shape [B, 1, S_q, S_k] = [2, 1, 1, 5]
        # (S_q=1 because MLA collapses to the last query position).
        attn_mask = torch.ones(2, 1, 1, 5, device=DEFAULT_DEVICE)

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        pv = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)

        out = attn(x, attn_mask=attn_mask, prefix_kv=(pk, pv))
        assert out.shape == (2, 5, 32)

    def test_prefix_always_visible_regardless_of_mask(self):
        """A zero input mask still allows prefix-attended output (prefix segment is 1s).

        We construct a mask that zeros out the dynamic tokens, then
        verify the output with prefix is non-zero — the prefix segment
        was widened with ones regardless of the input mask.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(2, 5, 32, device=DEFAULT_DEVICE)

        # Mask that hides the dynamic tokens entirely.
        attn_mask = torch.zeros(2, 1, 1, 5, device=DEFAULT_DEVICE)

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)
        pv = torch.randn(2, 4, prefix_len, head_dim, device=DEFAULT_DEVICE)

        out = attn(x, attn_mask=attn_mask, prefix_kv=(pk, pv))
        # Output should be non-zero because the prefix segment
        # contributes via the ones we prepended to the mask.
        assert out.abs().mean() > 1e-6


class TestMLAPrefixWithLinearCache:
    """MLA linear KVCache + prefix interaction."""

    def test_prefix_with_linear_kv_cache_runs(self):
        """Forward succeeds with both kv_cache and prefix_kv set."""
        from llm.core.kv_cache import KVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(1, 5, 32, device=DEFAULT_DEVICE)

        kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            num_kv_heads=4,
            head_dim=8,
            device=DEVICE_STR,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)
        pv = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)

        out = attn(x, kv_cache=kv_cache, prefix_kv=(pk, pv))
        assert out.shape == (1, 5, 32)
        assert out.isfinite().all()

    def test_cached_kv_does_not_contain_prefix(self):
        """The KV cache stores dynamic tokens only, not the prefix.

        After forward with prefix_kv, the cache's seq_len should equal
        the input's seq_len (5), not seq_len + prefix_len (7).
        """
        from llm.core.kv_cache import KVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(1, 5, 32, device=DEFAULT_DEVICE)

        kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            num_kv_heads=4,
            head_dim=8,
            device=DEVICE_STR,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)
        pv = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)

        attn(x, kv_cache=kv_cache, prefix_kv=(pk, pv))

        # Cache should hold only the 5 dynamic tokens, not 7.
        assert kv_cache.seq_len == 5


class TestMLAPrefixWithPagedCache:
    """MLA paged KVCache + prefix interaction.

    The prefix is concatenated AFTER the paged gather returns
    (``_paged_kv_write`` returns raw K, V which are then concatenated
    with the prefix). The paged cache itself stores only dynamic tokens.
    """

    def test_prefix_with_paged_cache_runs(self):
        from llm.core.paged_attention.paged_kv_cache import PagedKVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8).to(DEFAULT_DEVICE)
        x = torch.randn(1, 5, 32, device=DEFAULT_DEVICE)

        paged = PagedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=8,
            num_blocks=4,
            block_size=4,
            device=DEVICE_STR,
            dtype=torch.float32,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)
        pv = torch.randn(1, 4, prefix_len, 8, device=DEFAULT_DEVICE)

        out = attn(
            x,
            paged_kv_cache=paged,
            layer_idx=0,
            batch_indices=torch.tensor([0], dtype=torch.long),
            prefix_kv=(pk, pv),
        )
        assert out.shape == (1, 5, 32)
        assert out.isfinite().all()


class TestMLAWorksAtHalfPrecision:
    """``attn_impl="mla"`` must forward in fp16 and bf16, not only fp32.

    ``MultiLatentAttention.forward`` built its uniform latent weights with
    ``torch.ones(...)`` and no ``dtype``, so the tensor was float32 while the
    model ran in half precision. The following ``torch.bmm`` then raised
    ``RuntimeError: expected scalar type Float but found Half`` (and
    ``BFloat16``), which made MLA unusable for serving: the demo path is fp32,
    but every real checkpoint path runs AMP or an explicitly half-precision
    model, and the serving benchmark with ``--attn-impl mla`` crashed before
    producing a number.

    Nothing covered this: MDA's dtype was pinned in no test, and the other three
    ``torch.ones``/``torch.zeros`` calls in the module already passed an
    explicit ``dtype``, so the omission was a single outlier rather than a
    pattern.
    """

    @staticmethod
    def _model(dtype_cast, device):
        import torch

        from llm.models.decoder import DecoderModel

        torch.manual_seed(0)
        model = DecoderModel(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attn_impl="mla",
            max_seq_len=32,
        )
        return dtype_cast(model).to(device).eval()

    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
    def test_forward_produces_the_model_dtype(self, dtype_name):
        import torch

        dtype = getattr(torch, dtype_name)
        if dtype is torch.float16 and not torch.cuda.is_available():
            pytest.skip("float16 forward needs CUDA on this backend")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = self._model(lambda m: m.to(dtype), device)
        ids = torch.randint(0, 64, (2, 8), device=device)
        with torch.no_grad():
            out = model(ids)

        assert out.dtype == dtype, f"forward returned {out.dtype}, expected {dtype}"
        assert torch.isfinite(out).all(), "half-precision MLA forward produced non-finite values"

    def test_no_dtype_less_tensor_creation_remains(self):
        """Static pin: every tensor creation in the module must state a dtype.

        This catches the whole class rather than the one call site that broke —
        a future ``torch.ones`` added without ``dtype`` reintroduces the same
        crash for half-precision users only, which is exactly the gap that let
        this ship.
        """
        import pathlib
        import re

        source = pathlib.Path(__file__).resolve().parents[2] / "src" / "llm" / "core" / "attn" / "mla.py"
        text = source.read_text(encoding="utf-8")

        offenders = []
        for match in re.finditer(r"torch\.(ones|zeros|full|empty)\(", text):
            # The call's argument list, to the matching close paren.
            start = match.end() - 1
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        call = text[start : i + 1]
                        break
            else:  # pragma: no cover - unbalanced source
                continue
            if "dtype=" not in call and "zeros_like" not in match.group(0):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"mla.py:{line}: {match.group(1)}(... ) has no dtype=")

        assert not offenders, (
            "torch tensor creations without an explicit dtype default to float32, which breaks "
            "half-precision forward with 'expected scalar type Float but found Half':\n  " + "\n  ".join(offenders)
        )


class TestMLARefusesGroupedQueryHeads:
    """``attn_impl="mla"`` has no GQA, so ``num_kv_heads != num_heads`` must be refused.

    The placeholder MLA produces ``num_kv_heads`` keys but its forward path
    assumes ``num_heads == num_kv_heads`` (there is no ``repeat_kv`` step — the
    prefix-injection comment states this outright). The constructor nevertheless
    accepted any ``num_kv_heads``, so a serving config with
    ``attn_impl="mla", num_kv_heads=2, num_heads=4`` booted cleanly and then
    failed on the first CACHED token:

        IndexError: index 2 is out of bounds for dimension 1 with size 2

    raised inside ``KVCache.update_at_indices`` — the cache is allocated from
    ``self.num_kv_heads`` but written with the model's head count, so the error
    points at the cache rather than at the configuration that caused it.

    ``mha`` and ``flash_attn`` DO implement GQA (they repeat KV heads), so the
    right behaviour is a construction-time refusal that names the alternative,
    not a silent mis-sizing.
    """

    @staticmethod
    def _build(**kwargs):
        from llm.core.attn.mla import MultiLatentAttention

        return MultiLatentAttention(hidden_size=32, num_heads=4, **kwargs)

    def test_grouped_query_heads_are_refused_at_construction(self):
        with pytest.raises(ValueError, match="does not support num_kv_heads"):
            self._build(num_kv_heads=2)

    def test_the_error_names_a_working_alternative(self):
        """A refusal is only useful if it says what to do instead."""
        with pytest.raises(ValueError, match="does not support num_kv_heads") as excinfo:
            self._build(num_kv_heads=2)
        message = str(excinfo.value)
        assert "mha" in message, "the error should point at an implementation that does support GQA"
        assert "flash_attn" in message

    def test_matching_num_kv_heads_still_works(self):
        """The refusal must not break the supported shape."""
        attn = self._build(num_kv_heads=4)
        assert attn.num_kv_heads == 4
        assert attn.num_kv_heads == attn.num_heads

    def test_unset_num_kv_heads_still_defaults_to_num_heads(self):
        attn = self._build()
        assert attn.num_kv_heads == attn.num_heads == 4

    def test_the_cached_forward_no_longer_reaches_the_index_error(self):
        """End-to-end: the config is rejected instead of failing mid-generation.

        Builds the model the way serving does and asserts the failure happens at
        model construction, so no partially-initialised engine can exist.
        """
        import torch

        from llm.models.decoder import DecoderModel

        with pytest.raises(ValueError, match="does not support num_kv_heads"):
            DecoderModel(
                vocab_size=64,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                attn_impl="mla",
                max_seq_len=32,
            )
        # Silence an unused-import warning while keeping the intent visible.
        assert torch is not None
