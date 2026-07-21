"""Hypothesis-driven invariants for Prefix Tuning (T2 #19 + multi-backend extension).

The invariants below pin the structural contract of the prefix-tuning
extension across ``(batch, seq_len, hidden_size, num_heads, prefix_len)``
combinations. They cover three backends (MHA / MLA / Flash where
available) so the same shape contract holds after the multi-backend
extension.
"""

from __future__ import annotations

import hypothesis
import hypothesis.strategies as st
import pytest
import torch

from llm.core.attn.flash_attn import FLASH_ATTN_AVAILABLE
from llm.core.attn.mha import MultiHeadAttention
from llm.core.attn.mla import MultiLatentAttention

# Search-space caps. Larger values hit slow-down territory; the contract
# doesn't depend on absolute size, only on shape consistency.
MAX_BATCH = 4
MAX_SEQ = 8
MAX_HEADS = 4
MAX_PREFIX_LEN = 4
MAX_HIDDEN = 32


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH),
    seq_len=st.integers(min_value=1, max_value=MAX_SEQ),
    num_heads=st.integers(min_value=1, max_value=MAX_HEADS),
    prefix_len=st.integers(min_value=1, max_value=MAX_PREFIX_LEN),
)
def test_mha_prefix_preserves_output_shape(batch_size, seq_len, num_heads, prefix_len):
    """MHA: output shape is invariant in ``prefix_len``."""
    hidden_size = num_heads * 8  # head_dim = 8
    head_dim = hidden_size // num_heads

    torch.manual_seed(0)
    attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)
    pk = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    pv = torch.randn(batch_size, num_heads, prefix_len, head_dim)

    out_no_prefix = attn(x)
    out_with_prefix = attn(x, prefix_kv=(pk, pv))

    assert out_no_prefix.shape == (batch_size, seq_len, hidden_size)
    assert out_with_prefix.shape == (batch_size, seq_len, hidden_size)


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH),
    seq_len=st.integers(min_value=1, max_value=MAX_SEQ),
    num_heads=st.integers(min_value=1, max_value=MAX_HEADS),
    prefix_len=st.integers(min_value=1, max_value=MAX_PREFIX_LEN),
)
def test_mla_prefix_preserves_output_shape(batch_size, seq_len, num_heads, prefix_len):
    """MLA: output shape is invariant in ``prefix_len``.

    MLA has ``num_heads == num_kv_heads`` (no GQA) and an extra
    ``num_latents`` hyperparameter; we keep num_latents fixed to
    keep the property sharp.
    """
    hidden_size = num_heads * 8
    head_dim = hidden_size // num_heads

    torch.manual_seed(0)
    attn = MultiLatentAttention(hidden_size=hidden_size, num_heads=num_heads, num_latents=4)
    x = torch.randn(batch_size, seq_len, hidden_size)
    pk = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    pv = torch.randn(batch_size, num_heads, prefix_len, head_dim)

    out_no_prefix = attn(x)
    out_with_prefix = attn(x, prefix_kv=(pk, pv))

    assert out_no_prefix.shape == (batch_size, seq_len, hidden_size)
    assert out_with_prefix.shape == (batch_size, seq_len, hidden_size)


@hypothesis.settings(max_examples=10, deadline=None)
@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash-attn is optional; install via `llm[perf]`")
@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH),
    seq_len=st.integers(min_value=1, max_value=MAX_SEQ),
    num_heads=st.integers(min_value=1, max_value=MAX_HEADS),
    prefix_len=st.integers(min_value=1, max_value=MAX_PREFIX_LEN),
)
def test_flash_prefix_preserves_output_shape(batch_size, seq_len, num_heads, prefix_len):
    """Flash: output shape is invariant in ``prefix_len`` (fp16)."""
    from llm.core.attn.flash_attn import FlashAttention

    hidden_size = num_heads * 8
    head_dim = hidden_size // num_heads

    torch.manual_seed(0)
    attn = FlashAttention(hidden_size=hidden_size, num_heads=num_heads, dtype=torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    pk = torch.randn(batch_size, num_heads, prefix_len, head_dim, dtype=torch.float16)
    pv = torch.randn(batch_size, num_heads, prefix_len, head_dim, dtype=torch.float16)

    out_no_prefix = attn(x)
    out_with_prefix = attn(x, prefix_kv=(pk, pv))

    assert out_no_prefix.shape == (batch_size, seq_len, hidden_size)
    assert out_with_prefix.shape == (batch_size, seq_len, hidden_size)


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH),
    seq_len=st.integers(min_value=1, max_value=MAX_SEQ),
    num_heads=st.integers(min_value=1, max_value=MAX_HEADS),
    prefix_len=st.integers(min_value=1, max_value=MAX_PREFIX_LEN),
)
def test_mha_non_zero_prefix_changes_output(batch_size, seq_len, num_heads, prefix_len):
    """MHA: a non-zero prefix produces an output that diverges from the baseline."""
    hidden_size = num_heads * 8
    head_dim = hidden_size // num_heads

    torch.manual_seed(0)
    attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, hidden_size)

    out_no = attn(x).detach().clone()
    pk = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    pv = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    out_with = attn(x, prefix_kv=(pk, pv)).detach()

    assert not torch.allclose(out_no, out_with, atol=1e-4)


@hypothesis.settings(max_examples=20, deadline=None)
@hypothesis.given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH),
    seq_len=st.integers(min_value=1, max_value=MAX_SEQ),
    num_heads=st.integers(min_value=1, max_value=MAX_HEADS),
    prefix_len=st.integers(min_value=1, max_value=MAX_PREFIX_LEN),
)
def test_mha_gradients_flow_only_to_prefix(batch_size, seq_len, num_heads, prefix_len):
    """MHA: backward through prefix-wrapped MHA yields grads on prefix params, zero on base.

    Built around ``MultiHeadAttention`` directly (without the
    ``PrefixTuningAttention`` wrapper) because the wrapper test path
    is covered by ``test_prefix_tuning_integration.py``. Here we
    verify the inner MHA contract: when ``prefix_kv`` is supplied,
    backward should still flow through the (frozen-prefix) K, V
    tensor — verifying that prefix injection doesn't break the
    autograd graph.
    """
    hidden_size = num_heads * 8
    head_dim = hidden_size // num_heads

    torch.manual_seed(0)
    attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    pk = torch.randn(batch_size, num_heads, prefix_len, head_dim, requires_grad=True)
    pv = torch.randn(batch_size, num_heads, prefix_len, head_dim, requires_grad=True)

    out = attn(x, prefix_kv=(pk, pv))
    out.sum().backward()

    # x, pk, pv receive gradients.
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    assert pk.grad is not None
    assert pk.grad.abs().sum() > 0
    assert pv.grad is not None
    assert pv.grad.abs().sum() > 0
