"""Integration tests: block-sparse & StreamingLLM masks through real attention (TASK-239).

Proves the mask builders actually shape the *computed* attention output, not just
a boolean table:
- feeding the mask into the repo's real ``sdpa`` backend matches an explicit
  manual masked-attention reference computed on the same Q/K/V;
- feeding it through ``MultiHeadAttention.forward`` changes the output for a
  genuinely sparse pattern and is bit-identical to dense when it covers all keys.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from llm.core.attn import MultiHeadAttention
from llm.core.attn.block_sparse import build_block_sparse_mask, mask_to_additive
from llm.core.attn.sdpa import sdpa
from llm.core.attn.streaming_llm import build_streamingllm_mask

SEQ, B, N, D = 16, 2, 4, 16


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (
        torch.randn(B, N, SEQ, D),
        torch.randn(B, N, SEQ, D),
        torch.randn(B, N, SEQ, D),
    )


def _sdpd_mask(mask: torch.Tensor) -> torch.Tensor:
    """Repo ``sdpa`` convention: bool ``True`` = mask out -> ``~sparse``, broadcast."""
    return (~mask).bool().unsqueeze(0).unsqueeze(0)


def _manual_reference(q, k, v, mask: torch.Tensor) -> torch.Tensor:
    """Explicit masking: 0 where attend, -inf where blocked, pure torch SDPA."""
    additive = mask_to_additive(mask).to(q.dtype).unsqueeze(0).unsqueeze(0)
    return functional.scaled_dot_product_attention(q, k, v, attn_mask=additive, is_causal=False)


def test_sdpa_block_sparse_matches_explicit_masking():
    q, k, v = _qkv()
    sparse = build_block_sparse_mask(SEQ, block_size=2, window_blocks=1, global_blocks=1, causal=False)
    out = sdpa(q, k, v, attn_mask=_sdpd_mask(sparse), is_causal=False)
    assert torch.allclose(out, _manual_reference(q, k, v, sparse), atol=1e-5)


def test_sdpa_streaming_matches_explicit_masking():
    q, k, v = _qkv()
    stream = build_streamingllm_mask(SEQ, num_sink=2, window_size=4, causal=True)
    out = sdpa(q, k, v, attn_mask=_sdpd_mask(stream), is_causal=False)
    assert torch.allclose(out, _manual_reference(q, k, v, stream), atol=1e-5)


def test_sdpa_full_coverage_sparse_equals_no_mask():
    """A sparse mask that blocks nothing is a no-op for the real attention."""
    q, k, v = _qkv()
    full = build_block_sparse_mask(SEQ, block_size=2, window_blocks=0, global_blocks=SEQ // 2, causal=False)
    assert bool(full.all())
    out_full = sdpa(q, k, v, attn_mask=_sdpd_mask(full), is_causal=False)
    out_plain = sdpa(q, k, v, attn_mask=None, is_causal=False)
    assert torch.allclose(out_full, out_plain, atol=1e-6)


def test_mha_forward_applies_mask_and_full_coverage_parity():
    mha = MultiHeadAttention(hidden_size=64, num_heads=8, is_causal=False, include_norm_residual=True, p=0.0)
    hidden = torch.randn(B, SEQ, 64)

    dense = mha(hidden, attn_mask=None, is_causal=False)
    dense = dense[0] if isinstance(dense, tuple) else dense

    # A genuinely sparse pattern must change the computed output.
    sparse = build_block_sparse_mask(SEQ, block_size=2, window_blocks=1, global_blocks=1, causal=False)
    out = mha(hidden, attn_mask=_sdpd_mask(sparse), is_causal=False)
    out = out[0] if isinstance(out, tuple) else out
    assert not torch.allclose(out, dense, atol=1e-4), "sparse mask changed nothing in the real forward"

    # Full-coverage sparse is bit-identical to dense (parity through MHA.forward).
    full = build_block_sparse_mask(SEQ, block_size=2, window_blocks=0, global_blocks=SEQ // 2, causal=False)
    out_full = mha(hidden, attn_mask=_sdpd_mask(full), is_causal=False)
    out_full = out_full[0] if isinstance(out_full, tuple) else out_full
    assert torch.allclose(dense, out_full, atol=1e-6)
