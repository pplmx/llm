import pytest
import torch

from llm.core.attn.sdpa import sdpa


@pytest.fixture
def sample_qkv_tensors():
    """Create sample Q, K, V tensors for testing."""
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    return q, k, v


def test_sdpa_basic(sample_qkv_tensors):
    """Test basic functionality without masks."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v)
    assert output.shape == q.shape


def test_sdpa_with_attn_mask_bool(sample_qkv_tensors):
    """Test with boolean attention mask (True=Mask Out)."""
    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    # Mask last token
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[..., -1] = True

    output = sdpa(q, k, v, attn_mask=mask)
    assert output.shape == q.shape


def test_sdpa_with_attn_mask_float(sample_qkv_tensors):
    """Test with float attention mask (0/-inf)."""
    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    # Mask last token with large negative value
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32)
    mask[..., -1] = float("-inf")

    output = sdpa(q, k, v, attn_mask=mask)
    assert output.shape == q.shape


def test_sdpa_window_size(sample_qkv_tensors):
    """Test window_size functionality."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, window_size=2)
    assert output.shape == q.shape


def test_sdpa_causal(sample_qkv_tensors):
    """Test causal masking."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, is_causal=True)
    assert output.shape == q.shape


def test_sdpa_causal_and_window(sample_qkv_tensors):
    """Test combined causal and window masking."""
    q, k, v = sample_qkv_tensors
    output = sdpa(q, k, v, is_causal=True, window_size=2)
    assert output.shape == q.shape


def test_sdpa_float_additive_mask_not_dropped_on_complex_path(sample_qkv_tensors):
    """Regression (RIL ISS-115): a float additive attention mask (0 / -inf)
    must be honored on the complex path (``window_size`` or
    ``is_causal=True`` + mask), not silently dropped.

    The complex path merges masks manually; previously a non-bool mask hit a
    bare ``pass`` and vanished — with a ``-inf`` key masked out the output
    still attended to that key (window+float diverged by ~1.7, causal+float
    by ~0.37). The boolean part must be converted to the same additive space
    (0 / -inf) and summed with the caller's mask.
    """
    import torch.nn.functional as functional

    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    # Mask the last key column with -inf (additive, Torch SDPA convention).
    add = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.float32)
    add[..., -1] = float("-inf")

    # Reference: torch.native merges the additive mask with causal itself.
    ref_causal = functional.scaled_dot_product_attention(q, k, v, attn_mask=add, is_causal=True)
    out_causal = sdpa(q, k, v, attn_mask=add, is_causal=True)
    assert torch.allclose(out_causal, ref_causal, atol=1e-5)

    # Reference for window+float: build the combined additive mask by hand.
    window = torch.abs(torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]) > 2
    window_add = torch.where(window, torch.tensor(float("-inf")), torch.tensor(0.0))
    ref_mask = window_add + add
    ref_window = functional.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask, is_causal=False)
    out_window = sdpa(q, k, v, attn_mask=add, window_size=2)
    assert torch.allclose(out_window, ref_window, atol=1e-5)


def test_sdpa_bool_mask_plus_window_merges(sample_qkv_tensors):
    """Boolean masks still merge with the window mask on the complex path
    (the bool | bool branch is untouched by the float-mask fix)."""
    import torch.nn.functional as functional

    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[..., -1] = True  # true = mask out
    window = torch.abs(torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]) > 2

    ref_mask = window | mask
    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=~ref_mask, is_causal=False)
    out = sdpa(q, k, v, attn_mask=mask, window_size=2)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_long_01_mask_matches_additive_reference(sample_qkv_tensors):
    """Int64 0/1 masks from the data pipeline (SFT/DPO/reward emit
    ``torch.LongTensor`` attention_mask where 1 = real token, 0 = pad) must
    be honored on both the complex (causal + mask) and fast paths.

    The plant was a latent correctness bug: the long mask hit a bare
    ``pass`` on the complex path (padding tokens silently attended) and a
    dtype error on the fast path. The 0/1 convention is 1 = *keep*, so the
    mask-out predicate is ``== 0`` and the fast path maps 1:1 to bool
    (True = keep) — inverting either would mask the real tokens instead.
    """
    import torch.nn.functional as functional

    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    # Right-padding: no query row is fully masked after the causal merge, so
    # the additive reference stays finite and comparable.
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mask[0, seq_len - 2 :] = 0
    mask[1, seq_len - 1] = 0

    # Additive reference: causal (-inf upper) + pad-key (-inf at 0 positions).
    causal_keep = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1)
    causal_add = causal_keep.to(q.dtype).masked_fill(causal_keep, float("-inf"))
    pad4 = (mask == 0).unsqueeze(1).unsqueeze(2)
    pad_add = pad4.to(q.dtype).masked_fill(pad4, float("-inf"))
    add = causal_add.expand(batch_size, 1, seq_len, seq_len) + pad_add

    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=add, is_causal=False)
    out = sdpa(q, k, v, attn_mask=mask, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)

    # Fast path: no causal, padding keys only. Reference additive mask is
    # just the pad-key -inf.
    ref_fast = functional.scaled_dot_product_attention(q, k, v, attn_mask=pad_add, is_causal=False)
    out_fast = sdpa(q, k, v, attn_mask=mask)
    assert torch.allclose(out_fast, ref_fast, atol=1e-5)

    # The mask is actually doing something: without it (causal only) the
    # padded keys are attended, so the outputs must differ.
    out_no_mask = sdpa(q, k, v, is_causal=True)
    assert not torch.allclose(out, out_no_mask)


def test_sdpa_float_2d_fast_path_broadcasts(sample_qkv_tensors):
    """A ``[B, S]`` float mask (reward_task / reward tests pass float32
    attention_mask) on the no-causal fast path must be broadcast to a key
    axis instead of raising a shape error."""
    import torch.nn.functional as functional

    q, k, v = sample_qkv_tensors
    batch_size, _num_heads, seq_len, _ = q.shape

    fm = torch.ones(batch_size, seq_len, dtype=torch.float32)
    fm[0, seq_len - 1] = 0.0
    fm[1, :2] = 0.0

    out = sdpa(q, k, v, attn_mask=fm)
    add = fm.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S] additive reference
    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=add, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)
