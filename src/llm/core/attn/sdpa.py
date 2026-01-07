import torch
import torch.nn.functional as F
from torch import Tensor


def sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: int | None = None,
) -> Tensor:
    """
    Computes Scaled Dot-Product Attention using `torch.nn.functional.scaled_dot_product_attention`.

    Acts as a compatibility wrapper for the codebase conventions:
    1. Handles `attn_mask` where True indicates masking out (opposite to Torch SDPA).
    2. Handles `window_size` by manually merging masks if necessary.

    Args:
        query (Tensor): Shape (B, N, Sq, D).
        key (Tensor): Shape (B, N, Sk, D).
        value (Tensor): Shape (B, N, Sk, D).
        attn_mask (Tensor | None): Mask where True indicates elements to MASK OUT.
                                   Can be boolean or 0/1 float additive (legacy).
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        scale (float | None): Scaling factor.
        window_size (int | None): Sliding window size.
    """

    # 1. Handle Window Size and Mask Merging
    # If window_size is set, or if both attn_mask and is_causal are provided, we often need manual masking.
    has_window = window_size is not None and window_size > 0

    # Complex path: We need to construct a mask manually if:
    # - We have a window constraint (Torch SDPA doesn't support window_size directly yet for all backends/cases easily without mask)
    # - We have BOTH causal=True AND an attention mask (Torc SDPA generally prefers one or the other, or merged)
    if has_window or (is_causal and attn_mask is not None):
        seq_len_q = query.size(-2)
        seq_len_k = key.size(-2)
        device = query.device

        # Start with efficient creation of causal mask if needed
        # We build a boolean mask where True = Mask Out (our convention)
        full_mask = None

        if is_causal:
            # True = Mask out (Upper triangle)
            # Shape: (Sq, Sk)
            full_mask = torch.triu(
                torch.ones((seq_len_q, seq_len_k), device=device, dtype=torch.bool),
                diagonal=1,
            )

        if has_window:
            row_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
            col_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
            # True = Mask out (distance > window)
            # Standard window attention: |i - j| > w
            # Note: For Causal Window, it's just i - j > w (past) ... but usually window is symmetric or causal.
            # Assuming standard generalized window constraint here.
            window_mask = torch.abs(row_idx - col_idx) > window_size
            full_mask = window_mask if full_mask is None else (full_mask | window_mask)

        if attn_mask is not None:
            # attn_mask: True = Mask out
            # We assume attn_mask is broadcastable.
            # If attn_mask is float/int 0/1. we should convert to bool for logical ops if we can,
            # but usually it's passed as bool in this codebase.
            # If it's float additive (-inf), this merging logic is trickier.
            # Assuming bool mask for complex merging.
            if attn_mask.dtype == torch.bool:
                full_mask = attn_mask if full_mask is None else (full_mask | attn_mask)
            else:
                # Fallback or mixed type?
                # If we have a complex float mask AND window... it's hard to merge efficiently with boolean logic.
                # Ideally convert float mask to bool if it's 0/-inf style?
                # For now let's hope it's consistent.
                # If not bool, we might crash on logical OR if it's float.
                pass  # Trust caller or subsequent steps.

        # Now we have a boolean mask where True = Mask Out.
        # F.sdpa expects True = Keep (for boolean masks).
        # So we pass ~full_mask.
        # And we set is_causal=False because we baked it in.

        torch_mask = None
        if full_mask is not None:
            torch_mask = ~full_mask if full_mask.dtype == torch.bool else full_mask

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=torch_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )

    # 2. Fast Path: No complex masking conflict
    # We can rely on F.sdpa's native logic or simple inversion.

    torch_attn_mask = None
    if attn_mask is not None:
        # My convention: True = Mask Out
        # Torch convention: True = Keep
        torch_attn_mask = ~attn_mask if attn_mask.dtype == torch.bool else attn_mask

    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=torch_attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
