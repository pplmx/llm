import torch
import torch.nn.functional as functional
from torch import Tensor


def _add_attn_bias(mask: Tensor | None, bias: Tensor, query: Tensor) -> Tensor:
    """Fold a signed additive ``bias`` (ALiBi) into the SDPA mask.

    ``F.scaled_dot_product_attention`` accepts a single float additive mask;
    the codebase's bool mask convention (True = keep) and 0/-inf float
    additive masks must be converted to a float additive tensor first, then
    the signed ALiBi bias is summed in (ALiBi is NOT a mask — it is a
    position-dependent score adjustment, RIL — ALiBi milestone).
    """
    if mask is None:
        return bias.to(query.dtype)
    if mask.dtype == torch.bool:
        base = torch.where(
            mask,
            torch.zeros((), device=query.device, dtype=query.dtype),
            torch.full((), float("-inf"), device=query.device, dtype=query.dtype),
        )
    else:
        base = mask.to(query.dtype)
    return base + bias


def sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    attn_bias: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    window_size: int | None = None,
    prefix_len: int = 0,
) -> Tensor:
    """
    Computes Scaled Dot-Product Attention using `torch.nn.functional.scaled_dot_product_attention`.

    Acts as a compatibility wrapper for the codebase conventions:
    1. Handles `attn_mask` where True indicates masking out (opposite to Torch SDPA).
    2. Handles `window_size` by manually merging masks if necessary.
    3. Folds a signed additive `attn_bias` (ALiBi) into the float mask.

    Args:
        query (Tensor): Shape (B, N, Sq, D).
        key (Tensor): Shape (B, N, Sk, D).
        value (Tensor): Shape (B, N, Sk, D).
        attn_mask (Tensor | None): Mask where True indicates elements to MASK OUT.
                                   Can be boolean or 0/1 float additive (legacy).
        attn_bias (Tensor | None): Signed additive score bias (ALiBi), shape
                                   broadcastable to [B, N, Sq, Sk]. Summed
                                   into the mask (never through the 0/-inf
                                   masking channels).
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        scale (float | None): Scaling factor.
        window_size (int | None): Sliding window size.
        prefix_len (int): How many prefix K/V columns were prepended ahead of
            the real keys (Prefix Tuning). Shifts any causal diagonal by this
            amount so the real context stays visible (round-71 fix).
    """

    # 1. Handle Window Size and Mask Merging
    # If window_size is set, or if both attn_mask and is_causal are provided, we often need manual masking.
    has_window = window_size is not None and window_size > 0

    # Complex path: We need to construct a mask manually if:
    # - We have a window constraint (Torch SDPA doesn't support window_size directly yet for all backends/cases easily without mask)
    # - We have BOTH causal=True AND an attention mask (Torc SDPA generally prefers one or the other, or merged)
    # - causal=True with a prefix shift: torch's native is_causal mask is
    #   left-aligned (diagonal=1) and cannot express the prefix offset, so the
    #   shifted causal mask must be materialized here (round-71 prefix fix).
    if has_window or (is_causal and (attn_mask is not None or prefix_len != 0)):
        seq_len_q = query.size(-2)
        seq_len_k = key.size(-2)
        device = query.device

        # Start with efficient creation of causal mask if needed
        # We build a boolean mask where True = Mask Out (our convention)
        full_mask = None

        if is_causal:
            # True = Mask out (Upper triangle)
            # Shape: (Sq, Sk)
            #
            # ``prefix_len`` shifts the diagonal: when prefix K/V was prepended
            # ahead of the real keys (Prefix Tuning), key column ``j`` is at
            # absolute position ``j`` while query row ``i`` is at absolute
            # position ``prefix_len + i`` — so the causal bound is
            # ``j <= prefix_len + i``, i.e. ``diagonal = 1 + prefix_len``.
            # Without the shift the real (right-shifted) context keys were all
            # masked and the query attended only prefix keys (RIL round-71,
            # prefix-causal HIGH).
            full_mask = torch.triu(
                torch.ones((seq_len_q, seq_len_k), device=device, dtype=torch.bool),
                diagonal=1 + prefix_len,
            )

        if has_window:
            # The query block sits immediately after the cached keys, so its
            # rows carry absolute positions ``[seq_len_k - seq_len_q,
            # seq_len_k)``. During prefill (seq_len_k == seq_len_q) that is
            # ``[0, seq_len_k)``; during a KV-cache decode step
            # (seq_len_q == 1) it is ``[seq_len_k - 1]`` — the current
            # position. Using the *relative* row index (0..seq_len_q-1)
            # against absolute key columns attends the OLDEST window_size
            # keys at every decode step instead of the keys just before the
            # current position (RIL — decode window bug).
            row_offset = max(0, seq_len_k - seq_len_q)
            row_idx = (torch.arange(seq_len_q, device=device) + row_offset).unsqueeze(1)
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
            elif attn_mask.dtype.is_floating_point:
                # Float additive mask (0 = keep, -inf = mask out, Torch SDPA
                # convention). It cannot be merged with the boolean
                # ``full_mask`` via ``|`` — additive and boolean masks are
                # different spaces. Convert the boolean part to an additive
                # mask (0 / -inf) and sum them, so the caller's additive mask
                # is NOT silently dropped on the window / causal+mask path
                # (RIL ISS-115).
                bool_part = full_mask
                # Normalize the float mask's shape to a key-additive layout
                # ([B, 1, 1, S_k]) so it plays well with the rank-2
                # causal/window ``full_mask``. reward_task / reward tests
                # pass a plain ``[B, S]`` float mask.
                float_mask = attn_mask
                if float_mask.ndim == 2:
                    float_mask = float_mask.unsqueeze(1).unsqueeze(1)
                if bool_part is not None:
                    additive_base = torch.where(
                        bool_part,
                        torch.tensor(float("-inf"), device=query.device, dtype=float_mask.dtype),
                        torch.zeros((), device=query.device, dtype=float_mask.dtype),
                    )
                    full_mask = additive_base + float_mask
                else:
                    full_mask = float_mask
            else:
                # Integer 0/1 mask (e.g. the long padding mask emitted by the
                # SFT/DPO/reward data pipelines, where 1 = real token and
                # 0 = pad). 1 is *keep*, so the mask-out predicate here is
                # ``== 0`` — NOT ``to(bool)`` (which would flip padding to
                # keep and *real* tokens to mask out). The data pipeline
                # emits ``[B, S]``; expand to ``[B, 1, S]`` so it broadcasts
                # against the ``[Sq, Sk]`` causal/window ``full_mask``.
                mask_out = attn_mask == 0
                if mask_out.ndim == 2:
                    mask_out = mask_out.unsqueeze(1)
                full_mask = mask_out if full_mask is None else (full_mask | mask_out)

        # Now we have a mask where True = Mask Out (bool) or 0/-inf additive
        # (float), matching the caller's convention.
        # F.sdpa expects True = Keep (for boolean masks); float masks are
        # passed through as additive. And we set is_causal=False because we
        # baked causal in.

        torch_mask = None
        if full_mask is not None:
            torch_mask = ~full_mask if full_mask.dtype == torch.bool else full_mask
            # F.sdpa expects a boolean mask of rank >= 3 (broadcastable to
            # [B, N, Sq, Sk]). The window/causal construction is rank-2
            # ([Sq, Sk]); a caller-supplied [B, 1, S] int mask merges to
            # rank-3 ([B, Sq, Sk]), which has no head dimension. Expand the
            # head axis so the merged mask actually applies per head instead
            # of raising "expanded size ... at non-singleton dimension 1".
            if torch_mask.ndim == 3 and query.ndim == 4:
                torch_mask = torch_mask.unsqueeze(1)

        if attn_bias is not None:
            torch_mask = _add_attn_bias(torch_mask, attn_bias, query)

        return functional.scaled_dot_product_attention(
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
        if attn_mask.dtype == torch.bool:
            torch_attn_mask = ~attn_mask
        elif attn_mask.dtype.is_floating_point:
            # Float additive mask (0 = keep, -inf = mask out): pass through
            # unchanged — this is already the Torch SDPA convention. A
            # caller-supplied [B, S] float mask (reward_task / reward tests)
            # needs a broadcastable key axis for 4-D query, same as int.
            torch_attn_mask = attn_mask
            if torch_attn_mask.ndim == 2 and query.ndim == 4:
                torch_attn_mask = torch_attn_mask.unsqueeze(1).unsqueeze(1)
        else:
            # Integer 0/1 mask (e.g. a long padding mask from a data
            # pipeline): 1 = real = keep, 0 = pad. Torch wants True = Keep,
            # so the long mask maps 1:1 onto a bool mask with NO inversion
            # (1 -> True = keep); passing the raw long mask would be the
            # bitwise-not-free path but Torch SDPA rejects non-bool/float
            # dtypes. The data pipeline emits ``[B, S]``; F.sdpa rejects a
            # 2-D bool mask (and even ``[B, 1, S]``) for a 4-D query — it
            # needs at least ``[B, 1, 1, S]`` / ``[1, 1, S_q, S_k]``. Insert
            # a broadcastable head+key axis.
            torch_attn_mask = attn_mask.to(torch.bool)
            if torch_attn_mask.ndim == 2 and query.ndim == 4:
                torch_attn_mask = torch_attn_mask.unsqueeze(1).unsqueeze(1)

    if attn_bias is not None:
        if is_causal:
            # ``F.scaled_dot_product_attention`` rejects an explicit float
            # mask combined with ``is_causal=True``, and ALiBi forces a float
            # additive mask. Materialise the causal mask (upper triangle →
            # -inf) and fold ALiBi in; the mask now does the causality work.
            seq_len_q, seq_len_k = query.size(-2), key.size(-2)
            causal = torch.triu(
                torch.full(
                    (seq_len_q, seq_len_k),
                    float("-inf"),
                    device=query.device,
                    dtype=query.dtype,
                ),
                diagonal=1 + prefix_len,
            )  # [Sq, Sk]; lower triangle = 0 (keep); diagonal shifted by prefix
            torch_attn_mask = _add_attn_bias(torch_attn_mask, causal, query)
            torch_attn_mask = _add_attn_bias(torch_attn_mask, attn_bias.to(query.dtype), query)
            is_causal = False
        else:
            torch_attn_mask = _add_attn_bias(torch_attn_mask, attn_bias, query)

    return functional.scaled_dot_product_attention(
        query, key, value, attn_mask=torch_attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
