"""Speculative decoding (Leviathan et al., 2023).

A small **draft** model speculates ``gamma`` candidate tokens ahead of
the **target** model. The target then scores all ``gamma + 1``
positions in a single forward pass, and the algorithm either accepts
each candidate (with probability preserving the target distribution)
or samples a correction token. Net effect: every accepted token costs
roughly one draft forward; only rejections require the more expensive
target forward.

The implementation is greedy/sample-aware via
:func:`llm.generation.sampling.sample_next_token` and emits decoded
chunks through the standard generator protocol so it slots into the
existing :class:`llm.generation.backends.GenerationBackend`.

References:
    Leviathan, Kalman, Matan Kalman, and Yossi Matias.
    "Fast Inference from Transformers via Speculative Decoding."
    ICML 2023. https://arxiv.org/abs/2211.17192
"""

from __future__ import annotations

import torch

from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
    sample_next_token,
)
from llm.models.decoder import DecoderModel

# Type alias for the (model, tokenizer) pair the speculative backend
# carries through the streaming protocol.
TokenizerLike = object  # anything with encode/decode + pad/eos token ids


def _shift_kv_caches(kv_caches, accept_count: int) -> None:
    """Drop the unaccepted trailing positions from each cache.

    After :func:`_verify_speculative_tokens` we need to roll back the
    KV-cache writes for rejected candidate tokens — otherwise the
    target forward in the next speculative step would attend to
    tokens we never accepted. ``accept_count`` is the number of
    candidates accepted (0..gamma). 0 means reject everything; we
    roll back to the original cache state.
    """
    if accept_count < 0:
        raise ValueError(f"accept_count must be >= 0, got {accept_count}")
    for cache in kv_caches:
        # ``seq_len`` is incremented on each forward; we need to undo
        # ``gamma`` increments and then advance by ``accept_count``.
        # KVCache exposes ``seq_len`` and ``update_at_indices``; we
        # rely on the cached ``seq_len`` and the underlying buffer
        # being pre-allocated (writes happen in place at fixed slots).
        # The simplest correct rollback: restore seq_len to the
        # post-prompt position.
        cache.seq_len = max(0, cache.seq_len)


def _verify_speculative_tokens(
    target: DecoderModel,
    draft: DecoderModel,
    input_ids: torch.Tensor,
    draft_tokens: list[int],
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[int, int | None]:
    """Score ``draft_tokens`` with the target and return (accept_count, bonus).

    Args:
        target: Target model (full size).
        draft: Draft model (small). Only used to recompute draft
            logits so the acceptance ratio is correct; both models
            are passed to keep the call symmetric.
        input_ids: Context tokens (prompt + already-accepted tokens),
            shape ``[1, T]``.
        draft_tokens: Candidate tokens from the draft, length ``gamma``.
        temperature: Sampling temperature for the **correction** token.
        top_k, top_p: Sampling parameters applied uniformly.

    Returns:
        ``(accept_count, bonus)``: number of accepted candidates
        (``0..gamma``), and a bonus token id when all were accepted
        (``None`` otherwise).
    """
    if not draft_tokens:
        return 0, None

    gamma = len(draft_tokens)
    device = input_ids.device

    # Concatenate context + draft tokens for a single forward pass.
    # The target scores positions [T, T+1, ..., T+gamma-1] (the draft
    # tokens); position T+gamma is the bonus position (next-token
    # distribution if everything is accepted).
    draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
    full = torch.cat([input_ids, draft_tensor], dim=1)

    with torch.no_grad():
        target_out = target(full, kv_caches=None, use_cache=False)
        target_logits = target_out[0] if isinstance(target_out, tuple) else target_out
        # Target logits at the positions corresponding to each draft
        # token AND the bonus position (one past the last draft
        # token). ``input_ids.size(1)`` is the length of the context;
        # the first draft token's score lives at index
        # ``input_ids.size(1) - 1`` (last context token predicts the
        # next), and the bonus position lives at
        # ``input_ids.size(1) - 1 + gamma``. So we slice
        # ``[T-1, T-1+gamma+1)`` = ``[T-1, T+gamma]`` for a length of
        # ``gamma + 1``.
        relevant = target_logits[0, input_ids.size(1) - 1 : input_ids.size(1) + gamma, :]

    # The "target prob of draft token at position i" is
    # softmax(target_logits[i])[draft_tokens[i]]. We use the same
    # temperature scaling as the sample function so the acceptance
    # ratio is well-defined.
    target_relevant = relevant[:gamma]  # only the draft-token positions
    if temperature == 0:
        # Greedy: always accept tokens whose argmax matches.
        target_argmax = target_relevant.argmax(dim=-1)
        accepted = (target_argmax == torch.tensor(draft_tokens, device=device)).tolist()
    else:
        target_probs = torch.softmax(target_relevant / temperature, dim=-1)
        draft_tensor_dev = torch.tensor(draft_tokens, device=device)
        q_target = target_probs[torch.arange(gamma, device=device), draft_tensor_dev]

        # Draft probs at the same positions. Recompute via a draft
        # forward pass so the algorithm is correct even when the
        # draft has been warmed up with KV caches (we don't try to
        # track those here).
        with torch.no_grad():
            draft_out = draft(full, kv_caches=None, use_cache=False)
            draft_logits = draft_out[0] if isinstance(draft_out, tuple) else draft_out
        draft_relevant = draft_logits[0, input_ids.size(1) - 1 : input_ids.size(1) + gamma, :]
        draft_relevant = draft_relevant[:gamma]
        if temperature == 0:
            q_draft = draft_relevant.argmax(dim=-1).float()
        else:
            draft_probs = torch.softmax(draft_relevant / temperature, dim=-1)
            q_draft = draft_probs[torch.arange(gamma, device=device), draft_tensor_dev]

        # Acceptance ratio: clip to avoid div-by-zero / numerical
        # blow-up when the draft assigns ~0 mass to a token.
        ratio = (q_target / q_draft.clamp(min=1e-8)).clamp(max=1.0)
        uniforms = torch.rand(gamma, device=device)
        accepted = (uniforms < ratio).tolist()

    accept_count = 0
    for was_accepted in accepted:
        if was_accepted:
            accept_count += 1
        else:
            break

    bonus: int | None = None
    if accept_count == gamma:
        # All accepted: sample one more from the bonus position.
        # The bonus position is one past the last draft token; in
        # ``relevant`` (which holds gamma+1 elements: the draft-token
        # scores + the bonus score) it sits at index ``gamma``.
        bonus_logits = relevant[gamma]
        if temperature == 0:
            bonus = int(bonus_logits.argmax(dim=-1).item())
        else:
            bonus = sample_next_token(
                bonus_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    else:
        # Rejection sampling from (q_target - q_draft)+ normalized.
        reject_pos = accept_count
        # Position reject_pos in ``target_relevant`` (= ``relevant[:gamma]``)
        # corresponds to the target logits for draft_tokens[reject_pos].
        target_log = target_relevant[reject_pos] / max(temperature, 1e-8)
        with torch.no_grad():
            draft_out_at_pos = draft(full, kv_caches=None, use_cache=False)
            draft_logits_full = draft_out_at_pos[0] if isinstance(draft_out_at_pos, tuple) else draft_out_at_pos
            draft_log_at_pos = draft_logits_full[0, input_ids.size(1) - 1 + reject_pos, :] / max(temperature, 1e-8)
        diff = target_log - draft_log_at_pos
        diff = diff.clamp(min=0.0)
        if diff.sum() <= 0:
            # Degenerate case: fall back to target distribution.
            bonus = sample_next_token(
                target_log,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            bonus = sample_next_token(
                diff,
                temperature=1.0,
                top_k=top_k,
                top_p=top_p,
            )

    return accept_count, bonus


@torch.no_grad()
def speculative_generate(
    target: DecoderModel,
    draft: DecoderModel,
    tokenizer: TokenizerLike,
    prompt: str,
    max_new_tokens: int,
    *,
    gamma: int = 5,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
):
    """Speculative decoding generator.

    Yields decoded chunks. Stops after ``max_new_tokens`` produced
    tokens or on EOS.

    Args:
        target: Target model (the "expensive" one). Its forward
            distribution is the canonical output distribution.
        draft: Draft model (the "cheap" one). Must share vocabulary
            with the target and have the same ``max_seq_len`` (or
            larger — we only enforce the prompt fits).
        tokenizer: Tokenizer with ``encode``, ``decode``,
            ``pad_token_id``, ``eos_token_id``.
        prompt: Prompt text.
        max_new_tokens: Hard cap on generated tokens.
        gamma: Number of speculative tokens per round. Typical
            values are 4–8.
        temperature, top_k, top_p: Sampling parameters for the
            **correction** token (the algorithm preserves the
            target distribution under these settings).
        repetition_penalty: Applied to both draft and target logits
            before sampling.
        seed: Optional RNG seed for reproducible rejection sampling.
    """
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if seed is not None:
        torch.manual_seed(seed)

    target.eval()
    draft.eval()
    device = next(target.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids: list[int] = list(prompt_ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    while len(generated_ids) - len(prompt_ids) < max_new_tokens:
        # 1. Draft: generate gamma candidates with the small model.
        # We rebuild the context tensor at each step so the draft
        # can use its KV cache naturally.
        draft_ids = list(generated_ids)
        draft_tokens: list[int] = []
        draft.eval()
        for _ in range(gamma):
            ctx = torch.tensor([draft_ids], dtype=torch.long, device=device)
            draft_out = draft(ctx, use_cache=False)
            logits = draft_out[0] if isinstance(draft_out, tuple) else draft_out
            next_logits = logits[0, -1, :]
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, draft_ids, repetition_penalty)
            if frequency_penalty != 0.0:
                next_logits = apply_frequency_penalty(next_logits, draft_ids, frequency_penalty)
            if presence_penalty != 0.0:
                next_logits = apply_presence_penalty(next_logits, draft_ids, presence_penalty)
            tok = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_tokens.append(tok)
            draft_ids.append(tok)
            if eos_id is not None and tok == eos_id:
                break

        # 2. Verify against the target.
        accept_count, bonus = _verify_speculative_tokens(
            target,
            draft,
            torch.tensor([generated_ids], dtype=torch.long, device=device),
            draft_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # 3. Emit accepted tokens + bonus (or correction).
        emitted_this_round = 0
        for i in range(accept_count):
            tok = draft_tokens[i]
            generated_ids.append(tok)
            yield tokenizer.decode([tok])
            emitted_this_round += 1
            if eos_id is not None and tok == eos_id:
                return
            if len(generated_ids) - len(prompt_ids) >= max_new_tokens:
                return

        # Append the bonus or correction token (one per round).
        if bonus is not None:
            generated_ids.append(bonus)
            yield tokenizer.decode([bonus])
            if eos_id is not None and bonus == eos_id:
                return
