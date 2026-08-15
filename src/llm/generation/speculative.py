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

from typing import Protocol

import torch

from llm.generation.eager import _normalize_stop, _reject_impossible_context
from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_logit_bias,
    apply_presence_penalty,
    apply_repetition_penalty,
    mask_undecodable_logits,
    sample_next_token,
    sampling_probs,
)
from llm.models.decoder import DecoderModel


# Type alias for the (model, tokenizer) pair the speculative backend
# carries through the streaming protocol.
class TokenizerLike(Protocol):
    """Anything with encode/decode + optional pad/eos token ids."""

    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, /) -> list[int]: ...
    def decode(self, tokens: list[int], /) -> str: ...


def _verify_speculative_tokens(
    target: DecoderModel,
    draft: DecoderModel,
    input_ids: torch.Tensor,
    draft_tokens: list[int],
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    tokenizer_vocab_size: int | None = None,
) -> tuple[int, int | None]:
    """Score ``draft_tokens`` with the target and return (accept_count, bonus).

    Args:
        target: Target model (full size).
        draft: Draft model (small). Used to recompute draft logits so
            the acceptance ratio is correct; both models are passed to
            keep the call symmetric.
        input_ids: Context tokens (prompt + already-accepted tokens),
            shape ``[1, T]``.
        draft_tokens: Candidate tokens from the draft, length ``gamma``.
        temperature: Sampling temperature for the **correction** token.
        top_k, top_p: Sampling parameters applied uniformly.
        repetition_penalty: Repetition penalty applied to the target and
            draft logits exactly like the eager backend (once per token
            present in the history). ``1.0`` disables it.
        frequency_penalty: Per-occurrence frequency penalty. ``0.0``
            disables it.
        presence_penalty: Per-token presence penalty. ``0.0`` disables it.
        logit_bias: Per-token additive logit bias, applied after the
            penalties (same ordering as the eager backend).

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

    # Sampling-time penalties must be applied to the logits before the
    # acceptance probability, residual, and correction/bonus sampling —
    # exactly like the eager backend does at every decode step. The
    # history for draft position ``i`` is the context plus the first
    # ``i`` draft tokens; the bonus position sees the full sequence.
    full_ids = full[0].tolist()
    context_len = input_ids.size(1)

    def _apply_penalties(logits_row: torch.Tensor, history: list[int]) -> torch.Tensor:
        """Apply the same penalty pipeline as :func:`llm.generation.eager.generate`."""
        if repetition_penalty != 1.0:
            logits_row = apply_repetition_penalty(logits_row, history, repetition_penalty)
        if frequency_penalty != 0.0:
            logits_row = apply_frequency_penalty(logits_row, history, frequency_penalty)
        if presence_penalty != 0.0:
            logits_row = apply_presence_penalty(logits_row, history, presence_penalty)
        if logit_bias:
            logits_row = apply_logit_bias(logits_row, logit_bias)
        return logits_row

    with torch.no_grad():
        target_out = target(full, kv_caches=None, use_cache=False)
        target_logits = target_out[0] if isinstance(target_out, tuple) else target_out
        # Target logits at the positions corresponding to each draft
        # token AND the bonus position (one past the last draft
        # token). ``context_len`` is the length of the context; the
        # first draft token's score lives at index ``context_len - 1``
        # (last context token predicts the next), and the bonus
        # position lives at ``context_len - 1 + gamma``. So we slice
        # ``[T-1, T-1+gamma+1)`` = ``[T-1, T+gamma]`` for a length of
        # ``gamma + 1``.
        relevant = target_logits[0, context_len - 1 : context_len + gamma, :]
    # Mask any token id the tokenizer cannot decode (the model vocab may
    # exceed the tokenizer's — padding etc.), so an accepted/bonus token
    # sampled below is always decodable by the caller (RIL ISS-125). Must
    # be applied BEFORE the penalty/acceptance math so the scored
    # distribution still matches what the sampler draws from.
    mask_undecodable_logits(relevant, tokenizer_vocab_size)

    # Row ``i`` uses the context plus the first ``i`` draft tokens as
    # its penalty history; the bonus row (``gamma``) uses everything.
    target_penalized = torch.stack(
        [_apply_penalties(relevant[i], full_ids[: context_len + i]) for i in range(gamma + 1)]
    )

    # The "target prob of draft token at position i" is the prob under
    # the distribution the *sampler* would draw from at that position.
    # ``sample_next_token`` applies top-k/top-p filtering on top of
    # temperature scaling, so scoring the acceptance ratio against a raw
    # full-vocab softmax would make the accepted-token set diverge from
    # the eager backend's output (RIL ISS-99) — e.g. it could accept a
    # token top-k would have masked. Use the exact filtered distribution
    # via ``sampling_probs`` (same helper as ``sample_next_token``) for
    # both the target and the draft.
    target_relevant = target_penalized[:gamma]  # only the draft-token positions
    if temperature == 0:
        # Greedy: always accept tokens whose argmax matches.
        target_argmax = target_relevant.argmax(dim=-1)
        accepted = (target_argmax == torch.tensor(draft_tokens, device=device)).tolist()
    else:
        target_probs = sampling_probs(target_relevant, temperature=temperature, top_k=top_k, top_p=top_p)
        draft_tensor_dev = torch.tensor(draft_tokens, device=device)
        q_target = target_probs[torch.arange(gamma, device=device), draft_tensor_dev]

        # Draft probs at the same positions, with the same per-position
        # penalties as the target (the draft loop in
        # ``speculative_generate`` samples from the penalized logits,
        # so the ratio must use the penalized draft distribution).
        with torch.no_grad():
            draft_out = draft(full, kv_caches=None, use_cache=False)
            draft_logits = draft_out[0] if isinstance(draft_out, tuple) else draft_out
        draft_relevant = draft_logits[0, context_len - 1 : context_len + gamma, :]
        mask_undecodable_logits(draft_relevant, tokenizer_vocab_size)
        draft_penalized = torch.stack(
            [_apply_penalties(draft_relevant[i], full_ids[: context_len + i]) for i in range(gamma)]
        )
        draft_probs = sampling_probs(draft_penalized, temperature=temperature, top_k=top_k, top_p=top_p)
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
        bonus_logits = target_penalized[gamma]
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
        # Rejection: sample the correction token from the normalized
        # residual ``(q_target - q_draft)+`` so the overall output
        # distribution still matches the target (Leviathan et al. 2023,
        # Algorithm 2). The residual must be computed on the
        # temperature-scaled **probabilities** — a logit difference
        # ``softmax(a - b)`` is proportional to the ratio ``p/q`` and
        # biases the output distribution away from the target.
        reject_pos = accept_count
        if temperature == 0:
            # Greedy: the correction is whatever the target would have
            # emitted deterministically at the rejection position
            # (its argmax).
            bonus = sample_next_token(
                target_relevant[reject_pos],
                temperature=0.0,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            # ``target_probs`` / ``draft_probs`` hold the temperature-
            # scaled probability distributions at every draft position
            # (computed above); row ``reject_pos`` corresponds to the
            # rejected candidate.
            residual = (target_probs[reject_pos] - draft_probs[reject_pos]).clamp(min=0.0)
            if residual.sum() <= 0:
                # Degenerate case: fall back to the target distribution.
                bonus = sample_next_token(
                    target_relevant[reject_pos],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            else:
                # ``sample_next_token`` applies a softmax, so pass the
                # log-residual to sample exactly from the normalized
                # residual.
                bonus = sample_next_token(
                    torch.log(residual),
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
    logit_bias: dict[int, float] | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
):
    """Speculative decoding generator.

    Yields decoded chunks. Stops after ``max_new_tokens`` produced
    tokens or on EOS.

    Args:
        target: Target model (the "expensive" one). Its forward
            distribution is the canonical output distribution.
        draft: Draft model (the "cheap" one). Must share vocabulary
            with the target and have the same ``max_seq_len`` (or
            larger - we only enforce the prompt fits).
        tokenizer: Tokenizer with ``encode``, ``decode``,
            ``pad_token_id``, ``eos_token_id``.
        prompt: Prompt text.
        max_new_tokens: Hard cap on generated tokens.
        gamma: Number of speculative tokens per round. Typical
            values are 4-8.
        temperature: Sampling temperature for the **correction**
            token (the algorithm preserves the target distribution
            under these settings).
        top_k: Top-k sampling parameter for the correction token.
        top_p: Nucleus-sampling (top-p) parameter for the correction
            token.
        repetition_penalty: Applied to both draft and target logits
            before sampling.
        seed: Optional RNG seed for reproducible rejection sampling.
        stop: OpenAI-compat stop sequence(s). Generation halts the
            moment the accumulated output contains any of these as a
            suffix; the stop string itself is NOT included in the
            yielded output. Accepts a single string or a list of
            strings. ``None`` is a no-op.
    """
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if seed is not None:
        torch.manual_seed(seed)

    target.eval()
    draft.eval()
    device = next(target.parameters()).device

    # Same context-window handshake as the eager backend (RIL ISS-124): a
    # budget that cannot fit is rejected up front, and an over-long prompt is
    # truncated to fit. The speculative backend keeps rebuilding the full
    # context tensor each round and forwarding it through the learned
    # positional-encoding table, so without this a long prompt crashed with
    # ``ValueError: Sequence endpoint N exceeds maximum sequence length``
    # while the eager backend served the same input fine (it truncates).
    max_seq_len = getattr(target, "max_seq_len", None)
    _reject_impossible_context(max_seq_len, max_new_tokens)
    prompt_ids = tokenizer.encode(prompt)
    if max_seq_len is not None and len(prompt_ids) + max_new_tokens > max_seq_len:
        keep = max(1, max_seq_len - max_new_tokens)
        prompt_ids = prompt_ids[-keep:]

    generated_ids: list[int] = list(prompt_ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Stop-sequence tracking via a small suffix buffer (same strategy
    # as stream_generate: keep at most ``max_stop_len`` chars un-yielded
    # so the buffer is O(max_stop_len) and suffix matching is exact).
    stops = _normalize_stop(stop)
    max_stop_len = max((len(s) for s in stops), default=0) if stops else 0
    buffer = ""

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
            mask_undecodable_logits(next_logits, getattr(tokenizer, "vocab_size", None))
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, draft_ids, repetition_penalty)
            if frequency_penalty != 0.0:
                next_logits = apply_frequency_penalty(next_logits, draft_ids, frequency_penalty)
            if presence_penalty != 0.0:
                next_logits = apply_presence_penalty(next_logits, draft_ids, presence_penalty)
            if logit_bias:
                next_logits = apply_logit_bias(next_logits, logit_bias)
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
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            tokenizer_vocab_size=getattr(tokenizer, "vocab_size", None),
        )

        # 3. Emit accepted tokens + bonus (or correction). The EOS token is
        # never part of the yielded output — halt on it *before* decoding and
        # appending, matching the eager backend (RIL ISS-96/ISS-98) which
        # stops on EOS without emitting the EOS token's decoded text.
        for i in range(accept_count):
            tok = draft_tokens[i]
            if eos_id is not None and tok == eos_id:
                if stops and buffer:
                    yield buffer
                return
            generated_ids.append(tok)
            text_chunk = tokenizer.decode([tok])
            if stops and text_chunk:
                buffer += text_chunk
                for s in stops:
                    if buffer.endswith(s):
                        prefix = buffer[: len(buffer) - len(s)]
                        if prefix:
                            yield prefix
                        return
                if len(buffer) > max_stop_len:
                    safe_len = len(buffer) - max_stop_len
                    yield buffer[:safe_len]
                    buffer = buffer[safe_len:]
            else:
                yield text_chunk
            if len(generated_ids) - len(prompt_ids) >= max_new_tokens:
                if stops and buffer:
                    yield buffer
                return

        # Append the bonus or correction token (one per round).
        if bonus is not None:
            if eos_id is not None and bonus == eos_id:
                if stops and buffer:
                    yield buffer
                return
            generated_ids.append(bonus)
            text_chunk = tokenizer.decode([bonus])
            if stops and text_chunk:
                buffer += text_chunk
                for s in stops:
                    if buffer.endswith(s):
                        prefix = buffer[: len(buffer) - len(s)]
                        if prefix:
                            yield prefix
                        return
                if len(buffer) > max_stop_len:
                    safe_len = len(buffer) - max_stop_len
                    yield buffer[:safe_len]
                    buffer = buffer[safe_len:]
            else:
                yield text_chunk

    # Flush any remaining buffered text when the loop exhausts
    # max_new_tokens without a stop or EOS triggering an early return.
    if stops and buffer:
        yield buffer
    return
