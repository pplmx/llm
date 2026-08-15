from collections.abc import Generator

import torch

from llm.core.kv_cache import create_decoder_kv_caches
from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_logit_bias,
    apply_presence_penalty,
    apply_repetition_penalty,
    mask_undecodable_logits,
    sample_next_token,
)
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def _mask_pad_logits(logits: torch.Tensor, pad_token_id: int | None) -> None:
    """Mask PAD token logits when the id is within model vocabulary bounds."""
    if pad_token_id is None:
        return
    vocab_size = logits.size(-1)
    if 0 <= pad_token_id < vocab_size:
        if logits.dim() == 1:
            logits[pad_token_id] = -float("inf")
        else:
            logits[:, pad_token_id] = -float("inf")


def _reject_impossible_context(max_seq_len: int | None, max_new_tokens: int) -> None:
    """Reject a generation budget that cannot fit in the context window.

    When ``max_new_tokens >= max_seq_len`` there is no room for even a
    single prompt token plus the requested budget: the prefill is clamped
    to one token and every decode step over-runs the KV cache (or attends
    beyond context with ``use_cache=False``), crashing mid-stream with an
    opaque cache-overflow error. Mirror the serving tier's up-front
    ``ValueError`` so library callers fail fast with a clear message.
    """
    if max_seq_len is not None and max_new_tokens >= max_seq_len:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) must be less than the model's "
            f"max_seq_len ({max_seq_len}); the prompt would have no room "
            "to fit in the context window."
        )


def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
    """Normalize the OpenAI-compat ``stop`` field to ``list[str] | None``.

    OpenAI accepts either a single string or a list of up to 4 strings;
    we standardize internally to a list so the streaming check is one
    loop instead of two. ``None`` and ``[]`` both mean "no stop" —
    pass-through ``None`` is the zero-cost default.

    Empty strings are filtered out: ``"".endswith("")`` is always True
    and would immediately halt generation.
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop] if stop else None
    filtered = [s for s in stop if s]
    return filtered if filtered else None


@torch.no_grad()
def stream_generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    use_cache: bool = True,
    stop: str | list[str] | None = None,
) -> Generator[str]:
    """
    Generator function for incremental text generation.

    Args:
        stop: OpenAI-compat stop sequence(s). Generation halts the
            moment the accumulated output contains any of these as a
            suffix; the stop string itself is NOT included in the
            yielded output. Accepts a single string or a list of
            strings (OpenAI caps at 4). ``None`` is a no-op.

    yields:
        str: Newly generated text chunk (usually one token decoded).
    """
    model.eval()
    device = next(model.parameters()).device
    _reject_impossible_context(getattr(model, "max_seq_len", None), max_new_tokens)
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    max_seq_len = getattr(model, "max_seq_len", 512)
    kv_caches = create_decoder_kv_caches(model, batch_size=1) if use_cache else None

    # Prefill: truncate if needed to fit max_seq_len. Defensive: never slice
    # to an *empty* prompt — when max_new_tokens >= max_seq_len the slice
    # bound goes non-positive and the tensor becomes 0-length, crashing the
    # forward with a 500. Clamp to the last token so the model always sees a
    # non-empty context (the serving tier rejects this config up front).
    if input_tensor.size(1) + max_new_tokens > max_seq_len:
        keep = max(1, max_seq_len - max_new_tokens)
        input_tensor = input_tensor[:, -keep:]
        # Update input_ids to match truncated tensor
        input_ids = input_tensor[0].tolist()

    if use_cache:
        logits, kv_caches = model(input_tensor, kv_caches=kv_caches, use_cache=True)
        next_token_logits = logits[0, -1, :]
    else:
        # Initial forward pass without cache
        logits = model(input_tensor, use_cache=False)
        next_token_logits = logits[0, -1, :]

    _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))
    mask_undecodable_logits(next_token_logits, getattr(tokenizer, "vocab_size", None))

    generated_ids = input_ids.copy()

    # Stop-sequence tracking. We use a small buffer (``buffer``) that
    # holds decoded text not yet yielded to the caller. After each new
    # token is decoded we append it to the buffer and check whether the
    # buffer *ends with* any stop string (OpenAI semantics: generation
    # halts when a stop sequence appears as a suffix; the stop string
    # itself is NOT included in the output). If no stop is found, we
    # yield the portion of the buffer that extends beyond
    # ``max_stop_len`` characters from the end — that prefix is safe
    # because no stop sequence of length <= max_stop_len can span the
    # boundary. Only the last ``max_stop_len`` characters are kept
    # buffered so memory stays O(max_stop_len) regardless of how long
    # generation runs.
    stops = _normalize_stop(stop)
    max_stop_len = max((len(s) for s in stops), default=0) if stops else 0
    buffer = ""
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(next_token_logits, generated_ids, repetition_penalty)
        if frequency_penalty != 0.0:
            next_token_logits = apply_frequency_penalty(next_token_logits, generated_ids, frequency_penalty)
        if presence_penalty != 0.0:
            next_token_logits = apply_presence_penalty(next_token_logits, generated_ids, presence_penalty)
        if logit_bias:
            next_token_logits = apply_logit_bias(next_token_logits, logit_bias)

        token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        # Model end-of-sequence: flush any buffered stop-prefix text and
        # halt. The EOS token itself is NOT part of the output (matches the
        # speculative backend's halting and standard LLM serving semantics);
        # without this the eager loop kept decoding through max_new_tokens
        # past EOS, emitting junk.
        if eos_id is not None and token_id == eos_id:
            if stops and buffer:
                yield buffer
            return
        generated_ids.append(token_id)
        text_chunk = tokenizer.decode([token_id])

        if stops and text_chunk:
            buffer += text_chunk
            # Check for a stop suffix — the first match wins.
            for s in stops:
                if buffer.endswith(s):
                    prefix = buffer[: len(buffer) - len(s)]
                    if prefix:
                        yield prefix
                    return
            # No stop found. Yield the safe prefix (everything beyond
            # the last max_stop_len characters) and keep the tail.
            if len(buffer) > max_stop_len:
                safe_len = len(buffer) - max_stop_len
                yield buffer[:safe_len]
                buffer = buffer[safe_len:]
        else:
            yield text_chunk

        next_input = torch.tensor([token_id], dtype=torch.long, device=device).unsqueeze(0)

        if use_cache:
            logits, kv_caches = model(next_input, kv_caches=kv_caches, use_cache=True)
            next_token_logits = logits[0, -1, :]
        else:
            # Without cache, append new token to full sequence and forward pass
            # generated_ids already has the new token appended
            full_input = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(full_input, use_cache=False)
            next_token_logits = logits[0, -1, :]

        _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))
        mask_undecodable_logits(next_token_logits, getattr(tokenizer, "vocab_size", None))

    # Flush any remaining buffered text after the loop ends (e.g. when
    # the buffer never exceeded max_stop_len or no stop sequence was found).
    if stops and buffer:
        yield buffer


def generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    use_cache: bool = True,
    stop: str | list[str] | None = None,
) -> str:
    """
    Generate text from a prompt using a trained model.
    """
    generator = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logit_bias=logit_bias,
        use_cache=use_cache,
        stop=stop,
    )
    return prompt + "".join(list(generator))


@torch.no_grad()
def batch_generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    stop: str | list[str] | None = None,
) -> list[str]:
    """
    Batch generate text from multiple prompts.

    Args:
        model: The decoder model.
        tokenizer: The tokenizer.
        prompts: List of input prompts.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature. 0 for greedy.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        repetition_penalty: Repetition penalty.
        frequency_penalty: OpenAI-compatible per-frequency penalty
            (subtracts ``frequency_penalty * count(token)`` from each
            seen token's logit). ``0.0`` is a no-op.
        presence_penalty: OpenAI-compatible per-presence penalty
            (subtracts a flat ``presence_penalty`` from each seen
            token's logit regardless of count). ``0.0`` is a no-op.
        logit_bias: OpenAI-compatible additive per-token biases
            (``{token_id: bias}`` added to the affected logits
            before sampling). ``None`` is a no-op.
        stop: OpenAI-compat stop sequence(s). Generation for each
            sequence halts the moment the generated text (post-prompt)
            contains any stop string; the stop string itself is NOT
            included in the returned text. Accepts a single string or
            a list of strings. ``None`` is a no-op.

    Returns:
        List of generated texts (prompt + generated tokens, with any
        stop sequence truncated).
    """
    if not prompts:
        return []

    model.eval()
    device = next(model.parameters()).device
    _reject_impossible_context(getattr(model, "max_seq_len", None), max_new_tokens)
    batch_size = len(prompts)

    # Encode all prompts
    encoded_prompts = [tokenizer.encode(p) for p in prompts]

    # Truncate prompts that exceed ``max_seq_len - max_new_tokens`` **before**
    # padding and ``generated_ids`` initialisation.  Doing the truncate here
    # (instead of slicing the padded tensor afterwards) keeps
    # ``generated_ids`` in sync with the tokens the model actually attends
    # to in the prefill forward pass.  Otherwise the repetition-penalty
    # context would include token ids the model never saw.
    max_seq_len = getattr(model, "max_seq_len", 512)
    truncate_len = max_seq_len - max_new_tokens
    if truncate_len > 0:
        max_prompt_len = max(len(ids) for ids in encoded_prompts)
        if max_prompt_len + max_new_tokens > max_seq_len:
            encoded_prompts = [ids[-truncate_len:] if len(ids) > truncate_len else ids for ids in encoded_prompts]

    prompt_lengths = [len(p) for p in encoded_prompts]
    max_prompt_len = max(prompt_lengths) if prompt_lengths else 0

    # Get pad token id
    pad_id = getattr(tokenizer, "pad_token_id", 0)

    # Left-pad sequences to align generation positions
    padded_inputs = []
    for ids in encoded_prompts:
        padding_len = max_prompt_len - len(ids)
        padded_inputs.append([pad_id] * padding_len + ids)

    input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)

    # Left-pad attention mask (True = mask out, matching the codebase SDPA
    # convention in ``llm.core.attn.sdpa``).  The left-pad columns are real
    # pad-token embeddings under the (default) causal mask: without an
    # explicit mask the prefill forward attends over the pad K/V, AND those
    # pad columns stay in the KV cache for every decode step — silently
    # diverging from the single-prompt path (RIL ISS-070).  We build one
    # mask sized to the full generation window and slice it per forward:
    # the prefill key length is ``max_prompt_len`` and each decode step t
    # grows the key length to ``max_prompt_len + t + 1``.  Generated columns
    # (beyond ``max_prompt_len``) are never masked.  Slices stay 4-D
    # ``[B, 1, 1, k_len]`` so they broadcast to ``[B, N, Sq, Sk]`` like the
    # batch-engine's ``run_attn_mask``.
    max_total_len = max_prompt_len + max_new_tokens
    pad_mask = torch.zeros((batch_size, 1, 1, max_total_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(encoded_prompts):
        pad_len = max_prompt_len - len(ids)
        if pad_len > 0:
            pad_mask[i, 0, 0, :pad_len] = True

    # Track generated ids per sequence — seeded from the (possibly truncated)
    # encoded prompts so the repetition-penalty context matches the model's
    # actual prefill input.
    generated_ids: list[list[int]] = [ids.copy() for ids in encoded_prompts]

    kv_caches = create_decoder_kv_caches(model, batch_size=batch_size)
    logits, kv_caches = model(
        input_tensor,
        kv_caches=kv_caches,
        use_cache=True,
        attn_mask=pad_mask[..., :max_prompt_len],
    )
    next_token_logits = logits[:, -1, :]  # [B, vocab_size]

    _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))
    mask_undecodable_logits(next_token_logits, getattr(tokenizer, "vocab_size", None))

    for step in range(max_new_tokens):
        for i in range(batch_size):
            row_logits = next_token_logits[i]
            if repetition_penalty != 1.0:
                row_logits = apply_repetition_penalty(row_logits, generated_ids[i], repetition_penalty)
            if frequency_penalty != 0.0:
                row_logits = apply_frequency_penalty(row_logits, generated_ids[i], frequency_penalty)
            if presence_penalty != 0.0:
                row_logits = apply_presence_penalty(row_logits, generated_ids[i], presence_penalty)
            if logit_bias:
                row_logits = apply_logit_bias(row_logits, logit_bias)
            token_id = sample_next_token(
                row_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated_ids[i].append(token_id)

        next_tokens = torch.tensor(
            [[generated_ids[i][-1]] for i in range(batch_size)],
            dtype=torch.long,
            device=device,
        )

        # Decode key length grows by one per step (the cache already holds
        # ``max_prompt_len + step`` keys after the prefill, plus the new one).
        logits, kv_caches = model(
            next_tokens,
            kv_caches=kv_caches,
            use_cache=True,
            attn_mask=pad_mask[..., : max_prompt_len + step + 1],
        )
        next_token_logits = logits[:, -1, :]

        _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))
        mask_undecodable_logits(next_token_logits, getattr(tokenizer, "vocab_size", None))

    # Truncate each sequence at its first EOS so both decode paths below
    # omit the EOS token and any junk generated after it (a sequence that
    # already finished keeps occupying its batch slot, but its tail is cut
    # here). Matches stream_generate / the speculative backend.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        for i in range(batch_size):
            gen_start = len(encoded_prompts[i])
            for j in range(gen_start, len(generated_ids[i])):
                if generated_ids[i][j] == eos_id:
                    del generated_ids[i][j:]
                    break

    # Decode results, applying stop sequences when provided.
    # OpenAI semantics: generation halts when a stop sequence appears as
    # a **suffix** of the running output. We simulate incremental decode
    # to find the first suffix match — .find() would match anywhere and
    # could prematurely truncate on prompt-embedded sequences or matches
    # that wouldn't have been a suffix during streaming.
    stops = _normalize_stop(stop)
    if stops:
        prompt_texts = [tokenizer.decode(p) for p in encoded_prompts]
        results = []
        for i in range(batch_size):
            running = prompt_texts[i]
            p_len = len(prompt_texts[i])
            # Walk generated tokens one by one, checking for suffix stop
            # after each decode (mirrors stream_generate incremental logic).
            gen_start = len(encoded_prompts[i])
            for tid in generated_ids[i][gen_start:]:
                running += tokenizer.decode([tid])
                generated_part = running[p_len:]
                truncated = False
                for s in stops:
                    if generated_part.endswith(s):
                        generated_part = generated_part[: -len(s)]
                        truncated = True
                        break
                if truncated:
                    break
            results.append(prompt_texts[i] + generated_part)
        return results

    return [tokenizer.decode(ids) for ids in generated_ids]
