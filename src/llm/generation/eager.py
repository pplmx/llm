from collections.abc import Generator

import torch

from llm.core.kv_cache import create_decoder_kv_caches
from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_logit_bias,
    apply_presence_penalty,
    apply_repetition_penalty,
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


def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
    """Normalize the OpenAI-compat ``stop`` field to ``list[str] | None``.

    OpenAI accepts either a single string or a list of up to 4 strings;
    we standardize internally to a list so the streaming check is one
    loop instead of two. ``None`` and ``[]`` both mean "no stop" —
    pass-through ``None`` is the zero-cost default.
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


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
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    max_seq_len = getattr(model, "max_seq_len", 512)
    kv_caches = create_decoder_kv_caches(model, batch_size=1) if use_cache else None

    # Prefill: truncate if needed to fit max_seq_len
    if input_tensor.size(1) + max_new_tokens > max_seq_len:
        input_tensor = input_tensor[:, -(max_seq_len - max_new_tokens) :]
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
    batch_size = len(prompts)

    # Encode all prompts
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    prompt_lengths = [len(p) for p in encoded_prompts]
    max_prompt_len = max(prompt_lengths)

    # Get pad token id
    pad_id = getattr(tokenizer, "pad_token_id", 0)

    # Left-pad sequences to align generation positions
    padded_inputs = []
    for ids in encoded_prompts:
        padding_len = max_prompt_len - len(ids)
        padded_inputs.append([pad_id] * padding_len + ids)

    input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)

    # Track generated ids per sequence
    generated_ids: list[list[int]] = [ids.copy() for ids in encoded_prompts]

    # Prefill
    max_seq_len = getattr(model, "max_seq_len", 512)
    if input_tensor.size(1) + max_new_tokens > max_seq_len:
        truncate_len = max_seq_len - max_new_tokens
        input_tensor = input_tensor[:, -truncate_len:]

    kv_caches = create_decoder_kv_caches(model, batch_size=batch_size)
    logits, kv_caches = model(input_tensor, kv_caches=kv_caches, use_cache=True)
    next_token_logits = logits[:, -1, :]  # [B, vocab_size]

    _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))

    for _ in range(max_new_tokens):
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

        logits, kv_caches = model(next_tokens, kv_caches=kv_caches, use_cache=True)
        next_token_logits = logits[:, -1, :]

        _mask_pad_logits(next_token_logits, getattr(tokenizer, "pad_token_id", None))

    # Decode results, applying stop sequences when provided.
    stops = _normalize_stop(stop)
    if stops:
        results = []
        for i in range(batch_size):
            prompt_text = tokenizer.decode(encoded_prompts[i])
            full_text = tokenizer.decode(generated_ids[i])
            generated_text = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else full_text
            for s in stops:
                idx = generated_text.find(s)
                if idx != -1:
                    generated_text = generated_text[:idx]
                    break
            results.append(prompt_text + generated_text)
        return results

    return [tokenizer.decode(ids) for ids in generated_ids]
