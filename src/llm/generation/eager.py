from collections.abc import Generator

import torch

from llm.core.kv_cache import create_decoder_kv_caches
from llm.generation.sampling import apply_repetition_penalty, sample_next_token
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
    use_cache: bool = True,
) -> Generator[str]:
    """
    Generator function for incremental text generation.

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

    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(next_token_logits, generated_ids, repetition_penalty)

        token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated_ids.append(token_id)
        text_chunk = tokenizer.decode([token_id])
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


def generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
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
        use_cache=use_cache,
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

    Returns:
        List of generated texts (prompt + generated tokens).
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

    # Decode results
    return [tokenizer.decode(ids) for ids in generated_ids]
