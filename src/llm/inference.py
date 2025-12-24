from collections.abc import Generator

import torch

from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


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
    past_key_values = None

    # Prefill: truncate if needed to fit max_seq_len
    if input_tensor.size(1) + max_new_tokens > max_seq_len:
        input_tensor = input_tensor[:, -(max_seq_len - max_new_tokens) :]

    logits, past_key_values = model(input_tensor, use_cache=True)
    next_token_logits = logits[0, -1, :]

    if hasattr(tokenizer, "pad_token_id"):
        next_token_logits[tokenizer.pad_token_id] = -float("inf")

    generated_ids = input_ids.copy()

    for _ in range(max_new_tokens):
        # Repetition penalty (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            score = torch.gather(next_token_logits, 0, torch.tensor(generated_ids, device=device))
            # If score < 0, multiply by penalty; if score > 0, divide by penalty
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            next_token_logits.scatter_(0, torch.tensor(generated_ids, device=device), score)

        # Sampling strategy
        if temperature == 0:
            # Greedy Search
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Sampling
            next_token_logits = next_token_logits / temperature

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[..., -1, None]] = -torch.inf

            # Top-P (Nucleus Sampling)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        text_chunk = tokenizer.decode([token_id])
        yield text_chunk

        next_input = next_token.unsqueeze(0)
        logits, past_key_values = model(next_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[0, -1, :]

        if hasattr(tokenizer, "pad_token_id"):
            next_token_logits[tokenizer.pad_token_id] = -float("inf")


def generate(
    model: DecoderModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
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
    )
    return prompt + "".join(list(generator))


if __name__ == "__main__":
    corpus = ["hello world!", "this is a test.", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 2. Initialize model (using vocab_size matching tokenizer)
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
    )
    print("Model initialized (no weights loaded).")

    # 3. Run generation
    print("\nTesting Greedy Search (temperature=0):")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0,
    )
    print(f"Generated: {generated_text}")

    print("\nTesting Sampling (temperature=0.8, top_k=5):")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
    )
    print(f"Generated: {generated_text}")
