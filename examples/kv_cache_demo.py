#!/usr/bin/env python3
"""
Example: Using KVCache for efficient autoregressive generation.

This example demonstrates how to use the pre-allocated KVCache
for memory-efficient and fast text generation.
"""

import torch

from llm.core.kv_cache import KVCache
from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def main():
    # Create a simple tokenizer
    corpus = ["Hello world", "This is a test", "How are you doing today"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # Create model
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2

    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,  # GQA: 4 query heads, 2 KV heads
        max_seq_len=128,
    )
    model.eval()

    print("=" * 50)
    print("KVCache Example: Efficient Autoregressive Generation")
    print("=" * 50)

    # Method 1: Using generate() function (KVCache handled internally)
    print("\n1. Using generate() function:")
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_new_tokens=20,
        temperature=0.8,
    )
    print(f"   Generated: {text}")

    # Method 2: Manual KVCache creation and usage
    print("\n2. Manual KVCache usage:")

    # Create KV caches for all layers
    kv_caches = KVCache.from_model_config(
        max_batch_size=1,
        max_seq_len=64,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        device="cpu",
        dtype=torch.float32,
    )

    # Encode prompt
    prompt = "Hi"
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    # Generation loop with explicit KVCache
    generated_ids = input_ids.tolist()[0]

    for _ in range(10):
        with torch.no_grad():
            logits, _ = model(input_ids, kv_caches=kv_caches, use_cache=True)

        # Get next token
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_token)

        # Only pass new token for next iteration
        input_ids = torch.tensor([[next_token]])

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids)
    print(f"   Generated: {generated_text}")

    # Show cache info
    cache = kv_caches[0]
    print("\n3. Cache Info:")
    print(f"   Current seq_len: {cache.seq_len}")
    print(f"   Max seq_len: {cache.max_seq_len}")
    print(f"   Cache shape: {cache.k_cache.shape}")

    # Reset for new sequence
    for c in kv_caches:
        c.reset()
    print(f"   After reset: seq_len = {kv_caches[0].seq_len}")


if __name__ == "__main__":
    main()
