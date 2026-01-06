#!/usr/bin/env python3
"""Basic inference example with DecoderModel."""

from llm import DecoderModel, SimpleCharacterTokenizer, generate

# 1. Create a simple tokenizer from corpus
corpus = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "This is a test of the language model",
]
tokenizer = SimpleCharacterTokenizer(corpus)

# 2. Initialize model
model = DecoderModel(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    max_seq_len=128,
)

# 3. Generate text (untrained model will produce random output)
prompt = "Hello"
generated = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=20,
    temperature=0.8,
)

print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
