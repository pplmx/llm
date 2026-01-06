#!/usr/bin/env python3
"""Example: Using OpenAI SDK with llm-serve.

Prerequisites:
    1. Install openai: pip install openai
    2. Start the server: llm-serve
"""

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

# Connect to local llm-serve
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test",  # Any string works if auth is disabled
)

# Non-streaming example
print("=== Non-streaming ===")
response = client.chat.completions.create(
    model="llm",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=20,
    temperature=0.8,
)
print(f"Response: {response.choices[0].message.content}")

# Streaming example
print("\n=== Streaming ===")
stream = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Tell me a short story"}],
    max_tokens=50,
    stream=True,
)

print("Response: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
