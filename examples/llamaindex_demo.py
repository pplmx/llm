#!/usr/bin/env python3
"""Example: using LlamaIndex against ``llm serve``.

``OpenAILike`` is LlamaIndex's adapter for any OpenAI-compatible endpoint, so it
targets this server without a custom integration. Use it instead of
``llama_index.llms.openai.OpenAI`` when the endpoint is not api.openai.com: the
latter validates model names against OpenAI's catalogue and rejects arbitrary ids.

Prerequisites:
    1. ``pip install llama-index-llms-openai-like`` (not a project dependency).
    2. Start the server: ``llm serve`` (defaults to port 8000).

``max_tokens`` here becomes ``max_completion_tokens`` on the wire — LlamaIndex
follows the current OpenAI spelling, which is why the server had to learn it.

Run against a non-default host with ``--base-url``.
"""

import argparse
import sys

try:
    from llama_index.llms.openai_like import OpenAILike
except ImportError:
    print("Please install the LlamaIndex integration: pip install llama-index-llms-openai-like")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Server's OpenAI base URL")
    parser.add_argument("--prompt", default="Hello, world", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=32, help="Token budget")
    parser.add_argument("--api-key", default="test", help="API key (any string when auth is disabled)")
    args = parser.parse_args()

    llm = OpenAILike(
        model="llm",  # the server ignores the id, but the client requires one
        api_base=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=0.0,
        is_chat_model=True,
    )

    print("=== Completion ===")
    response = llm.complete(args.prompt)
    print(str(response))

    print("\n=== Streaming ===")
    for chunk in llm.stream_complete(args.prompt):
        print(chunk.delta, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
