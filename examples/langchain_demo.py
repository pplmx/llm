#!/usr/bin/env python3
"""Example: using LangChain against ``llm serve``.

The server speaks the OpenAI chat-completions protocol, so LangChain's
``ChatOpenAI`` works against it unchanged — no custom adapter class is needed.
That is the whole point of the OpenAI-compatible layer: every framework that
already speaks OpenAI gets this server for free.

Prerequisites:
    1. Install LangChain's OpenAI integration: ``pip install langchain-openai``
       (not a project dependency — it is only needed to run this example).
    2. Start the server: ``llm serve`` (defaults to port 8000).

Why ``max_completion_tokens`` matters here: OpenAI deprecated ``max_tokens`` in
favour of ``max_completion_tokens``, and ``ChatOpenAI`` sends the NEW name only.
Before this server accepted it, every LangChain request silently ran with the
default token budget — and on a short-context model failed with a 400 that named
a field the caller never sent.

Run against a non-default host with ``--base-url``.
"""

import argparse
import sys

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install the LangChain OpenAI integration: pip install langchain-openai")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Server's OpenAI base URL")
    parser.add_argument("--prompt", default="Hello, world", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=32, help="Token budget")
    parser.add_argument("--api-key", default="test", help="API key (any string when auth is disabled)")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model="llm",  # the server ignores the id, but LangChain requires one
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    print("=== Non-streaming ===")
    response = llm.invoke(args.prompt)
    print(response.content)

    # Usage metadata is what a client bills against; surfacing it here shows the
    # token budget was actually applied rather than defaulted.
    usage = response.usage_metadata or {}
    print(f"\nusage: prompt={usage.get('input_tokens')} completion={usage.get('output_tokens')}")

    print("\n=== Streaming ===")
    for chunk in llm.stream(args.prompt):
        print(chunk.content, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
