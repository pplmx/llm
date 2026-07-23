#!/bin/bash
cd /workspace/llm
uv run pytest \
    tests/generation/test_stop_sequences.py \
    -v --tb=short
