#!/bin/bash
cd /workspace/llm
uv run pytest \
    tests/serving/test_frequency_penalty_plumbing.py \
    -v --tb=short
