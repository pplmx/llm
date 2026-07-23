#!/bin/bash
cd /workspace/llm
uv run .claude/run_stop_tests.py "$@"
