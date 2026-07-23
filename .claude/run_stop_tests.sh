#!/bin/bash
cd /workspace/llm
.venv/bin/python .claude/run_stop_tests.py "$@"
