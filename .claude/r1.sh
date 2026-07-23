#!/bin/bash
cd /workspace/llm
.venv/bin/python -c "
import pytest, sys
sys.exit(pytest.main([
    'tests/generation/test_stop_sequences.py',
    '-v', '--tb=short'
]))
"
