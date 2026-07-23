#!/bin/bash
cd /workspace/llm
.venv/bin/python -c "
import pytest
pytest.main([
    'tests/serving/test_frequency_penalty_plumbing.py',
    '-v', '--tb=short'
])
"
