#!/bin/bash
cd /workspace/llm
uv run pytest \
    tests/generation/test_stop_sequences.py::test_generation_config_stop_defaults_to_none \
    tests/generation/test_stop_sequences.py::test_generation_config_accepts_single_string_stop \
    tests/generation/test_stop_sequences.py::test_generation_config_accepts_list_of_strings \
    tests/generation/test_stop_sequences.py::test_generation_config_stop_is_immutable \
    -v --tb=short
