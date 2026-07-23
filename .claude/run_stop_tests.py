import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main([
        "tests/generation/test_stop_sequences.py",
        "-v", "--tb=short", "-x"
    ]))
