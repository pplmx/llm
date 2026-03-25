# tests/core/conftest.py
import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Session-scoped device fixture - call once per test session."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()
