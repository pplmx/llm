import pytest
import torch

from llm.models.decoder import DecoderModel
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig


def pytest_configure(config):
    """Configure custom markers based on GPU availability."""
    gpu_count = torch.cuda.device_count()
    config.addinivalue_line("markers", "need_gpu(n): tests that require at least n GPUs")
    config.addinivalue_line("markers", "gpu: tests that require at least 1 GPU")
    if gpu_count >= 8:
        config.addinivalue_line("markers", "multi_gpu: tests that require multiple GPUs (2+)")
        config.addinivalue_line("markers", "full_cluster: tests that require 8 GPUs")
    elif gpu_count >= 2:
        config.addinivalue_line("markers", "multi_gpu: tests that require multiple GPUs (2+)")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on GPU availability."""
    gpu_count = torch.cuda.device_count()
    _ = config  # config is required by pytest but we only need gpu_count

    for item in items:
        # Check for need_gpu marker
        need_gpu_marker = item.get_closest_marker("need_gpu")
        if need_gpu_marker:
            required_gpus = need_gpu_marker.args[0] if need_gpu_marker.args else 1
            if gpu_count < required_gpus:
                item.add_marker(pytest.mark.skip(f"需要 {required_gpus} GPU, 当前 {gpu_count}"))

        # Check for full_cluster marker (requires 8 GPUs)
        if item.get_closest_marker("full_cluster") and gpu_count < 8:
            item.add_marker(pytest.mark.skip(f"需要 8 GPU, 当前 {gpu_count}"))


@pytest.fixture(scope="session")
def device():
    """Returns the best available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def gpu_count():
    """Returns the number of available GPUs."""
    return torch.cuda.device_count()


@pytest.fixture(scope="session")
def cuda_available():
    """Returns True if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset torch random seed before each test to ensure reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def tiny_config():
    """Provides a minimal configuration for fast unit testing."""
    return Config(
        model=ModelConfig(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16),
        training=TrainingConfig(batch_size=2, epochs=1, num_samples=10),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    """Provides a minimal DecoderModel instance on the appropriate device."""
    return DecoderModel(
        vocab_size=tiny_config.model.vocab_size,
        hidden_size=tiny_config.model.hidden_size,
        num_layers=tiny_config.model.num_layers,
        num_heads=tiny_config.model.num_heads,
        max_seq_len=tiny_config.model.max_seq_len,
        device=device,
    )
