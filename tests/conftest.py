import os

import pytest
import torch

from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from tests.support.corpus import DEFAULT_INFERENCE_CORPUS
from tests.support.devices import DEFAULT_DEVICE, cuda_device_count, cuda_usable
from tests.support.devices import all_gpu_devices as get_all_gpu_devices
from tests.support.tokenizers import LineTokenizer, StubTokenizer


def pytest_configure(config):
    """Configure custom markers based on GPU availability.

    Markers are always registered (so static ``@pytest.mark.need_gpu`` decorators
    don't trigger warnings), but auto-skip logic in ``pytest_collection_modifyitems``
    enforces the actual GPU count at collection time.
    """
    config.addinivalue_line("markers", "need_gpu(n): tests that require at least n GPUs")
    config.addinivalue_line("markers", "gpu: tests that require at least 1 GPU")
    config.addinivalue_line("markers", "multi_gpu: tests that require multiple GPUs (2+)")
    config.addinivalue_line("markers", "full_cluster: tests that require 8 GPUs")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on GPU availability.

    Uses :func:`cuda_usable` from ``tests.support.devices`` so that GPUs
    reported as ``is_available()`` but with 0 free VRAM (common in CI)
    are correctly treated as unusable.
    """
    gpu_count = cuda_device_count()
    _ = config  # config is required by pytest but we only need gpu_count

    for item in items:
        # Check for need_gpu marker — auto-skip if insufficient GPUs
        need_gpu_marker = item.get_closest_marker("need_gpu")
        if need_gpu_marker:
            required_gpus = need_gpu_marker.args[0] if need_gpu_marker.args else 1
            if gpu_count < required_gpus:
                item.add_marker(pytest.mark.skip(f"需要 {required_gpus} GPU, 当前 {gpu_count}"))

        # Check for full_cluster marker (requires 8 GPUs)
        if item.get_closest_marker("full_cluster") and gpu_count < 8:
            item.add_marker(pytest.mark.skip(f"需要 8 GPU, 当前 {gpu_count}"))


def _pick_gpu() -> torch.device:
    """Pick a GPU device, distributing across workers when under pytest-xdist.

    When running with ``pytest -n <N>``, each worker process auto-selects a
    different GPU via ``PYTEST_XDIST_WORKER`` (set by xdist per process).
    Without xdist (or on the master process), prefers the GPU that had the
    most free memory when the test session started.
    Falls back to ``cpu`` when no GPU is allocatable.

    Uses :func:`cuda_usable` so that GPUs visible-but-OOM are skipped.
    """
    gpu_devices = get_all_gpu_devices()
    if not gpu_devices:
        return torch.device("cpu")
    if DEFAULT_DEVICE in gpu_devices:
        gpu_devices.remove(DEFAULT_DEVICE)
        gpu_devices.insert(0, DEFAULT_DEVICE)

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    worker_index = int(worker.replace("gw", "")) if worker != "master" else 0
    return gpu_devices[worker_index % len(gpu_devices)]


@pytest.fixture(scope="session")
def device():
    """Returns a single GPU device (distributed across workers under xdist), else cpu.

    This fixture prioritises GPU: when CUDA is usable it returns the first
    GPU (or a worker-specific GPU under pytest-xdist).  Tests that need
    **all** available GPUs should use the ``all_gpu_devices`` fixture instead.
    """
    return _pick_gpu()


@pytest.fixture(scope="session")
def gpu_count():
    """Returns the number of **usable** GPUs (0 if CUDA is visible-but-OOM)."""
    return cuda_device_count()


@pytest.fixture(scope="session")
def cuda_available():
    """Returns True if CUDA is available and usable."""
    return cuda_usable()


@pytest.fixture(scope="session")
def all_gpu_devices():
    """Returns a list of all usable GPU devices (``[cuda:0, cuda:1, …]``).

    Prioritises GPU usage — when CUDA is available this returns every GPU
    device.  When CUDA is not usable, returns an empty list.  Tests that
    want to leverage multiple GPUs (distributed training, sharding, etc.)
    should use this fixture and skip if the list is empty.

    Example::

        def test_distributed(all_gpu_devices):
            if len(all_gpu_devices) < 2:
                pytest.skip("Need at least 2 GPUs")
            # Use all_gpu_devices for distributed work
    """
    return get_all_gpu_devices()


@pytest.fixture(scope="session")
def all_devices():
    """Returns all testable devices, GPU-first.

    If CUDA is usable, returns ``[cuda:0, cuda:1, …]`` — every GPU.
    If CUDA is not usable, returns ``[cpu]``.

    Tests that should run on **all** available devices (GPU preferred)
    should parametrise on this fixture::

        @pytest.mark.parametrize("device", <all_devices>)
    """
    devices = get_all_gpu_devices()
    return devices or [torch.device("cpu")]


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset torch random seed before each test to ensure reproducibility.

    Only seeds CUDA when it is *usable* (not just visible), avoiding
    ``AcceleratorError`` on GPUs that report as available but have 0 VRAM.
    """
    torch.manual_seed(42)
    if cuda_usable():
        torch.cuda.manual_seed_all(42)
    yield
    torch.manual_seed(42)
    if cuda_usable():
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


@pytest.fixture
def stub_tokenizer():
    """Minimal tokenizer for generation/serving tests."""
    return StubTokenizer()


@pytest.fixture
def line_tokenizer():
    """Ord-based tokenizer for streaming tests outside tests/data/."""
    return LineTokenizer()


@pytest.fixture
def model_and_tokenizer(device):
    """Real character tokenizer + small decoder for inference tests."""
    tokenizer = SimpleCharacterTokenizer(DEFAULT_INFERENCE_CORPUS)
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        device=device,
    )
    return model, tokenizer
