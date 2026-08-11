"""
Multi-GPU DDP (Distributed Data Parallel) Tests.

These tests verify that the training framework works correctly with multiple GPUs.
They will be automatically skipped if the required number of GPUs is not available.
"""

from __future__ import annotations

import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from tests.support.devices import DEFAULT_DEVICE, all_gpu_devices, cuda_usable

DDP_MIN_FREE_BYTES = 4 * 1024**3
DDP_JOIN_TIMEOUT_S = 120


def _free_port() -> int:
    """Bind an ephemeral port so consecutive DDP tests do not share MASTER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _release_parent_cuda_caches() -> None:
    """Drop parent-process CUDA caches so spawn workers are less likely to deadlock."""
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError, torch.AcceleratorError:
            continue


def setup_ddp_env() -> int:
    """Configure process-group env vars and return the chosen master port."""
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    # Fail fast instead of hanging forever when one rank dies during init.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return port


def ddp_test_worker(rank, world_size, device_indices, config, results):
    """Worker function for DDP test."""
    try:
        device_index = device_indices[rank]

        # Initialize distributed
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Set device
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

        # Create model and move to device
        model = DecoderModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_seq_len=config["max_seq_len"],
            device=device,
        )

        # Wrap with DDP
        model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[device_index])

        # Simple forward pass
        batch_size = config["batch_size"]
        seq_len = config["max_seq_len"]
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)

        model_ddp.train()
        logits = model_ddp(input_ids)

        # Check output shape
        expected_shape = (batch_size, seq_len, config["vocab_size"])
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

        # Backward pass
        loss = logits.mean()
        loss.backward()

        # Verify gradients exist
        for param in model_ddp.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient should exist"

        results[rank] = {"success": True, "device": str(device)}

    except (RuntimeError, OSError, ValueError, TypeError) as e:
        results[rank] = {"success": False, "error": str(e)}

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_ddp(world_size: int, device_indices: list[int], config: dict) -> None:
    """Spawn DDP workers with a join timeout so a stuck NCCL init fails the test."""
    assert all(index is not None for index in device_indices)
    _release_parent_cuda_caches()
    setup_ddp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        ddp_test_worker,
        args=(world_size, device_indices, config, results),
        nprocs=world_size,
        join=False,
    )
    assert context is not None

    # ProcessContext.join(timeout=...) returns False on timeout instead of raising.
    end_at = time.monotonic() + DDP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"DDP spawn exceeded {DDP_JOIN_TIMEOUT_S}s on devices {device_indices}")
        if context.join(timeout=remaining):
            break

    for rank in range(world_size):
        assert rank in results, f"Rank {rank} produced no result (devices={device_indices})"
        assert results[rank]["success"], f"Rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_ddp_two_gpu():
    """Test DDP with 2 GPUs."""
    gpu_devices = all_gpu_devices(min_free_bytes=DDP_MIN_FREE_BYTES)
    if len(gpu_devices) < 2:
        pytest.skip("需要至少 2 个 GPU")

    world_size = 2
    device_indices = [device.index for device in gpu_devices[:world_size]]
    config = {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "max_seq_len": 32,
        "batch_size": 4,
    }
    _run_ddp(world_size, device_indices, config)


@pytest.mark.full_cluster
@pytest.mark.slow
def test_ddp_eight_gpu():
    """Test DDP with all 8 GPUs (full cluster)."""
    gpu_devices = all_gpu_devices(min_free_bytes=DDP_MIN_FREE_BYTES)
    if len(gpu_devices) < 8:
        pytest.skip("需要 8 个 GPU")

    world_size = 8
    device_indices = [device.index for device in gpu_devices[:world_size]]
    config = {
        "vocab_size": 200,
        "hidden_size": 128,
        "num_layers": 3,
        "num_heads": 4,
        "max_seq_len": 64,
        "batch_size": 8,
    }
    _run_ddp(world_size, device_indices, config)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_ddp_all_available_gpus():
    """Test DDP with all usable GPUs that currently have enough free memory (2+)."""
    _release_parent_cuda_caches()
    gpu_devices = all_gpu_devices(min_free_bytes=DDP_MIN_FREE_BYTES)
    world_size = len(gpu_devices)
    if world_size < 2:
        pytest.skip("需要至少 2 个 GPU")
    device_indices = [device.index for device in gpu_devices]

    config = {
        "vocab_size": 150,
        "hidden_size": 96,
        "num_layers": 2,
        "num_heads": 2,
        "max_seq_len": 32,
        "batch_size": world_size,  # Scale batch with GPUs
    }
    _run_ddp(world_size, device_indices, config)


@pytest.mark.gpu
def test_single_gpu_training():
    """Test training on single GPU."""
    if not cuda_usable():
        pytest.skip("需要 GPU")

    device = DEFAULT_DEVICE

    model = DecoderModel(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for a few steps
    model.train()
    losses = []
    for _ in range(5):
        input_ids = torch.randint(0, 100, (4, 16), device=device)
        labels = torch.randint(0, 100, (4, 16), device=device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, 100), labels.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Verify training happened
    assert losses[-1] != losses[0], "Loss should change"
    assert not any(torch.isnan(torch.tensor(loss)) for loss in losses), "No NaN losses"


@pytest.mark.gpu
def test_gpu_memory_tracking():
    """Test that GPU memory tracking works correctly."""
    if not cuda_usable():
        pytest.skip("需要 GPU")

    device = DEFAULT_DEVICE

    # Record initial memory
    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device)

    # Create model
    model = DecoderModel(
        vocab_size=500,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=128,
        device=device,
    )

    model.train()

    # Forward + backward
    input_ids = torch.randint(0, 500, (8, 64), device=device)
    labels = torch.randint(0, 500, (8, 64), device=device)

    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, 500), labels.view(-1))
    loss.backward()

    # Check memory was allocated
    peak_mem = torch.cuda.max_memory_allocated(device)
    assert peak_mem > initial_mem, "Memory should be allocated"

    # Cleanup
    del model, logits, loss
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_multi_gpu_config_detection():
    """Test that GPU configuration is correctly detected."""
    if not cuda_usable():
        pytest.skip("需要 GPU")

    config = Config(
        model=ModelConfig(vocab_size=100, hidden_size=64, num_layers=2, num_heads=2, max_seq_len=32),
        training=TrainingConfig(batch_size=4, epochs=1),
        optimization=OptimizationConfig(use_amp=True),
    )

    # Check that distributed config detects GPUs
    gpus = torch.cuda.device_count()
    assert gpus >= 1, "Should detect at least 1 GPU"

    # Check gpus_per_node defaults correctly
    assert config.distributed.gpus_per_node == gpus, f"Expected {gpus}, got {config.distributed.gpus_per_node}"


def _ddp_weight_consistency_worker(rank, world_size, device_indices, results):
    """Build a model under per-rank seeding (as DistributedManager.setup does),
    broadcast via ``broadcast_parameters`` (as TrainingEngine does), and report
    whether every rank converged to identical weights."""
    try:
        import torch.distributed as _dist

        device_index = device_indices[rank]
        _dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

        # Exactly the framework's per-rank seed sequence.
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

        model = DecoderModel(
            vocab_size=200,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
            device=device,
        )
        # Without this broadcast, each rank's RNG-seeded init diverges and DDP
        # trains a gradient-mean of *different* models for the whole run.
        from llm.training.core.distributed import broadcast_parameters

        broadcast_parameters(model, src=0)

        weights = [param.detach().cpu() for param in model.parameters()]
        results[rank] = weights
        _dist.destroy_process_group()
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        results[rank] = f"ERR {e}"
    finally:
        if _dist.is_initialized():
            _dist.destroy_process_group()


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_ddp_ranks_share_identical_initial_weights():
    """Regression: fresh multi-GPU runs must start from identical weights.

    ``DistributedManager.setup`` seeds ``42 + rank`` before ``build_model()``
    so each rank RNG-initialises differently; DDP only averages gradients and
    never reconciles divergent initialisations. ``broadcast_parameters`` fixes
    the fresh-run path — here we assert every rank's weights are identical
    after it runs, regardless of the per-rank seeds.
    """
    import torch.distributed as _dist

    gpu_devices = all_gpu_devices(min_free_bytes=DDP_MIN_FREE_BYTES)
    if len(gpu_devices) < 2:
        pytest.skip("需要至少 2 个 GPU")

    world_size = 2
    device_indices = [d.index for d in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    setup_ddp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _ddp_weight_consistency_worker,
        args=(world_size, device_indices, results),
        nprocs=world_size,
        join=False,
    )
    assert context is not None
    end_at = time.monotonic() + DDP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"DDP spawn exceeded {DDP_JOIN_TIMEOUT_S}s on devices {device_indices}")
        if context.join(timeout=remaining):
            break

    for rank in range(world_size):
        assert rank in results, f"Rank {rank} produced no result (devices={device_indices})"
        assert not isinstance(results[rank], str), f"Rank {rank} failed: {results[rank]}"

    r0 = results[0]
    for rank in range(1, world_size):
        assert len(r0) == len(results[rank]), "Rank parameter count mismatch"
        for i, (a, b) in enumerate(zip(r0, results[rank], strict=True)):
            assert torch.equal(a, b), f"Rank {rank}, param {i} diverged from rank 0"
    del _dist  # import used only inside the worker
