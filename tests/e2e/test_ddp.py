"""
Multi-GPU DDP (Distributed Data Parallel) Tests.

These tests verify that the training framework works correctly with multiple GPUs.
They will be automatically skipped if the required number of GPUs is not available.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig


def setup_ddp_env():
    """Setup environment variables for DDP."""
    os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_DEBUG"] = "WARN"


def ddp_test_worker(rank, world_size, config, results):
    """Worker function for DDP test."""
    try:
        # Initialize distributed
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

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
        model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

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

    except Exception as e:
        results[rank] = {"success": False, "error": str(e)}

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_ddp_two_gpu():
    """Test DDP with 2 GPUs."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("需要至少 2 个 GPU")

    setup_ddp_env()
    world_size = 2
    config = {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "max_seq_len": 32,
        "batch_size": 4,
    }

    # Use spawn to create processes
    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        ddp_test_worker,
        args=(world_size, config, results),
        nprocs=world_size,
        join=True,
    )

    # Check all ranks succeeded
    for rank in range(world_size):
        assert results[rank]["success"], f"Rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.full_cluster
@pytest.mark.slow
def test_ddp_eight_gpu():
    """Test DDP with all 8 GPUs (full cluster)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("需要 8 个 GPU")

    setup_ddp_env()
    world_size = 8
    config = {
        "vocab_size": 200,
        "hidden_size": 128,
        "num_layers": 3,
        "num_heads": 4,
        "max_seq_len": 64,
        "batch_size": 8,
    }

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        ddp_test_worker,
        args=(world_size, config, results),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        assert results[rank]["success"], f"Rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_ddp_all_available_gpus():
    """Test DDP with all available GPUs (2+)."""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")

    world_size = torch.cuda.device_count()
    if world_size < 2:
        pytest.skip("需要至少 2 个 GPU")

    setup_ddp_env()
    config = {
        "vocab_size": 150,
        "hidden_size": 96,
        "num_layers": 2,
        "num_heads": 2,
        "max_seq_len": 32,
        "batch_size": world_size,  # Scale batch with GPUs
    }

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        ddp_test_worker,
        args=(world_size, config, results),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        assert results[rank]["success"], f"Rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.gpu
def test_single_gpu_training():
    """Test training on single GPU."""
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")

    device = torch.device("cuda:0")

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
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")

    device = torch.device("cuda:0")

    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()

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
    peak_mem = torch.cuda.max_memory_allocated()
    assert peak_mem > initial_mem, "Memory should be allocated"

    # Cleanup
    del model, logits, loss
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_multi_gpu_config_detection():
    """Test that GPU configuration is correctly detected."""
    if not torch.cuda.is_available():
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
