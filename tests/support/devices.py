"""Shared GPU device detection and selection utilities for tests."""

from __future__ import annotations

import torch

__all__ = [
    "ALL_DEVICES",
    "ALL_GPU_DEVICES",
    "DEFAULT_DEVICE",
    "all_devices",
    "all_gpu_devices",
    "cuda_device_count",
    "cuda_device_strings",
    "cuda_usable",
]


def _free_memory(device_index: int) -> int | None:
    """Return free bytes for a CUDA device, or ``None`` when inaccessible."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
    except RuntimeError, torch.AcceleratorError:
        return None
    return free_bytes


def _usable_device_indices(min_free_bytes: int = 1) -> list[int]:
    """Return visible CUDA indices meeting the free-memory requirement."""
    if not torch.cuda.is_available():
        return []
    return [
        index
        for index in range(torch.cuda.device_count())
        if (free_bytes := _free_memory(index)) is not None and free_bytes >= min_free_bytes
    ]


def cuda_usable(device: torch.device | str | int | None = None, *, min_free_bytes: int = 1) -> bool:
    """Return whether at least one requested CUDA device has enough free VRAM."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        return bool(_usable_device_indices(min_free_bytes))

    resolved = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    if resolved.type != "cuda":
        return False
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    free_bytes = _free_memory(index)
    return free_bytes is not None and free_bytes >= min_free_bytes


def cuda_device_count() -> int:
    """Number of CUDA devices that are both visible **and** usable."""
    return len(_usable_device_indices())


def all_gpu_devices(*, min_free_bytes: int = 1) -> list[torch.device]:
    """Return CUDA devices meeting the free-memory requirement."""
    return [torch.device(f"cuda:{index}") for index in _usable_device_indices(min_free_bytes)]


def all_devices() -> list[torch.device]:
    """Return all testable devices.

    Prioritises GPU: if CUDA is usable, returns every GPU device;
    otherwise falls back to CPU.  Tests that want to verify behaviour
    across *all* available devices (GPU preferred) should parametrise
    on this list.
    """
    gpus = all_gpu_devices()
    return gpus if gpus else [torch.device("cpu")]


def cuda_device_strings() -> list[str]:
    """Return ``["cuda:0", "cuda:1", …]`` for every usable GPU.

    Convenience alias for tests that parametrise on device *strings*
    rather than ``torch.device`` objects.
    """
    return [str(device) for device in all_gpu_devices()]


def _default_device() -> torch.device:
    """Select the usable GPU with the most free memory, or CPU.

    On a free-memory tie, prefer the lower device index.
    """
    candidates = [
        (free_bytes, index) for index in _usable_device_indices() if (free_bytes := _free_memory(index)) is not None
    ]
    if not candidates:
        return torch.device("cpu")
    _, index = max(candidates, key=lambda item: (item[0], -item[1]))
    return torch.device(f"cuda:{index}")


DEFAULT_DEVICE = _default_device()
ALL_GPU_DEVICES: list[str] = cuda_device_strings()
ALL_DEVICES: list[str] = ALL_GPU_DEVICES or ["cpu"]
