"""Shared GPU device detection and selection utilities for tests.

GPUs are selected **by remaining VRAM**: a device is only considered usable
when it has at least :data:`MIN_FREE_VRAM_BYTES` of free memory, and device
selection prefers the GPU with the most free VRAM.  This mirrors the guard in
:class:`llm.training.core.engine.TrainingEngine` so that a device the test
harness picks is never rejected (and OOM) by the engine itself.
"""

from __future__ import annotations

import torch

__all__ = [
    "ALL_DEVICES",
    "ALL_GPU_DEVICES",
    "DEFAULT_DEVICE",
    "MIN_FREE_VRAM_BYTES",
    "all_devices",
    "all_gpu_devices",
    "cuda_device_count",
    "cuda_device_strings",
    "cuda_usable",
    "gpu_vram_info",
    "multi_gpu_devices",
    "select_gpus_by_free_vram",
]

# 512 MiB floor for a GPU to be considered usable by tests.  Mirrors the
# ``_MIN_FREE_VRAM_BYTES`` guard in ``TrainingEngine._cuda_usable`` so that a
# device the test harness picks won't be rejected (and OOM) by the engine.
MIN_FREE_VRAM_BYTES = 512 * 1024 * 1024


def _free_memory(device_index: int) -> int | None:
    """Return free bytes for a CUDA device, or ``None`` when inaccessible."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
    except RuntimeError, torch.AcceleratorError:
        return None
    return free_bytes


def _usable_device_indices(min_free_bytes: int = MIN_FREE_VRAM_BYTES) -> list[int]:
    """Return visible CUDA indices meeting the free-memory requirement."""
    if not torch.cuda.is_available():
        return []
    return [
        index
        for index in range(torch.cuda.device_count())
        if (free_bytes := _free_memory(index)) is not None and free_bytes >= min_free_bytes
    ]


def cuda_usable(device: torch.device | str | int | None = None, *, min_free_bytes: int = MIN_FREE_VRAM_BYTES) -> bool:
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


def all_gpu_devices(*, min_free_bytes: int = MIN_FREE_VRAM_BYTES) -> list[torch.device]:
    """Return CUDA devices meeting the free-memory requirement, sorted by free VRAM (descending).

    The GPU with the *most* free VRAM comes first, so callers that need a
    single GPU (e.g. ``DEFAULT_DEVICE``) or want to fill slots greedily
    always pick from the fattest available device.
    """
    indices = _usable_device_indices(min_free_bytes)

    # Sort by free VRAM descending; tie-break by lower device index for
    # deterministic selection (matches _default_device semantics).
    def _vram(index: int) -> tuple[int, int]:
        free = _free_memory(index) or 0
        return (free, -index)

    indices.sort(key=_vram, reverse=True)
    return [torch.device(f"cuda:{index}") for index in indices]


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

    Convenience alias for tests that parametrize on device *strings*
    rather than ``torch.device`` objects.
    """
    return [str(device) for device in all_gpu_devices()]


def _default_device() -> torch.device:
    """Select the usable GPU with the most free memory, or CPU.

    VRAM is queried via :func:`torch.cuda.mem_get_info`; only devices with at
    least :data:`MIN_FREE_VRAM_BYTES` free are considered.  On a free-memory
    tie, prefer the lower device index so selection is deterministic.
    """
    candidates = [
        (free_bytes, index) for index in _usable_device_indices() if (free_bytes := _free_memory(index)) is not None
    ]
    if not candidates:
        return torch.device("cpu")
    _, index = max(candidates, key=lambda item: (item[0], -item[1]))
    return torch.device(f"cuda:{index}")


def gpu_vram_info(*, min_free_bytes: int = MIN_FREE_VRAM_BYTES) -> list[tuple[int, int]]:
    """Return ``[(device_index, free_bytes), …]`` for usable GPUs sorted by free VRAM (descending).

    Useful for diagnostics, logging, and test selection.  Only devices
    with at least ``min_free_bytes`` free are included.
    """
    indices = _usable_device_indices(min_free_bytes)
    return sorted(
        ((index, _free_memory(index) or 0) for index in indices),
        key=lambda item: (item[1], -item[0]),
        reverse=True,
    )


def multi_gpu_devices(*, min_free_bytes: int = MIN_FREE_VRAM_BYTES) -> list[torch.device]:
    """Return all usable GPU devices sorted by free VRAM (descending).

    Alias for :func:`all_gpu_devices` with an explicit name that signals
    the intent: tests that want to leverage *every* GPU should use this
    and skip if fewer than the desired number are available.

    Example::

        def test_distributed_multi_gpu(multi_gpu_devices):
            if len(multi_gpu_devices) < 2:
                pytest.skip("Need at least 2 GPUs")
    """
    return all_gpu_devices(min_free_bytes=min_free_bytes)


def select_gpus_by_free_vram(
    max_gpus: int | None = None, *, min_free_bytes: int = MIN_FREE_VRAM_BYTES
) -> list[torch.device]:
    """Select up to ``max_gpus`` devices, prioritising free VRAM.

    Returns the fattest GPUs first.  When ``max_gpus`` is ``None`` all
    usable GPUs are returned.  Tests that need a subset (e.g. 2 GPUs for
    a distributed test) should use this so the worker GPUs always have
    the most headroom.
    """
    devices = all_gpu_devices(min_free_bytes=min_free_bytes)
    if max_gpus is not None:
        devices = devices[:max_gpus]
    return devices


DEFAULT_DEVICE = _default_device()
ALL_GPU_DEVICES: list[str] = cuda_device_strings()
ALL_DEVICES: list[str] = ALL_GPU_DEVICES or ["cpu"]
