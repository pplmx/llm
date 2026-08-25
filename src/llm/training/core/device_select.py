"""Best-effort CUDA device selection preferring the GPU with the most free VRAM.

On shared/busy multi-GPU hosts the naive mapping ``rank -> rank % device_count``
always lands rank 0 on ``cuda:0`` — which may be the most contended device —
so a run OOMs even though idle GPUs exist. :class:`TrainingEngine` and the
distributed launcher therefore derive their CUDA index from the list of
*usable* devices sorted by free VRAM (descending; lower index wins ties), so
rank ``k`` takes the ``k``-th fattest device first. When devices are equally
free this reproduces the historical ``rank % device_count`` mapping exactly.

Every worker computes the same ``torch.cuda.mem_get_info`` snapshot at spawn
time, so the order is consistent across simultaneously-spawned ranks.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = [
    "MIN_FREE_VRAM_BYTES",
    "_cuda_usable_impl",
    "select_cuda_index",
    "sort_by_free_vram",
    "usable_cuda_indices",
]

# 512 MiB headroom for CUDA context + model (mirrors the engine's guard).
MIN_FREE_VRAM_BYTES = 512 * 1024 * 1024

FreeBytesFn = Callable[[int], int | None]


def _free_bytes(device_index: int) -> int | None:
    """Free bytes on a CUDA device, or ``None`` when inaccessible."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
    except RuntimeError, torch.AcceleratorError:
        return None
    return free_bytes


def _resolve_fn(free_bytes_fn: FreeBytesFn | None) -> FreeBytesFn:
    """Return the supplied free-bytes getter or the module default (call-time bound).

    Resolving at call time (rather than a ``def``-time default) lets tests
    monkeypatch :func:`_free_bytes` and have every caller observe the new
    reading.
    """
    return _free_bytes if free_bytes_fn is None else free_bytes_fn


def _cuda_usable_impl(
    device_idx: int,
    *,
    free_bytes_fn: FreeBytesFn | None = None,
    min_free_bytes: int = MIN_FREE_VRAM_BYTES,
) -> bool:
    """True only if ``device_idx`` is a visible CUDA device with usable VRAM.

    ``torch.cuda.is_available()`` can return True in containers that report
    CUDA devices but have 0 usable VRAM (CUDA OOM on first allocation); the
    driver also reserves context memory, so a device reporting only a few
    hundred MiB free will OOM on the first real allocation. This rejects such
    devices by requiring ``>= min_free_bytes`` of free VRAM.
    """
    if not torch.cuda.is_available():
        return False
    if device_idx >= torch.cuda.device_count():
        return False
    free_bytes = _resolve_fn(free_bytes_fn)(device_idx)
    return free_bytes is not None and free_bytes >= min_free_bytes


def usable_cuda_indices(
    n_devices: int | None = None,
    *,
    free_bytes_fn: FreeBytesFn | None = None,
    min_free_bytes: int = MIN_FREE_VRAM_BYTES,
) -> list[int]:
    """Visible CUDA indices meeting the free-memory requirement, sorted by free VRAM descending.

    The GPU with the most free VRAM comes first; a tie prefers the lower
    device index so selection is deterministic across processes.
    """
    fn = _resolve_fn(free_bytes_fn)
    if n_devices is None:
        if not torch.cuda.is_available():
            return []
        n_devices = torch.cuda.device_count()
    indices = [index for index in range(n_devices) if fn(index) is not None]
    return sort_by_free_vram(indices, free_bytes_fn=fn, min_free_bytes=min_free_bytes)


def sort_by_free_vram(
    indices: list[int],
    *,
    free_bytes_fn: FreeBytesFn | None = None,
    min_free_bytes: int = MIN_FREE_VRAM_BYTES,
) -> list[int]:
    """Filter ``indices`` to those with ``>= min_free_bytes`` and sort free-VRAM descending."""
    fn = _resolve_fn(free_bytes_fn)
    usable = [index for index in indices if (free := fn(index)) is not None and free >= min_free_bytes]
    return sorted(usable, key=lambda index: ((fn(index) or 0), -index), reverse=True)


def select_cuda_index(
    local_rank: int,
    *,
    n_devices: int | None = None,
    free_bytes_fn: FreeBytesFn | None = None,
    min_free_bytes: int = MIN_FREE_VRAM_BYTES,
) -> int | None:
    """CUDA index for ``local_rank``: the ``local_rank``-th fattest usable GPU.

    Ranks beyond the usable count round-robin over the fat GPUs (matching the
    historical ``local_rank % device_count`` collision behaviour). Returns
    ``None`` when no device meets the free-memory floor.
    """
    ordered = usable_cuda_indices(n_devices, free_bytes_fn=free_bytes_fn, min_free_bytes=min_free_bytes)
    if not ordered:
        return None
    return ordered[local_rank % len(ordered)]
