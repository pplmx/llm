"""Root-level pytest conftest.

Runs *before* any test module or tests/conftest.py import torch. We use
this window to inspect GPU memory via ``nvidia-smi`` and, when no GPU has
usable free memory, set ``CUDA_VISIBLE_DEVICES=""`` so that
``torch.cuda.is_available()`` returns ``False`` once PyTorch is imported.

This prevents ``torch.AcceleratorError: CUDA error: out of memory`` from
firing deep inside PyTorch's optimizer / accelerator APIs
(``_accelerator_graph_capture_health_check`` calls
``torch.accelerator.current_stream()`` which raises on a CUDA-visible-but-OOM device).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess


def _has_usable_gpu_memory() -> bool:
    """Return True if at least one GPU has enough free memory for PyTorch.

    Uses ``nvidia-smi`` (the same tool the driver exposes) so we don't
    need to import torch here. If ``nvidia-smi`` is unavailable we
    conservatively return True (don't hide CUDA).
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return True
    try:
        result = subprocess.run(  # noqa: S603 — nvidia_smi from shutil.which, args hardcoded
            [nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return True
        # PyTorch needs ~512 MiB just to initialize a CUDA context; below
        # that, the OOM surfaces inside the driver rather than as a clean
        # allocation failure we can catch. Require headroom above that.
        # nvidia-smi output looks like "319 MiB" — extract the numeric part.
        free_mems = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if parts:
                with contextlib.suppress(ValueError):
                    free_mems.append(int(parts[0]))
        return any(mem > 512 for mem in free_mems)
    except OSError, subprocess.TimeoutExpired:
        return True


def _disable_core_dumps() -> None:
    """Stop a crashing test process from writing a multi-GB core file.

    The default here is ``kernel.core_pattern=core`` with ``ulimit -c
    unlimited``, so ANY child that aborts — and the multi-GPU distributed
    tests spawn workers that abort on NCCL/CUDA errors — drops a full core
    file in the CWD. Measured on this repo's own 8xA100 box: six concurrent
    ranks produced 51 core files totalling **90 GB**, filling the volume to
    100% and making every *subsequent* test fail with ``OSError(28, 'No space
    left on device')`` — which then reads as an unrelated flaky failure in
    whatever test happens to run next.

    ``RLIMIT_CORE = 0`` is inherited by spawned children, so this covers the
    multi-process suites without touching the host's global sysctl (which
    would need root and would outlive the test run).
    """
    import resource

    with contextlib.suppress(ValueError, OSError):  # not settable on every platform
        resource.setrlimit(resource.RLIMIT_CORE, (0, resource.RLIM_INFINITY))


_disable_core_dumps()

if not _has_usable_gpu_memory():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
