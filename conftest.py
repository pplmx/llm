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
    except (OSError, subprocess.TimeoutExpired):
        return True


if not _has_usable_gpu_memory():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
