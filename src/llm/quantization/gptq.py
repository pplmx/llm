"""
GPTQ (Frantar et al. 2022) post-training quantization.

Provides 4-bit / 8-bit Hessian-aware quantization orthogonal to
the simple-PTQ path in `ptq.py`.
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from llm.quantization.calibration import CalibrationDataCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPTQConfig:
    """Configuration for GPTQ quantization.

    Attributes:
        bits: Quantization bit width (4 or 8).
        group_size: Quantization group size along input dim.
            -1 means per-channel (one scale per output row).
            Positive integer g means one scale per g consecutive input cols.
        sym: If True, symmetric quantization (no zero-point).
        percdamp: Hessian damping as a fraction of mean(diag(H)).
            Prevents numerical issues when H is near-singular.
        blocksize: Number of weight columns processed per Cholesky block.
            Larger = faster but more memory. Must be divisible by group_size
            when group_size > 0.
        act_order: If True, sort weight columns by diag(H) descending
            before quantization. Improves accuracy at slight cost.
        static_groups: If True, compute group partitions once and reuse
            across all layers. Faster, slight accuracy loss.
    """

    bits: int = 4
    group_size: int = 128
    sym: bool = True
    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = False
    static_groups: bool = False

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(
                f"GPTQConfig.bits must be 4 or 8, got {self.bits}. "
                f"For mixed precision, use target_modules to skip sensitive layers."
            )
        if self.group_size != -1 and self.group_size < 0:
            raise ValueError(
                f"group_size must be -1 (per-channel) or positive, got {self.group_size}."
            )
        if not (0.0 < self.percdamp < 1.0):
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}.")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}.")
        if self.group_size > 0 and self.blocksize % self.group_size != 0:
            raise ValueError(
                f"blocksize ({self.blocksize}) must be divisible by "
                f"group_size ({self.group_size}) for correct packing alignment."
            )
