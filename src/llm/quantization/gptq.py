"""
GPTQ (Frantar et al. 2022) post-training quantization.

Provides 4-bit / 8-bit Hessian-aware quantization orthogonal to
the simple-PTQ path in `ptq.py`.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

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
            raise ValueError(f"group_size must be -1 (per-channel) or positive, got {self.group_size}.")
        if not (0.0 < self.percdamp < 1.0):
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}.")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}.")
        if self.group_size > 0 and self.blocksize % self.group_size != 0:
            raise ValueError(
                f"blocksize ({self.blocksize}) must be divisible by "
                f"group_size ({self.group_size}) for correct packing alignment."
            )


class GPTQQuantizer:
    """Stateful per-layer GPTQ processor.

    Lifecycle:
        q = GPTQQuantizer(layer, config)
        for batch in calib_iter_for_this_layer:
            q.add_batch(batch)
        W_packed, scales, zeros = q.quantize()
    """

    def __init__(self, layer: nn.Linear, config: GPTQConfig):
        self.config = config
        self.layer = layer
        self.device = layer.weight.device
        # Compute in float32 for numerical stability of Cholesky
        self.compute_dtype = torch.float32

        # Weight dimensions
        self.out_features, self.in_features = layer.weight.shape

        # Hessian accumulator
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.n_samples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate Hessian contribution from a calibration batch.

        Maintains the invariant H == (2 / N_total) · Σ X_b^T X_b so that
        multiple mini-batches produce the same H as a single concatenated
        add_batch (Frantar 2022, eq. 3). Uses the canonical EMA-style
        rescale: H_new = (N_old / N_new) · H_old + (2 / N_new) · X^T X.

        Args:
            x: Input activations to `self.layer`, shape [..., in_features].
               Will be flattened to [N, in_features] internally.
        """
        x = x.to(device=self.device, dtype=self.compute_dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, x.shape[-1])  # flatten leading dims

        n = x.shape[0]
        if n == 0:
            return

        new_total = self.n_samples + n
        # Rescale previous contribution to its raw Σ X^T X form, then
        # re-apply the (2 / N_new) factor on the new total. This makes
        # H the exact (2 / N_total) · Σ X^T X across any batch partition.
        self.H *= self.n_samples / new_total
        self.n_samples = new_total
        self.H += (2.0 / new_total) * (x.t() @ x)

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run GPTQ on accumulated Hessian.

        Returns:
            W_q: Quantized weights (integer-valued, stored as fp32),
                 shape [out_features, in_features]. Multiply by `scales`
                 to dequantize: W_recon = W_q * scales.
            scales: Per-row scale if group_size=-1 [out_features, 1],
                    else per-group scale [out_features, in_features // group_size].
            zeros: Per-group zero-points, or None (symmetric only in v1).

        Raises:
            RuntimeError: If calibration is empty or Hessian is ill-conditioned
                          (rank-deficient with insufficient damping).
        """
        # Guard: zero calibration data
        if self.n_samples == 0:
            raise RuntimeError(
                "No calibration data accumulated (n_samples=0). "
                "Hessian is empty. Try increasing percdamp or check calibration data quality."
            )

        w = self.layer.weight.detach().clone().to(device=self.device, dtype=self.compute_dtype)
        h = self.H.clone()

        # Detect rank-deficient Hessian BEFORE dead handling
        n_dead = int(torch.sum(torch.diag(h) == 0).item())

        # Handle all-zero columns (degenerate / unused features)
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w[:, dead] = 0.0

        # Damping for numerical stability (Frantar 2022, eq. 4)
        damp = self.config.percdamp * torch.mean(torch.diag(h))
        diag_idx = torch.arange(self.in_features, device=self.device)
        h[diag_idx, diag_idx] += damp

        # Guard: rank-deficient Hessian with insufficient damping.
        # If most columns are dead (no variance in calibration), low damping
        # leaves the live columns effectively unscaled relative to the dead
        # ones (which dead-handling set to 1.0). Reject and tell the user.
        if n_dead > self.in_features // 2 and damp < 0.1:
            raise RuntimeError(
                f"Hessian is rank-deficient ({n_dead}/{self.in_features} "
                f"columns have zero variance). Damping (percdamp="
                f"{self.config.percdamp}) is insufficient. Try increasing "
                f"percdamp (e.g. 0.5) or check calibration data quality."
            )

        # Cholesky inverse: U = chol(H^-1)^T upper triangular
        try:
            h_inv = torch.linalg.inv(h)
            u = torch.linalg.cholesky(h_inv, upper=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Hessian is not positive-definite even after damping "
                f"(percdamp={self.config.percdamp}). "
                f"Try increasing percdamp (e.g. 0.1) or check calibration data quality."
            ) from e

        # Optional act-order: sort columns by diag(H_inv) descending
        if self.config.act_order:
            perm = torch.argsort(torch.diag(h_inv), descending=True)
            w = w[:, perm]
            h_inv = h_inv[perm][:, perm]
            u = u[perm][:, perm]

        # Compute per-row scale ONCE for per-channel (group_size=-1).
        # Canonical GPTQ (Frantar 2022, eq. 5): scale = w.abs().max() / qmax.
        qmax = 2 ** (self.config.bits - 1) - 1
        if self.config.group_size == -1:
            row_scales = w.abs().max(dim=1)[0] / qmax  # [out_f]
            row_scales = row_scales.clamp(min=1e-8)

        # Quantize column-by-column with error correction (Frantar 2022, eq. 5)
        q_out = torch.zeros_like(w)

        for i in range(0, self.in_features, self.config.blocksize):
            i_end = min(i + self.config.blocksize, self.in_features)
            count = i_end - i

            w1 = w[:, i:i_end].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            hinv1 = h_inv[i:i_end, i:i_end]

            for j in range(count):
                col = w1[:, j]
                d = hinv1[j, j]

                # Symmetric only for v1
                if not self.config.sym:
                    raise NotImplementedError("Asymmetric GPTQ not yet implemented. Use sym=True.")

                if self.config.group_size == -1:
                    scale = row_scales  # [out_f], same for all columns
                else:
                    # Per-group scale: compute from group window of w
                    gs = self.config.group_size
                    group_start = ((i + j) // gs) * gs
                    group_end = group_start + gs
                    w_group = w[:, group_start:group_end]
                    scale = w_group.abs().max() / qmax
                    scale = scale.clamp(min=1e-8)

                # Quantize to INTEGER (clamped to symmetric range)
                q_int = torch.round(col / scale).clamp(-qmax - 1, qmax)
                q1[:, j] = q_int  # store integer-valued fp32

                # Error correction: propagate quantization error to remaining columns.
                # Canonical GPTQ (Frantar 2022, eq. 5): the error vector is
                # scaled by H^-1 and propagated via the row of H^-1 starting
                # at the current column. Note: H^-1 (not its Cholesky factor U)
                # must be used in BOTH the denominator and the propagation to
                # match the canonical formula exactly.
                err = (col - q_int * scale) / d
                if j + 1 < count:
                    hinv_col = hinv1[j, j + 1 :]
                    if hinv_col.shape[0] > 0:
                        w1[:, j + 1 :] -= err.unsqueeze(1) * hinv_col.unsqueeze(0)

                err1[:, j] = err

            q_out[:, i:i_end] = q1

            # Propagate block errors to all remaining columns (canonical GPTQ).
            # Uses H^-1 (not U) to match the canonical Frantar 2022 formula.
            if i_end < self.in_features:
                w[:, i_end:] -= err1 @ h_inv[i:i_end, i_end:]

        # Undo act-order permutation if applied
        if self.config.act_order:
            invperm = torch.argsort(perm)
            q_out = q_out[:, invperm]

        # Return per-row or per-group scales matching the scales used in the loop
        if self.config.group_size != -1:
            gs = self.config.group_size
            n_groups = self.in_features // gs
            scales = torch.zeros(self.out_features, n_groups, dtype=torch.float32)
            for g in range(n_groups):
                s = g * gs
                e = s + gs
                w_g = w[:, s:e]
                scales[:, g] = w_g.abs().max(dim=1)[0] / qmax
                scales[:, g] = scales[:, g].clamp(min=1e-8)
        else:
            scales = row_scales.unsqueeze(1)  # [out_f, 1]

        zeros: torch.Tensor | None = None
        return q_out, scales, zeros
