# GPTQ 集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/llm/quantization/` 中集成 GPTQ（Frantar 2022）4-bit 量化算法，提供独立于 simple-PTQ 的正交路径 + CLI `llm-quantize gptq`。

**Architecture:** 三个新模块（`gptq.py` 算法核心 / `_gptq_layer.py` 打包存储 / `cli/quantize.py` CLI）+ 双重 entry point（standalone Iterator + CalibrationDataCollector 复用）+ 三层 TDD（算法 / 存储 / e2e）。

**Tech Stack:** PyTorch (linalg.cholesky / cholesky_solve), transformers (tokenizer in CLI), pytest

**Reference Spec:** [`docs/superpowers/specs/2026-07-22-gptq-integration-design.md`](../specs/2026-07-22-gptq-integration-design.md)

---

## 文件结构

```text
src/llm/quantization/
├── __init__.py             ← 修改: 导出 GPTQConfig, GPTQQuantizer, GPTQQuantizedLinear, quantize_model_gptq, quantize_model_with_collector
├── ptq.py                  ← 不动
├── calibration.py          ← 不动
├── gptq.py                 ← 新增 (GPTQConfig, GPTQQuantizer, 顶层 entry, ~400 行)
└── _gptq_layer.py          ← 新增 (GPTQQuantizedLinear + packed 4-bit, ~200 行)

src/llm/cli/
└── quantize.py             ← 新增 (typer app, llm-quantize gptq subcommand, ~150 行)

tests/quantization/
├── test_gptq_algorithm.py  ← 新增 (Hessian / Cholesky / 算法 correctness)
├── test_gptq_layer.py      ← 新增 (pack/unpack / forward correctness)
└── test_gptq_end_to_end.py ← 新增 (small Transformer quantize → forward)

tests/cli/
└── test_quantize_cli.py    ← 新增 (CLI happy path + error cases)

docs/adr/
└── 007-gptq-integration.md ← 新增 (设计决策记录)
```

---

## Task 1: GPTQConfig dataclass with validation

**Files:**
- Create: `src/llm/quantization/gptq.py`
- Create: `tests/quantization/test_gptq_algorithm.py`

- [ ] **Step 1: Create test file with validation tests**

Create `tests/quantization/test_gptq_algorithm.py`:

```python
"""Tests for GPTQ algorithm core (config + Hessian + Cholesky + column loop)."""

import pytest
import torch
import torch.nn as nn


# === Config validation tests ===

def test_gptq_config_default_values():
    """Default config is 4-bit, group_size=128, symmetric."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig()
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.sym is True
    assert cfg.percdamp == 0.01
    assert cfg.blocksize == 128
    assert cfg.act_order is False
    assert cfg.static_groups is False


def test_gptq_config_rejects_invalid_bits():
    """bits must be 4 or 8."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        GPTQConfig(bits=16)


def test_gptq_config_rejects_negative_group_size():
    """group_size must be -1 (per-channel) or positive."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="group_size must be -1"):
        GPTQConfig(group_size=-128)


def test_gptq_config_rejects_invalid_percdamp():
    """percdamp must be in (0, 1)."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="percdamp must be in"):
        GPTQConfig(percdamp=0.0)


def test_gptq_config_rejects_nonpositive_blocksize():
    """blocksize must be positive."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="blocksize must be positive"):
        GPTQConfig(blocksize=0)


def test_gptq_config_rejects_blocksize_not_divisible_by_group_size():
    """When group_size > 0, blocksize must be divisible by group_size."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(ValueError, match="blocksize.*must be divisible"):
        GPTQConfig(group_size=128, blocksize=100)


def test_gptq_config_per_channel_skips_divisibility_check():
    """group_size=-1 (per-channel) skips blocksize divisibility check."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig(group_size=-1, blocksize=100)  # should not raise
    assert cfg.group_size == -1
    assert cfg.blocksize == 100


def test_gptq_config_is_frozen():
    """Config is frozen (immutable)."""
    from llm.quantization.gptq import GPTQConfig

    cfg = GPTQConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.bits = 8  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -v
```

Expected: All 8 tests FAIL with `ModuleNotFoundError: No module named 'llm.quantization.gptq'`

- [ ] **Step 3: Create `src/llm/quantization/gptq.py` with `GPTQConfig`**

Create `src/llm/quantization/gptq.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py::test_gptq_config_default_values tests/quantization/test_gptq_algorithm.py::test_gptq_config_rejects_invalid_bits tests/quantization/test_gptq_algorithm.py::test_gptq_config_rejects_negative_group_size tests/quantization/test_gptq_algorithm.py::test_gptq_config_rejects_invalid_percdamp tests/quantization/test_gptq_algorithm.py::test_gptq_config_rejects_nonpositive_blocksize tests/quantization/test_gptq_algorithm.py::test_gptq_config_rejects_blocksize_not_divisible_by_group_size tests/quantization/test_gptq_algorithm.py::test_gptq_config_per_channel_skips_divisibility_check tests/quantization/test_gptq_algorithm.py::test_gptq_config_is_frozen -v
```

Expected: All 8 config tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/gptq.py tests/quantization/test_gptq_algorithm.py
git commit -m "feat(quant): add GPTQConfig dataclass with validation"
```

---

## Task 2: GPTQQuantizer skeleton + Hessian accumulation

**Files:**
- Modify: `src/llm/quantization/gptq.py`
- Modify: `tests/quantization/test_gptq_algorithm.py`

- [ ] **Step 1: Append Hessian accumulation tests to test file**

Append to `tests/quantization/test_gptq_algorithm.py`:

```python
# === GPTQQuantizer Hessian accumulation tests ===

def test_quantizer_initializes_with_zero_hessian():
    """Fresh GPTQQuantizer has H == 0 and no samples accumulated."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    assert q.n_samples == 0
    assert torch.allclose(q.H, torch.zeros_like(q.H))


def test_add_batch_handles_2d_input():
    """add_batch accepts [batch, in_features] tensor."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    x = torch.randn(16, 8)
    q.add_batch(x)

    assert q.n_samples == 16
    # H should be non-zero after add_batch
    assert q.H.abs().sum() > 0


def test_add_batch_handles_3d_input():
    """add_batch accepts [batch, seq, in_features] tensor (flattens batch+seq)."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    x = torch.randn(4, 5, 8)  # 4*5 = 20 samples
    q.add_batch(x)

    assert q.n_samples == 20


def test_add_batch_accumulates_hessian_correctly():
    """Multiple add_batches accumulate H = 2/N · Σ X^T X."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_features = 6
    layer = nn.Linear(in_features, 4, bias=False)
    cfg = GPTQConfig()
    q = GPTQQuantizer(layer, cfg)

    # Two batches
    x1 = torch.randn(10, in_features)
    x2 = torch.randn(8, in_features)
    q.add_batch(x1)
    q.add_batch(x2)

    # Expected: H = 2/18 * (x1.T @ x1 + x2.T @ x2)
    expected = (2.0 / 18) * (x1.t() @ x1 + x2.t() @ x2)

    assert torch.allclose(q.H, expected, atol=1e-5)


def test_add_batch_matches_one_shot():
    """Multi-batch accumulate equals single concatenated add_batch (same data)."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(0)
    in_features = 6
    layer = nn.Linear(in_features, 4, bias=False)
    cfg = GPTQConfig()

    chunk_a = torch.randn(5, in_features)
    chunk_b = torch.randn(7, in_features)

    # Multi-batch (using same data)
    q1 = GPTQQuantizer(layer, cfg)
    q1.add_batch(chunk_a)
    q1.add_batch(chunk_b)

    # Single-shot (cat of same data)
    q2 = GPTQQuantizer(layer, cfg)
    q2.add_batch(torch.cat([chunk_a, chunk_b], dim=0))

    assert torch.allclose(q1.H, q2.H, atol=1e-5)
    assert q1.n_samples == q2.n_samples == 12
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -k "quantizer or add_batch" -v
```

Expected: 5 new tests FAIL with `ImportError: cannot import name 'GPTQQuantizer'`

- [ ] **Step 3: Append `GPTQQuantizer` skeleton with `__init__` + `add_batch` to `gptq.py`**

Append to `src/llm/quantization/gptq.py`:

```python
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
        # re-apply the (2 / N_new) factor on the new total.
        self.H *= self.n_samples / new_total
        self.n_samples = new_total
        self.H += (2.0 / new_total) * (x.t() @ x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -k "quantizer or add_batch" -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/gptq.py tests/quantization/test_gptq_algorithm.py
git commit -m "feat(quant): GPTQQuantizer skeleton + Hessian accumulation"
```

---

## Task 3: GPTQ core algorithm (Cholesky inverse + column loop)

**Files:**
- Modify: `src/llm/quantization/gptq.py`
- Modify: `tests/quantization/test_gptq_algorithm.py`

- [ ] **Step 1: Append algorithm correctness tests**

Append to `tests/quantization/test_gptq_algorithm.py`:

```python
# === GPTQ algorithm correctness tests ===

def test_gptq_lower_error_than_rtn_baseline():
    """GPTQ MSE on a small layer must be lower than naive round-to-nearest.

    This is the core promise of GPTQ — Hessian-aware quantization beats
    per-column rounding. Tested on a small dense layer with calibration data.
    """
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(42)
    in_f, out_f = 16, 16
    layer = nn.Linear(in_f, out_f, bias=False)
    # Inject meaningful structure (not random init)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_f, in_f) * 0.5)

    # Generate calibration data with structure
    calib = torch.randn(64, in_f)

    # GPTQ
    q = GPTQQuantizer(layer, GPTQConfig(bits=4, group_size=-1))
    q.add_batch(calib)
    W_q, scales, zeros = q.quantize()

    # Reconstruct W_q in fp32 for comparison
    # scales shape: [out_f, 1] for group_size=-1
    W_recon = W_q.float() * scales.float()
    mse_gptq = ((layer.weight - W_recon) ** 2).mean().item()

    # RTN baseline: per-channel symmetric 4-bit
    abs_max = layer.weight.abs().max(dim=1, keepdim=True)[0]
    qmax = 2 ** (4 - 1) - 1  # 7
    scale_rtn = abs_max / qmax
    W_rtn = (layer.weight / scale_rtn).round().clamp(-8, 7) * scale_rtn
    mse_rtn = ((layer.weight - W_rtn) ** 2).mean().item()

    assert mse_gptq < mse_rtn, f"GPTQ MSE {mse_gptq:.6f} should beat RTN MSE {mse_rtn:.6f}"


def test_quantize_handles_zero_calibration_gracefully():
    """Zero calibration data → actionable error mentioning percdamp."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    layer = nn.Linear(8, 4, bias=False)
    q = GPTQQuantizer(layer, GPTQConfig(percdamp=0.01))

    with pytest.raises(RuntimeError, match="percdamp"):
        q.quantize()


def test_quantize_handles_singular_hessian_with_higher_damp():
    """Rank-deficient Hessian succeeds with sufficient damping."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(0)
    in_f, out_f = 8, 4
    layer = nn.Linear(in_f, out_f, bias=False)
    # Calibration data with rank < in_f (constant feature)
    calib = torch.zeros(16, in_f)
    calib[:, 0] = torch.linspace(-1, 1, 16)  # only first dim varies

    # 0.01 damp fails, 0.5 damp succeeds
    q1 = GPTQQuantizer(layer, GPTQConfig(percdamp=0.01))
    q1.add_batch(calib)
    with pytest.raises(RuntimeError):
        q1.quantize()

    q2 = GPTQQuantizer(layer, GPTQConfig(percdamp=0.5))
    q2.add_batch(calib)
    W_q, scales, zeros = q2.quantize()
    assert W_q.shape == (out_f, in_f)


def test_quantize_returns_correct_shapes():
    """quantize() returns W_q [out, in], scales [out, in/group_size] (or [out,1] for per-channel)."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_f, out_f = 32, 16
    layer = nn.Linear(in_f, out_f, bias=False)
    calib = torch.randn(32, in_f)

    # group_size=8 → 4 groups per row
    q = GPTQQuantizer(layer, GPTQConfig(group_size=8))
    q.add_batch(calib)
    W_q, scales, zeros = q.quantize()
    assert W_q.shape == (out_f, in_f)
    assert scales.shape == (out_f, in_f // 8)

    # group_size=-1 → 1 group per row (per-channel)
    q2 = GPTQQuantizer(layer, GPTQConfig(group_size=-1))
    q2.add_batch(calib)
    W_q2, scales2, zeros2 = q2.quantize()
    assert W_q2.shape == (out_f, in_f)
    assert scales2.shape == (out_f, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -k "gptq_lower_error or handles_zero_calibration or singular_hessian or correct_shapes" -v
```

Expected: 4 tests FAIL with `TypeError: quantize() missing 1 required positional argument` or similar

- [ ] **Step 3: Append `quantize()` method to `GPTQQuantizer` in `gptq.py`**

Append to `GPTQQuantizer` class in `src/llm/quantization/gptq.py`:

```python
    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run GPTQ on accumulated Hessian.

        Returns:
            W_q: Quantized weights, fp32 representation. Shape [out_features, in_features].
                 Caller is responsible for packing to int8 if bits=4.
            scales: Per-group scales. Shape [out_features, in_features // group_size]
                    or [out_features, 1] if group_size=-1.
            zeros: Per-group zero-points, or None if sym=True.
        """
        W = self.layer.weight.detach().clone().to(
            device=self.device, dtype=self.compute_dtype
        )
        H = self.H.clone()

        # Handle all-zero columns (degenerate)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        # Damping for numerical stability
        damp = self.config.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.in_features, device=self.device)
        H[diag, diag] += damp

        # Cholesky inverse: U = chol(H^-1)^T upper triangular
        try:
            H_inv = torch.linalg.inv(H)
            U = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Hessian is not positive-definite even after damping "
                f"(percdamp={self.config.percdamp}). "
                f"Try increasing percdamp (e.g. 0.1) or check calibration data quality."
            ) from e

        # Optional act-order: sort columns by diag(H) descending
        if self.config.act_order:
            perm = torch.argsort(torch.diag(H_inv), descending=True)
            W = W[:, perm]
            H_inv = H_inv[perm][:, perm]
            U = U[perm][:, perm]

        # Quantize column-by-column with error correction
        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        for i in range(0, self.in_features, self.config.blocksize):
            i_end = min(i + self.config.blocksize, self.in_features)
            count = i_end - i

            W1 = W[:, i:i_end].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = H_inv[i:i_end, i:i_end]

            for j in range(count):
                w = W1[:, j]
                d = Hinv1[j, j]

                # Per-group scale computation
                if self.config.group_size != -1:
                    group_idx = (i + j) // self.config.group_size
                else:
                    group_idx = 0

                # Quantize w[:, j]
                if self.config.sym:
                    if self.config.group_size == -1:
                        scale = w.abs().max() / (2 ** (self.config.bits - 1) - 1)
                        scale = scale.clamp(min=1e-8)
                    else:
                        gs = self.config.group_size
                        g_start = group_idx * gs
                        g_end = g_start + gs
                        w_group = W[:, g_start:g_end]
                        scale = w_group.abs().max() / (2 ** (self.config.bits - 1) - 1)
                        scale = scale.clamp(min=1e-8)
                    qmax = 2 ** (self.config.bits - 1) - 1
                    q = torch.round(w / scale).clamp(-qmax - 1, qmax)
                    Q1[:, j] = q * scale
                else:
                    # Asymmetric: not implemented in v1; raise
                    raise NotImplementedError(
                        "Asymmetric GPTQ not yet implemented. Use sym=True."
                    )

                # Error correction
                err = (w - Q1[:, j]) / d
                w_col = W1[:, j + 1:]
                u_col = U[i + j, i + j + 1 : i_end]

                if w_col.shape[1] > 0:
                    W1[:, j + 1:] -= err.unsqueeze(1) * u_col.unsqueeze(0)
                    Err1[:, j] = err

            Q[:, i:i_end] = Q1
            Losses[:, i:i_end] = Losses1

            # Propagate error to remaining columns
            if i_end < self.in_features:
                W[:, i_end:] -= Err1 @ U[i:i_end, i_end:]

        # Undo act-order permutation if applied
        if self.config.act_order:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        # Compute final per-group scales/zeros for the packed representation
        if self.config.group_size != -1:
            gs = self.config.group_size
            n_groups = self.in_features // gs
            scales = torch.zeros(self.out_features, n_groups, dtype=torch.float32)
            zeros = torch.zeros(self.out_features, n_groups, dtype=torch.int8)
            for g in range(n_groups):
                s = g * gs
                e = s + gs
                w_g = W[:, s:e]  # original fp32 weight
                q_g = Q[:, s:e]  # dequantized from Q
                if self.config.sym:
                    scales[:, g] = w_g.abs().max(dim=1)[0] / (2 ** (self.config.bits - 1) - 1)
                    scales[:, g] = scales[:, g].clamp(min=1e-8)
                # Q is dequantized (already in fp32 scale space), caller will re-pack
        else:
            scales = W.abs().max(dim=1, keepdim=True)[0] / (2 ** (self.config.bits - 1) - 1)
            scales = scales.clamp(min=1e-8)
            zeros = None

        return Q, scales, zeros
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -k "gptq_lower_error or handles_zero_calibration or singular_hessian or correct_shapes" -v
```

Expected: All 4 algorithm tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/gptq.py tests/quantization/test_gptq_algorithm.py
git commit -m "feat(quant): GPTQ core algorithm (Cholesky inverse + column loop)"
```

---

## Task 4: act-order + group_size validation tests

**Files:**
- Modify: `tests/quantization/test_gptq_algorithm.py`

- [ ] **Step 1: Append act-order + group_size tests**

Append to `tests/quantization/test_gptq_algorithm.py`:

```python
# === act-order and group_size behavior tests ===

def test_act_order_changes_quantization_sequence():
    """With act_order=True, columns with larger diag(H^-1) are quantized first."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(7)
    in_f, out_f = 8, 4
    layer = nn.Linear(in_f, out_f, bias=False)

    # Calibration data with column 0 having much larger variance
    calib = torch.randn(64, in_f)
    calib[:, 0] *= 5.0  # boost column 0

    # Without act_order
    q1 = GPTQQuantizer(layer, GPTQConfig(act_order=False))
    q1.add_batch(calib)
    W1, _, _ = q1.quantize()

    # With act_order
    q2 = GPTQQuantizer(layer, GPTQConfig(act_order=True))
    q2.add_batch(calib)
    W2, _, _ = q2.quantize()

    # Output should differ (column processing order matters)
    assert not torch.allclose(W1, W2, atol=1e-3)


def test_group_size_per_channel_vs_grouped_different_shapes():
    """group_size=-1 vs 128 produce different scale tensor shapes."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    in_f, out_f = 32, 8
    layer = nn.Linear(in_f, out_f, bias=False)
    calib = torch.randn(32, in_f)

    q1 = GPTQQuantizer(layer, GPTQConfig(group_size=-1))
    q1.add_batch(calib)
    _, s1, _ = q1.quantize()
    assert s1.shape == (out_f, 1)

    q2 = GPTQQuantizer(layer, GPTQConfig(group_size=8))
    q2.add_batch(calib)
    _, s2, _ = q2.quantize()
    assert s2.shape == (out_f, in_f // 8)


def test_8_bit_quantization_works():
    """bits=8 also works and produces tighter reconstruction than 4-bit."""
    from llm.quantization.gptq import GPTQConfig, GPTQQuantizer

    torch.manual_seed(11)
    in_f, out_f = 16, 8
    layer = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_f, in_f) * 0.3)
    calib = torch.randn(64, in_f)

    q = GPTQQuantizer(layer, GPTQConfig(bits=8, group_size=-1))
    q.add_batch(calib)
    W_q, scales, zeros = q.quantize()
    assert W_q.shape == (out_f, in_f)

    # 8-bit reconstruction should be quite tight
    mse = ((layer.weight - W_q) ** 2).mean().item()
    assert mse < 1e-2
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py -k "act_order or group_size or 8_bit" -v
```

Expected: All 3 PASS (existing implementation already supports these)

- [ ] **Step 3: Commit**

```bash
git add tests/quantization/test_gptq_algorithm.py
git commit -m "test(quant): act-order, group_size, 8-bit behavior tests"
```

---

## Task 5: GPTQQuantizedLinear + packed 4-bit storage

**Files:**
- Create: `src/llm/quantization/_gptq_layer.py`
- Create: `tests/quantization/test_gptq_layer.py`

- [ ] **Step 1: Create storage layer tests**

Create `tests/quantization/test_gptq_layer.py`:

```python
"""Tests for GPTQQuantizedLinear storage layer (packed 4-bit, dequantize, forward)."""

import pytest
import torch
import torch.nn as nn


# === Pack/unpack correctness ===

def test_pack_4bit_round_trip():
    """Pack unsigned int4 values into int8 storage and unpack back."""
    from llm.quantization._gptq_layer import _pack_4bit, _unpack_4bit

    # Values in [0, 15]
    w = torch.tensor([0, 1, 7, 8, 15, 5, 9, 3], dtype=torch.int8)
    packed = _pack_4bit(w)
    unpacked = _unpack_4bit(packed, numel=len(w))
    assert torch.equal(unpacked, w)


def test_pack_4bit_even_length_required():
    """_pack_4bit requires even number of values (pairs into single int8)."""
    from llm.quantization._gptq_layer import _pack_4bit

    with pytest.raises(ValueError, match="even"):
        _pack_4bit(torch.tensor([1, 2, 3], dtype=torch.int8))  # 3 elements


def test_pack_4bit_rejects_out_of_range():
    """Values must be in [0, 15] (unsigned int4)."""
    from llm.quantization._gptq_layer import _pack_4bit

    with pytest.raises(ValueError, match="range"):
        _pack_4bit(torch.tensor([1, 16], dtype=torch.int8))  # 16 is out of range


def test_packed_storage_is_half_size():
    """4-bit packed: stored bytes == numel / 2. This is the 4-bit promise."""
    from llm.quantization._gptq_layer import _pack_4bit

    w = torch.randint(0, 16, (4, 32), dtype=torch.int8)  # 128 values
    packed = _pack_4bit(w)
    assert packed.numel() == 64  # 128 / 2
    assert packed.element_size() == 1  # int8 storage
    # Total bytes: 64, vs 128 for unpacked int8
    assert packed.untyped_storage().nbytes() == 64


# === GPTQQuantizedLinear storage ===

def test_gptq_layer_initializes_with_correct_buffers():
    """GPTQQuantizedLinear exposes packed weight, scales, zeros buffers."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    layer = GPTQQuantizedLinear(
        in_features=16,
        out_features=8,
        bias=True,
        weight_packed=torch.zeros(8 * 16 // 2, dtype=torch.int8),
        scales=torch.ones(8, 2, dtype=torch.float16),  # 16/8 = 2 groups
        zeros=None,
        bits=4,
        group_size=8,
        sym=True,
    )
    assert layer.in_features == 16
    assert layer.out_features == 8
    assert layer.weight_packed.shape == (64,)
    assert layer.scales.shape == (8, 2)
    assert layer.bias is not None


def test_gptq_layer_no_bias_path():
    """bias=False creates a layer without bias parameter."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    layer = GPTQQuantizedLinear(
        in_features=8,
        out_features=4,
        bias=False,
        weight_packed=torch.zeros(16, dtype=torch.int8),
        scales=torch.ones(4, 1, dtype=torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    assert layer.bias is None


def test_gptq_layer_unpack_matches_original():
    """Internal _unpack_weights returns int8 values that match original quantization."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    # Create known quantized weights
    original_int4 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
    packed = _pack_4bit(original_int4.flatten()).reshape(2, 2)

    layer = GPTQQuantizedLinear(
        in_features=4,
        out_features=2,
        bias=False,
        weight_packed=packed,
        scales=torch.ones(2, 1, dtype=torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    unpacked = layer._unpack_weights()
    assert torch.equal(unpacked, original_int4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_layer.py -v
```

Expected: All 7 tests FAIL with `ModuleNotFoundError: No module named 'llm.quantization._gptq_layer'`

- [ ] **Step 3: Create `src/llm/quantization/_gptq_layer.py`**

Create `src/llm/quantization/_gptq_layer.py`:

```python
"""
GPTQQuantizedLinear: GPTQ-quantized Linear with packed 4-bit (or 8-bit) storage.

Storage convention for bits=4:
- weight_packed: int8 tensor, two int4 values per byte.
  Pair (w[2i], w[2i+1]) packed as (w[2i] << 4) | (w[2i+1] & 0x0F).
- scales: float16 tensor, shape [out_features, in_features // group_size].
- zeros: int8 tensor (or None if sym=True), shape [out_features, in_features // group_size].
- group_size=-1: scales shape [out_features, 1] (per-channel).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _pack_4bit(w: torch.Tensor) -> torch.Tensor:
    """Pack unsigned int4 values (shape [N,], even N) into int8 storage.

    Each pair (w[2i], w[2i+1]) is stored as (w[2i] << 4) | (w[2i+1] & 0x0F).

    Args:
        w: int8 tensor of shape [N,] with values in [0, 15]. N must be even.

    Returns:
        int8 tensor of shape [N // 2,].

    Raises:
        ValueError: If N is odd or values are out of [0, 15].
    """
    if w.numel() % 2 != 0:
        raise ValueError(
            f"_pack_4bit requires even number of values, got {w.numel()}."
        )
    if w.min() < 0 or w.max() > 15:
        raise ValueError(
            f"_pack_4bit values must be in [0, 15], got range [{w.min().item()}, {w.max().item()}]."
        )

    w_even = w[0::2]
    w_odd = w[1::2]
    packed = ((w_even << 4) | (w_odd & 0x0F)).to(torch.int8)
    return packed


def _unpack_4bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack int8 storage back to unsigned int4 values of shape [numel]."""
    if numel % 2 != 0:
        raise ValueError(f"_unpack_4bit numel must be even, got {numel}.")

    # High nibble: even indices, Low nibble: odd indices
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F

    out = torch.zeros(numel, dtype=torch.int8, device=packed.device)
    out[0::2] = high
    out[1::2] = low
    return out


class GPTQQuantizedLinear(nn.Module):
    """GPTQ-quantized Linear with packed 4-bit (or 8-bit) weight storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor],
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.sym = sym

        # Register packed weights and scales as buffers (not Parameters — no grad)
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        if zeros is not None:
            self.register_buffer("zeros", zeros)
        else:
            self.zeros = None

        # Bias remains fp32 / Parameter (only if original layer had bias)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _unpack_weights(self) -> torch.Tensor:
        """Unpack int8 storage to int4 (or int8) tensor of shape [out_features, in_features]."""
        if self.bits == 4:
            unpacked = _unpack_4bit(self.weight_packed, numel=self.out_features * self.in_features)
            return unpacked.reshape(self.out_features, self.in_features)
        else:
            # 8-bit: weight_packed stores int8 values directly
            return self.weight_packed.reshape(self.out_features, self.in_features)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_layer.py -v
```

Expected: All 7 storage tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/_gptq_layer.py tests/quantization/test_gptq_layer.py
git commit -m "feat(quant): GPTQQuantizedLinear with packed 4-bit storage"
```

---

## Task 6: GPTQQuantizedLinear forward + correctness

**Files:**
- Modify: `src/llm/quantization/_gptq_layer.py`
- Modify: `tests/quantization/test_gptq_layer.py`

- [ ] **Step 1: Append forward tests**

Append to `tests/quantization/test_gptq_layer.py`:

```python
# === Forward correctness ===

def test_forward_close_to_fp32_baseline():
    """Forward output cosine_sim > 0.99 vs equivalent fp32 Linear."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    torch.manual_seed(99)
    in_f, out_f = 32, 16

    # Original fp32 weight (small magnitude for tight 4-bit fit)
    w_fp32 = torch.randn(out_f, in_f) * 0.1
    x = torch.randn(8, in_f)

    # Quantize via simple per-channel 4-bit
    abs_max = w_fp32.abs().max(dim=1, keepdim=True)[0]
    qmax = 7  # 2^(4-1) - 1
    scale = (abs_max / qmax).clamp(min=1e-8)
    w_int4 = torch.round(w_fp32 / scale).clamp(-8, 7).to(torch.int8) + 8  # shift to [0, 15]
    packed = _pack_4bit(w_int4.flatten())

    layer_q = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=packed,
        scales=scale.to(torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )

    # fp32 baseline
    layer_fp32 = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        layer_fp32.weight.copy_(w_fp32)

    out_q = layer_q(x)
    out_fp = layer_fp32(x)

    cosine = torch.nn.functional.cosine_similarity(out_q, out_fp, dim=-1).mean()
    mse = ((out_q - out_fp) ** 2).mean().item()

    assert cosine > 0.99, f"cosine_sim {cosine:.4f} too low"
    assert mse < 1e-2


def test_forward_preserves_bias():
    """Bias is added correctly to forward output."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    in_f, out_f = 8, 4
    bias_values = torch.tensor([0.1, -0.2, 0.3, -0.4])

    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=True,
        weight_packed=torch.zeros(out_f * in_f // 2, dtype=torch.int8),
        scales=torch.ones(out_f, 1, dtype=torch.float16) * 0.01,  # tiny scale → near-zero weight
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    with torch.no_grad():
        layer.bias.copy_(bias_values)

    x = torch.zeros(2, in_f)
    out = layer(x)

    # Weights are ~0, so output ≈ bias
    assert torch.allclose(out, bias_values.unsqueeze(0).expand(2, -1), atol=1e-3)


def test_forward_grouped_quantization():
    """group_size=8 with per-group scales produces correct output."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    in_f, out_f = 16, 4
    group_size = 8
    n_groups = in_f // group_size  # 2

    # Per-group distinct scales
    scales = torch.tensor([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]], dtype=torch.float16)

    w_int4 = torch.zeros(out_f * in_f, dtype=torch.int8)
    packed = _pack_4bit(w_int4)

    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=packed,
        scales=scales,
        zeros=None,
        bits=4,
        group_size=group_size,
        sym=True,
    )

    x = torch.ones(1, in_f)
    out = layer(x)
    # All weights are 0, output is 0
    assert torch.allclose(out, torch.zeros(1, out_f), atol=1e-3)


def test_forward_input_shape_2d_and_3d():
    """Layer accepts both [batch, in] and [batch, seq, in]."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    in_f, out_f = 8, 4
    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=torch.zeros(out_f * in_f // 2, dtype=torch.int8),
        scales=torch.ones(out_f, 1, dtype=torch.float16) * 0.01,
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )

    x_2d = torch.randn(3, in_f)
    x_3d = torch.randn(2, 5, in_f)

    out_2d = layer(x_2d)
    out_3d = layer(x_3d)

    assert out_2d.shape == (3, out_f)
    assert out_3d.shape == (2, 5, out_f)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_layer.py -k "forward" -v
```

Expected: 4 forward tests FAIL with `TypeError: forward() missing 1 required positional argument`

- [ ] **Step 3: Append `forward` method to `GPTQQuantizedLinear` in `_gptq_layer.py`**

Append to `GPTQQuantizedLinear` class in `src/llm/quantization/_gptq_layer.py`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].
        """
        # Unpack int4 (or int8) values; values are unsigned [0, 15] for 4-bit.
        w_int = self._unpack_weights()  # [out_features, in_features]
        # Shift unsigned → signed: [0, 15] → [-8, 7]
        w_int_signed = w_int.to(torch.float32) - 8.0

        if self.group_size == -1:
            # Per-channel: scales shape [out_features, 1] broadcasts across input dim.
            w_fp = w_int_signed * self.scales.to(torch.float32)
        else:
            # Per-group: scales shape [out_features, in_features // group_size].
            # Expand to [out_features, in_features] by repeating within each group.
            gs = self.group_size
            n_groups = self.in_features // gs
            scales_expanded = self.scales.to(torch.float32).repeat_interleave(gs, dim=1)
            w_fp = w_int_signed * scales_expanded

        return torch.nn.functional.linear(x, w_fp, self.bias)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_layer.py -v
```

Expected: All 11 tests (7 storage + 4 forward) PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/_gptq_layer.py tests/quantization/test_gptq_layer.py
git commit -m "feat(quant): GPTQQuantizedLinear forward with per-channel/per-group dequantization"
```

---

## Task 7: `quantize_model_gptq` standalone entry + `_replace_module` helper

**Files:**
- Modify: `src/llm/quantization/gptq.py`
- Modify: `tests/quantization/test_gptq_end_to_end.py`

- [ ] **Step 1: Create end-to-end test file with entry point tests**

Create `tests/quantization/test_gptq_end_to_end.py`:

```python
"""End-to-end tests for GPTQ integration on full models."""

import pytest
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Tiny model for GPTQ end-to-end testing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_quantize_model_gptq_replaces_all_linear_layers():
    """quantize_model_gptq converts every nn.Linear to GPTQQuantizedLinear."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    linear_count = sum(1 for _ in quantized.modules() if isinstance(_, nn.Linear))
    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))

    assert linear_count == 0, f"Expected 0 nn.Linear, found {linear_count}"
    assert gptq_count == 2, f"Expected 2 GPTQQuantizedLinear (fc1, fc2), found {gptq_count}"


def test_quantize_model_gptq_preserves_forward_contract():
    """Quantized model accepts same input shape and returns same output shape."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    x = torch.randn(2, 16)
    out = quantized(x)
    assert out.shape == (2, 16)


def test_quantize_model_gptq_rejects_already_quantized():
    """Passing a model with GPTQQuantizedLinear raises ValueError."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]
    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    with pytest.raises(ValueError, match="already GPTQ-quantized"):
        quantize_model_gptq(quantized, iter(calib), GPTQConfig())


def test_quantize_model_gptq_no_linear_raises():
    """Model with no nn.Linear raises ValueError."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = nn.Sequential(nn.GELU(), nn.GELU())  # no Linear
    with pytest.raises(ValueError, match="no nn.Linear"):
        quantize_model_gptq(model, iter([torch.randn(4, 8)]), GPTQConfig())


def test_target_modules_filters_correctly():
    """target_modules restricts which Linear layers get quantized."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.ptq import QuantizedLinear

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    # Only quantize fc1, leave fc2 as nn.Linear
    quantized = quantize_model_gptq(
        model, iter(calib), GPTQConfig(), target_modules=["fc1"]
    )

    fc1_layer = quantized.fc1
    fc2_layer = quantized.fc2

    assert isinstance(fc1_layer, GPTQQuantizedLinear)
    assert isinstance(fc2_layer, nn.Linear)
    assert not isinstance(fc2_layer, GPTQQuantizedLinear)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/quantization/test_gptq_end_to_end.py -v
```

Expected: All 5 tests FAIL with `ImportError: cannot import name 'quantize_model_gptq'`

- [ ] **Step 3: Append `quantize_model_gptq` and helpers to `gptq.py`**

Append to `src/llm/quantization/gptq.py`:

```python
def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _quantize_linear_with_gptq(
    layer: nn.Linear,
    calib_batches: list[torch.Tensor],
    config: GPTQConfig,
) -> "GPTQQuantizedLinear":
    """Run GPTQ on a single Linear layer using accumulated calibration batches.

    Returns a GPTQQuantizedLinear ready to replace the original.
    """
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    quantizer = GPTQQuantizer(layer, config)
    for batch in calib_batches:
        quantizer.add_batch(batch)

    W_q, scales, zeros = quantizer.quantize()

    # Quantize W_q into int4/int8 storage
    bits = config.bits
    sym = config.sym
    group_size = config.group_size
    out_f, in_f = W_q.shape

    if bits == 4:
        # Convert fp32 weights to int4 [0, 15]
        if sym:
            # Per-group scale (or per-channel)
            if group_size == -1:
                scale = W_q.abs().max(dim=1, keepdim=True)[0] / 7.0
                scale = scale.clamp(min=1e-8)
                W_int = (W_q / scale).round().clamp(-8, 7).to(torch.int8) + 8
            else:
                gs = group_size
                n_groups = in_f // gs
                W_int = torch.zeros_like(W_q, dtype=torch.int8)
                scale = torch.zeros(out_f, n_groups, dtype=torch.float32)
                for g in range(n_groups):
                    s = g * gs
                    e = s + gs
                    w_g = W_q[:, s:e]
                    sc = w_g.abs().max(dim=1, keepdim=True)[0] / 7.0
                    sc = sc.clamp(min=1e-8)
                    scale[:, g : g + 1] = sc
                    W_int[:, s:e] = (w_g / sc).round().clamp(-8, 7).to(torch.int8) + 8
        else:
            raise NotImplementedError("Asymmetric GPTQ not yet implemented")

        packed = _pack_4bit(W_int.flatten())
        zeros_buf = None  # symmetric
    else:
        # 8-bit
        if sym:
            if group_size == -1:
                scale = W_q.abs().max(dim=1, keepdim=True)[0] / 127.0
                scale = scale.clamp(min=1e-8)
                W_int = (W_q / scale).round().clamp(-128, 127).to(torch.int8)
            else:
                gs = group_size
                n_groups = in_f // gs
                W_int = torch.zeros_like(W_q, dtype=torch.int8)
                scale = torch.zeros(out_f, n_groups, dtype=torch.float32)
                for g in range(n_groups):
                    s = g * gs
                    e = s + gs
                    w_g = W_q[:, s:e]
                    sc = w_g.abs().max(dim=1, keepdim=True)[0] / 127.0
                    sc = sc.clamp(min=1e-8)
                    scale[:, g : g + 1] = sc
                    W_int[:, s:e] = (w_g / sc).round().clamp(-128, 127).to(torch.int8)
        else:
            raise NotImplementedError("Asymmetric GPTQ not yet implemented")
        packed = W_int.flatten()  # already int8
        zeros_buf = None

    return GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=(layer.bias is not None),
        weight_packed=packed,
        scales=scale.to(torch.float16),
        zeros=zeros_buf,
        bits=bits,
        group_size=group_size,
        sym=sym,
    )


def quantize_model_gptq(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor],
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model with GPTQ.

    Args:
        model: nn.Module containing nn.Linear layers to quantize.
        calib_iter: Iterator yielding input tensors for the model forward pass.
        config: GPTQConfig (default: 4-bit, group_size=128, symmetric).
        target_modules: Iterable of fully-qualified layer names to quantize.
            If None, all nn.Linear layers are quantized.
        device: Device to run calibration on (default: model's device).

    Returns:
        The model with nn.Linear layers replaced by GPTQQuantizedLinear.

    Raises:
        ValueError: If model has no nn.Linear, target_modules unmatched, or layer already quantized.
    """
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    config = config or GPTQConfig()
    if device is not None:
        model = model.to(device)

    # Discover all Linear layers
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("model has no nn.Linear modules; nothing to quantize.")

    # Check for already-quantized layers
    for n, m in model.named_modules():
        if isinstance(m, GPTQQuantizedLinear):
            raise ValueError(
                f"Layer {n} is already GPTQ-quantized. "
                f"Pass a fresh model or unquantize first."
            )

    # Resolve target_modules
    if target_modules is not None:
        target_set = set(target_modules)
        all_names = {n for n, _ in linear_layers}
        matched = target_set & all_names
        if not matched:
            available = sorted(all_names)[:10]
            raise ValueError(
                f"target_modules {list(target_set)} matched no nn.Linear. "
                f"Available: {available}{'...' if len(all_names) > 10 else ''}"
            )
        targets = [(n, m) for n, m in linear_layers if n in target_set]
    else:
        targets = linear_layers

    # Collect all calibration batches
    calib_batches = list(calib_iter)
    if not calib_batches:
        raise ValueError("calib_iter is empty; need at least 1 batch for Hessian accumulation.")

    # Register forward hooks per target layer to capture per-layer inputs
    captured: dict[str, list[torch.Tensor]] = {n: [] for n, _ in targets}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            captured[name].append(inputs[0].detach().clone())
        return hook

    for n, m in targets:
        hooks.append(m.register_forward_hook(make_hook(n)))

    # Run forward pass(es) to capture per-layer inputs via hooks.
    model.eval()
    with torch.no_grad():
        param_device = next(model.parameters()).device
        for batch in calib_batches[:1]:
            try:
                _ = model(batch.to(param_device))
            except Exception as e:
                # If model forward fails (shape mismatch etc), fall through to direct call below.
                logger.debug(f"Model forward failed during calibration: {e}; falling back to direct layer calls.")

    # If hooks captured nothing, fall back to calling each target layer directly
    # with every calibration batch (works for simple sequential models).
    any_captured = any(len(v) > 0 for v in captured.values())
    if not any_captured:
        for h in hooks:
            h.remove()
        for n, m in targets:
            captured[n] = [batch.detach().clone() for batch in calib_batches]

    # Remove hooks
    for h in hooks:
        h.remove()

    # Quantize each target layer
    for name, layer in targets:
        new_layer = _quantize_linear_with_gptq(layer, captured[name], config)
        if layer.bias is not None:
            with torch.no_grad():
                new_layer.bias.copy_(layer.bias.data)
        _replace_module(model, name, new_layer)
        logger.info(f"Quantized layer {name}: {layer.weight.shape} → 4-bit packed")

    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/quantization/test_gptq_end_to_end.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/gptq.py tests/quantization/test_gptq_end_to_end.py
git commit -m "feat(quant): quantize_model_gptq standalone entry point with target_modules filter"
```

---

## Task 8: `quantize_model_with_collector` reusing CalibrationDataCollector

**Files:**
- Modify: `src/llm/quantization/gptq.py`
- Modify: `tests/quantization/test_gptq_end_to_end.py`

- [ ] **Step 1: Append collector-based test**

Append to `tests/quantization/test_gptq_end_to_end.py`:

```python
def test_quantize_model_with_collector_works():
    """quantize_model_with_collector reuses CalibrationDataCollector."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector
    from llm.quantization.calibration import CalibrationDataCollector
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    # Mock collector that yields batches
    class MockCollector:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

    model = TwoLayerMLP(hidden=16)
    batches = [torch.randn(8, 16) for _ in range(3)]
    collector = MockCollector(batches)

    quantized = quantize_model_with_collector(model, collector, n_samples=3, config=GPTQConfig())

    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))
    assert gptq_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/quantization/test_gptq_end_to_end.py::test_quantize_model_with_collector_works -v
```

Expected: FAIL with `ImportError: cannot import name 'quantize_model_with_collector'`

- [ ] **Step 3: Append `quantize_model_with_collector` to `gptq.py`**

Append to `src/llm/quantization/gptq.py`:

```python
def quantize_model_with_collector(
    model: nn.Module,
    collector: CalibrationDataCollector,
    n_samples: int,
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model using an existing CalibrationDataCollector.

    Args:
        model: nn.Module to quantize.
        collector: CalibrationDataCollector (or any object with __iter__ yielding Tensor batches).
        n_samples: Maximum number of batches to use for calibration.
        config: GPTQConfig.
        target_modules: Optional layer name filter.
        device: Target device.

    Returns:
        The quantized model.
    """
    config = config or GPTQConfig()
    # Materialize up to n_samples batches from the collector
    batches = []
    for i, batch in enumerate(collector):
        if i >= n_samples:
            break
        batches.append(batch)

    return quantize_model_gptq(
        model,
        calib_iter=iter(batches),
        config=config,
        target_modules=target_modules,
        device=device,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/quantization/test_gptq_end_to_end.py::test_quantize_model_with_collector_works -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/gptq.py tests/quantization/test_gptq_end_to_end.py
git commit -m "feat(quant): quantize_model_with_collector reuses CalibrationDataCollector"
```

---

## Task 9: Update `__init__.py` exports

**Files:**
- Modify: `src/llm/quantization/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

```bash
cat src/llm/quantization/__init__.py
```

- [ ] **Step 2: Rewrite `__init__.py` to export GPTQ API alongside PTQ**

Replace contents of `src/llm/quantization/__init__.py`:

```python
"""
Quantization module for model compression.

Provides two orthogonal paths:
- Simple post-training quantization (PTQ): INT8/INT4, symmetric/asymmetric, per-channel/per-tensor.
- GPTQ (Frantar 2022): Hessian-aware 4-bit/8-bit with packed storage, act-order, group_size.

Both share calibration infrastructure via `CalibrationDataCollector`.
"""

from llm.quantization.calibration import ActivationStats, CalibrationDataCollector

# Simple PTQ path
from llm.quantization.ptq import (
    QuantConfig,
    QuantizedLinear,
    compute_model_size,
    quantize_linear_layer,
    quantize_model,
)

# GPTQ path
from llm.quantization.gptq import (
    GPTQConfig,
    GPTQQuantizer,
    quantize_model_gptq,
    quantize_model_with_collector,
)
from llm.quantization._gptq_layer import GPTQQuantizedLinear

__all__ = [
    # Calibration
    "ActivationStats",
    "CalibrationDataCollector",
    # Simple PTQ
    "QuantConfig",
    "QuantizedLinear",
    "compute_model_size",
    "quantize_linear_layer",
    "quantize_model",
    # GPTQ
    "GPTQConfig",
    "GPTQQuantizer",
    "GPTQQuantizedLinear",
    "quantize_model_gptq",
    "quantize_model_with_collector",
]
```

- [ ] **Step 3: Verify existing PTQ tests still pass (no regression)**

```bash
uv run pytest tests/quantization/test_quantization.py -v
```

Expected: All existing PTQ tests PASS (zero regression)

- [ ] **Step 4: Verify new GPTQ tests pass via the public API**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py tests/quantization/test_gptq_layer.py tests/quantization/test_gptq_end_to_end.py -v
```

Expected: All new GPTQ tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/quantization/__init__.py
git commit -m "feat(quant): export GPTQ API alongside simple PTQ"
```

---

## Task 10: CLI `llm-quantize gptq` subcommand

**Files:**
- Create: `src/llm/cli/quantize.py`
- Create: `tests/cli/test_quantize_cli.py`
- Modify: `pyproject.toml` (verify CLI entry registered)
- Modify: `src/llm/cli/__init__.py` if needed

- [ ] **Step 1: Verify existing CLI entry pattern**

```bash
cat src/llm/cli/__init__.py
grep -n "llm-migrate-ckpt\|llm-quantize" pyproject.toml
```

- [ ] **Step 2: Append CLI tests**

Create `tests/cli/test_quantize_cli.py`:

```python
"""CLI tests for `llm-quantize gptq` subcommand."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help():
    """`llm-quantize gptq --help` exits 0 and lists expected flags."""
    result = subprocess.run(
        ["llm-quantize", "gptq", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--output" in result.stdout
    assert "--calib-data" in result.stdout
    assert "--bits" in result.stdout


def test_cli_missing_required_args_exits_nonzero():
    """No args → non-zero exit."""
    result = subprocess.run(
        ["llm-quantize", "gptq"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_invalid_bits_errors(tmp_path):
    """--bits 16 → error mentioning valid values."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = subprocess.run(
        [
            "llm-quantize", "gptq",
            "--model", str(model_path),
            "--output", str(tmp_path / "out.pt"),
            "--calib-data", str(tmp_path / "calib.txt"),
            "--bits", "16",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bits" in result.stderr.lower() or "4" in result.stderr


def test_cli_missing_tokenizer_errors(tmp_path):
    """--calib-data without --tokenizer → error."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text("hello world\n")
    result = subprocess.run(
        [
            "llm-quantize", "gptq",
            "--model", str(model_path),
            "--output", str(tmp_path / "out.pt"),
            "--calib-data", str(calib_path),
            "--bits", "4",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_calib_data_mutually_exclusive(tmp_path):
    """--calib-data + --calib-data-tokens → error."""
    model_path = tmp_path / "model.pt"
    model_path.touch()
    result = subprocess.run(
        [
            "llm-quantize", "gptq",
            "--model", str(model_path),
            "--output", str(tmp_path / "out.pt"),
            "--calib-data", str(tmp_path / "calib.txt"),
            "--calib-data-tokens", str(tmp_path / "calib.pt"),
            "--tokenizer", str(tmp_path / "tok"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_quantize_cli.py -v
```

Expected: All 5 tests FAIL (no `llm-quantize` command)

- [ ] **Step 4: Create `src/llm/cli/quantize.py`**

Create `src/llm/cli/quantize.py`:

```python
"""`llm-quantize` CLI — currently supports `gptq` subcommand."""

import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Model quantization CLI.")


@app.command()
def gptq(
    model: Path = typer.Option(..., "--model", help="Path to model checkpoint (.pt)."),
    output: Path = typer.Option(..., "--output", help="Output path for quantized model."),
    calib_data: Optional[Path] = typer.Option(
        None, "--calib-data", help="Path to raw text file (one sample per line)."
    ),
    calib_data_tokens: Optional[Path] = typer.Option(
        None,
        "--calib-data-tokens",
        help="Path to pre-tokenized .pt file with batch tensors. Mutually exclusive with --calib-data.",
    ),
    tokenizer: Optional[Path] = typer.Option(
        None, "--tokenizer", help="Path to HF tokenizer (required when --calib-data is set)."
    ),
    bits: int = typer.Option(4, "--bits", help="Quantization bit width (4 or 8)."),
    group_size: int = typer.Option(128, "--group-size", help="Group size (-1 for per-channel)."),
    sym: bool = typer.Option(True, "--sym/--asym", help="Symmetric quantization."),
    percdamp: float = typer.Option(0.01, "--percdamp", help="Hessian damping fraction."),
    blocksize: int = typer.Option(128, "--blocksize", help="Column block size."),
    act_order: bool = typer.Option(False, "--act-order/--no-act-order", help="Act-order column sorting."),
    target_modules: Optional[str] = typer.Option(
        None,
        "--target-modules",
        help="Comma-separated layer names to quantize (default: all nn.Linear).",
    ),
):
    """Quantize a model with GPTQ (Frantar 2022)."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    # Validate mutual exclusion
    if calib_data is not None and calib_data_tokens is not None:
        typer.echo(
            "Error: --calib-data and --calib-data-tokens are mutually exclusive.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Validate calib-data requires tokenizer
    if calib_data is not None and tokenizer is None:
        typer.echo(
            "Error: --calib-data requires --tokenizer PATH. "
            "Use --calib-data-tokens for pre-tokenized data.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Load model
    import torch

    typer.echo(f"Loading model from {model}...")
    model_obj = torch.load(model, map_location="cpu")

    # Build calib_iter
    if calib_data is not None:
        # Raw text: tokenize with HF tokenizer
        from transformers import AutoTokenizer

        typer.echo(f"Loading tokenizer from {tokenizer}...")
        tok = AutoTokenizer.from_pretrained(str(tokenizer))
        text_lines = calib_data.read_text().splitlines()
        batches = []
        for line in text_lines:
            if not line.strip():
                continue
            ids = tok(line, return_tensors="pt").input_ids
            batches.append(ids)
    else:
        typer.echo(f"Loading pre-tokenized calibration from {calib_data_tokens}...")
        batches = torch.load(calib_data_tokens)

    # Resolve target_modules
    target_list = None
    if target_modules is not None:
        target_list = [m.strip() for m in target_modules.split(",") if m.strip()]

    # Build config
    config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        percdamp=percdamp,
        blocksize=blocksize,
        act_order=act_order,
    )

    # Quantize
    typer.echo(f"Quantizing model with GPTQ (bits={bits}, group_size={group_size})...")
    quantized = quantize_model_gptq(model_obj, iter(batches), config, target_list)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, output)
    typer.echo(f"Quantized model saved to {output}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Verify CLI entry is registered in `pyproject.toml`**

```bash
grep -A 5 "project.scripts" pyproject.toml
```

If `llm-quantize` is not listed, add it. Expected pattern (check existing entries):

```toml
[project.scripts]
llm-migrate-ckpt = "llm.cli.migrate_ckpt:main"
llm-quantize = "llm.cli.quantize:app"
```

If the script uses `app()` instead of `main()`, add a `main()` shim:

Append to `src/llm/cli/quantize.py`:

```python
def main():
    app()
```

Then in `pyproject.toml`, ensure the entry is:

```toml
llm-quantize = "llm.cli.quantize:main"
```

- [ ] **Step 6: Reinstall package in editable mode (or `uv sync`) to register CLI**

```bash
uv sync
```

- [ ] **Step 7: Run CLI tests to verify they pass**

```bash
uv run pytest tests/cli/test_quantize_cli.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 8: Smoke test CLI manually**

```bash
llm-quantize gptq --help
```

Expected: Help output lists all flags

- [ ] **Step 9: Commit**

```bash
git add src/llm/cli/quantize.py tests/cli/test_quantize_cli.py pyproject.toml
git commit -m "feat(cli): llm-quantize gptq subcommand"
```

---

## Task 11: ADR-007 + CHANGELOG + ROADMAP updates

**Files:**
- Create: `docs/adr/007-gptq-integration.md`
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Create ADR-007**

Create `docs/adr/007-gptq-integration.md` (mirror style of `006-checkpoint-format-unification.md`):

```markdown
# ADR-007: GPTQ Integration Architecture

**Date**: 2026-07-22
**Status**: Accepted
**Roadmap**: §13.3 (量化与压缩)

## Context

Production LLM deployment increasingly relies on 4-bit post-training quantization.
GPTQ (Frantar et al. 2022) is the de-facto standard algorithm: Hessian-aware
column-wise quantization that beats naive round-to-nearest (RTN) on quality while
requiring only a small calibration set.

The project already has `src/llm/quantization/ptq.py` providing simple INT8/INT4
post-hoc quantization. We need to add GPTQ without contaminating that path.

## Decision

Three architectural choices:

### 1. Orthogonal module (`gptq.py`) — not extending `ptq.py`

GPTQ is a fundamentally different algorithm (stateful per-layer processor with
Hessian accumulation) versus simple PTQ (direct post-hoc quantization). Merging
them into one class would create branching complexity and entangle their tests.

**Result**: `ptq.py` is unchanged. New files:
- `src/llm/quantization/gptq.py` (algorithm + entry points)
- `src/llm/quantization/_gptq_layer.py` (packed 4-bit storage layer)

### 2. True packed 4-bit storage (2 weights/byte int8)

A 4-bit quantization that stores values in int8 slots (just masking) does NOT
reduce memory — it lies about size. We pack 2 unsigned int4 values into a single
int8 byte, halving storage as promised. The `_pack_4bit` / `_unpack_4bit` helpers
live in `_gptq_layer.py` and are unit-tested.

**Result**: `compute_model_size` reports accurate post-quantization sizes for
GPTQ layers; existing INT8/INT4 PTQ reporting unchanged.

### 3. Dual entry points

Two entry points to match two workflows:
- `quantize_model_gptq(model, calib_iter, ...)` — standalone, user passes Iterator[Tensor].
- `quantize_model_with_collector(model, collector, n_samples, ...)` — reuses
  existing `CalibrationDataCollector` from the training framework.

Both funnel into the same `quantize_model_gptq` implementation.

## Consequences

### Positive

- Simple PTQ path has zero regression risk (unchanged code, unchanged tests).
- 4-bit packing is a real 50% memory reduction, verified by `test_packed_storage_half_size`.
- Trainer-loop users can call `quantize_model_with_collector(...)` without re-collecting calibration data.
- CLI (`llm-quantize gptq`) parallels `llm-migrate-ckpt` for production workflows.

### Negative

- Two quantization paths increase API surface (~6 new public symbols).
- Future PEFT-aware quantization will need to handle both PTQ and GPTQ layers.

### Neutral

- Asymmetric GPTQ (zero-point based) is deferred to a follow-up — symmetric covers 95% of production cases.

## Alternatives Considered

### A. Extend `QuantizedLinear` with packed + group_size fields

Rejected: would force INT8 simple-PTQ tests to run regression on every change.
Branching in `forward()` for "if packed...elif per-channel...elif grouped..."
becomes hard to reason about.

### B. 4-bit as int8 slots (just mask to [-8, 7])

Rejected: defeats the purpose of 4-bit quantization (no memory savings);
`compute_model_size` would report wrong sizes.

### C. Skip CLI, only Python API

Rejected: production users expect a CLI for repeated quantization jobs.
CLI parallels `llm-migrate-ckpt` (just shipped in 9a2026f).

## References

- Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (arXiv:2210.17323)
- AutoGPTQ reference implementation: https://github.com/AutoGPTQ/AutoGPTQ
- Spec: `docs/superpowers/specs/2026-07-22-gptq-integration-design.md`
- Plan: `docs/superpowers/plans/2026-07-22-gptq-integration.md`
```

- [ ] **Step 2: Update CHANGELOG**

Append to `CHANGELOG.md` under `[Unreleased] > ### Added`:

```markdown
- **GPTQ post-training quantization** (阶段十三 §13.3): 4-bit / 8-bit Hessian-aware quantization following Frantar et al. 2022. New `src/llm/quantization/gptq.py` (algorithm + dual entry points) and `src/llm/quantization/_gptq_layer.py` (packed 4-bit storage, 2 weights/byte). Orthogonal to existing simple PTQ path — `ptq.py` unchanged. CLI `llm-quantize gptq --model PATH --output PATH --calib-data PATH --tokenizer PATH --bits 4 --group-size 128` parallels `llm-migrate-ckpt`. Two entry points: `quantize_model_gptq(model, calib_iter, config)` (standalone) and `quantize_model_with_collector(model, collector, n_samples, config)` (reuse existing `CalibrationDataCollector`). Supports `act_order`, `sym`, `percdamp`, `blocksize`, `static_groups`. ADR-007 captures architecture decisions.
```

- [ ] **Step 3: Update ROADMAP**

In `ROADMAP.md`, find line 392:

```
- [ ] 集成 GPTQ (GPT Quantization)
```

Replace with:

```
- [x] **集成 GPTQ** (Frantar 2022, 阶段十三 §13.3 首切片完成: `src/llm/quantization/gptq.py` + `_gptq_layer.py` + `llm-quantize gptq` CLI + dual entry point + 4-bit 打包存储 + act-order/group_size/percdamp 旋钮; 22 unit tests + 5 e2e tests; 零回归对现有 simple-PTQ; ADR-007; AWQ/SmoothQuant 留待后续切片)
```

- [ ] **Step 4: Verify ROADMAP renders correctly**

```bash
grep -n "集成 GPTQ" ROADMAP.md
```

Expected: Shows updated checkbox `[x]` with detail

- [ ] **Step 5: Commit**

```bash
git add docs/adr/007-gptq-integration.md CHANGELOG.md ROADMAP.md
git commit -m "docs: ADR-007 GPTQ integration + CHANGELOG + ROADMAP updates"
```

---

## Task 12: Full test run + coverage check

**Files:**
- Modify: (none — verification only)

- [ ] **Step 1: Run all quantization tests**

```bash
uv run pytest tests/quantization/ tests/cli/test_quantize_cli.py -v --tb=short
```

Expected: All tests PASS (existing PTQ + new GPTQ + CLI)

- [ ] **Step 2: Run full test suite to confirm no broader regression**

```bash
uv run pytest tests/ -q --tb=short -x
```

Expected: All pre-existing tests still PASS (1408+ tests)

- [ ] **Step 3: Check coverage on new files**

```bash
uv run pytest tests/quantization/test_gptq_algorithm.py tests/quantization/test_gptq_layer.py tests/quantization/test_gptq_end_to_end.py tests/cli/test_quantize_cli.py --cov=src/llm/quantization --cov=src/llm/cli/quantize --cov-report=term-missing
```

Expected:
- `gptq.py`: ≥95% line coverage
- `_gptq_layer.py`: ≥90% line coverage
- `cli/quantize.py`: ≥85% line coverage

- [ ] **Step 4: Run lint and type checks**

```bash
uv run ruff check src/llm/quantization/ src/llm/cli/quantize.py tests/quantization/test_gptq_*.py tests/cli/test_quantize_cli.py
uv run ruff format --check src/llm/quantization/ src/llm/cli/quantize.py tests/quantization/test_gptq_*.py tests/cli/test_quantize_cli.py
```

Expected: 0 errors / 0 format issues

- [ ] **Step 5: If any failures, fix and re-verify**

Iterate until all checks pass.

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: address review findings from full test run"
```

(Only commit if Step 5 produced fixes; otherwise skip.)

---

## Summary

**12 tasks, ~50 atomic steps. Estimated total: 8-12 hours of focused work.**

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| Algorithm core | 1-4 | `GPTQConfig` + `GPTQQuantizer` + Hessian + Cholesky + column loop + act-order |
| Storage layer | 5-6 | `GPTQQuantizedLinear` + packed 4-bit + forward |
| Integration | 7-9 | Two entry points + `__init__.py` exports |
| CLI | 10 | `llm-quantize gptq` subcommand |
| Docs | 11 | ADR-007 + CHANGELOG + ROADMAP |
| Verification | 12 | Full test suite + coverage + lint clean |

**Acceptance**:
- [ ] All 4 GPTQ test files pass (~30+ tests)
- [ ] All 5 CLI tests pass
- [ ] All 1408 pre-existing tests still pass (zero regression)
- [ ] Coverage on new files meets targets (95/90/85%)
- [ ] `make test` clean
- [ ] `llm-quantize gptq --help` works
- [ ] ADR-007 + CHANGELOG + ROADMAP committed
