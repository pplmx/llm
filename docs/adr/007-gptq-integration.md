# 007. GPTQ integration architecture

Date: 2026-07-22

## Status

Accepted

## Context

Production LLM deployment increasingly relies on 4-bit post-training quantization.
GPTQ (Frantar et al. 2022, arXiv:2210.17323) is the de-facto standard algorithm:
Hessian-aware column-wise quantization that beats naive round-to-nearest (RTN)
on quality while requiring only a small calibration set.

The project already has `src/llm/quantization/ptq.py` providing simple INT8/INT4
post-hoc quantization (`QuantConfig`, `QuantizedLinear`, `quantize_model`). The
PTQ path is direct — quantize the trained weight, no calibration needed. It
works but has two limits relevant here:

1. **No Hessian awareness.** PTQ rounds each weight independently of the
   layer's input distribution, so error accumulates column-by-column on
   calibration-relevant features. For 4-bit the gap vs GPTQ on quality is
   typically 1–3 perplexity points on language models.
2. **No packed storage.** `QuantizedLinear` stores INT8 values; for a "4-bit"
   config it just masks two INT8 slots, which is dishonest about memory
   (`compute_model_size` would still report INT8-byte counts).

We need to add GPTQ without disturbing the PTQ path: separate concerns, separate
tests, separate storage.

## Decision

Three architectural choices for the GPTQ slice:

### 1. Orthogonal module (`gptq.py`) — not extending `ptq.py`

GPTQ is a fundamentally different algorithm: a stateful per-layer processor
that accumulates a Hessian matrix and re-quantizes columns sequentially with
error correction. The simple-PTQ path is stateless post-hoc rounding. Merging
them would create branching complexity in `forward()` and entangle their
tests, so any change to GPTQ could regress PTQ users and vice versa.

**Result**: `ptq.py` is unchanged. New files:

- `src/llm/quantization/gptq.py` — `GPTQConfig`, `GPTQQuantizer` (algorithm
  core: Hessian accumulation, Cholesky inverse, column loop, act-order,
  group_size, dead-column handling), and the two top-level entry points
  (`quantize_model_gptq`, `quantize_model_with_collector`).
- `src/llm/quantization/_gptq_layer.py` — `GPTQQuantizedLinear` (packed 4-bit
  storage layer with dequantize-on-forward) and the `_pack_4bit` /
  `_unpack_4bit` helpers.

### 2. True packed 4-bit storage (2 weights/byte int8)

A 4-bit quantization that stores values in int8 slots (just masking) does NOT
reduce memory — it lies about size. GPTQ packs 2 unsigned int4 values into a
single int8 byte, halving storage as promised. The packing convention is
`(w[2i] << 4) | (w[2i+1] & 0x0F)` and is unit-tested for round-trip and
half-size invariant (`test_packed_storage_is_half_size`).

**Result**: `compute_model_size` reports accurate post-quantization sizes for
GPTQ layers; existing INT8/INT4 PTQ reporting unchanged.

### 3. Dual entry points

Two entry points match two workflows:

- `quantize_model_gptq(model, calib_iter, ...)` — standalone, user passes an
  `Iterator[Tensor]` of calibration batches and drives the loop themselves.
- `quantize_model_with_collector(model, collector, n_samples, ...)` — reuses
  the existing `CalibrationDataCollector` from the training framework
  (or any iterable yielding batches), capping consumption at `n_samples`.

Both funnel into the same `quantize_model_gptq` implementation; the
collector entry is a thin materializer. This lets trainer-loop users call
quantization at the end of training without re-running data collection, and
it lets standalone scripts (CI, batch jobs, the CLI) keep full control.

## Consequences

### Positive

- **Zero regression risk for PTQ.** Unchanged code, unchanged tests, unchanged
  public API.
- **Real 50% memory reduction at 4-bit** — verified by
  `test_packed_storage_is_half_size` (`packed.numel() == in_f * out_f / 2`,
  int8 storage).
- **Trainer-loop users get free reuse** of their existing
  `CalibrationDataCollector` via `quantize_model_with_collector`.
- **Public API surface matches spec** — exported via `src/llm/quantization/__init__.py`
  alongside the PTQ symbols, so `from llm.quantization import GPTQConfig` works.

### Negative

- **Two quantization paths increase API surface** (~5 new public symbols:
  `GPTQConfig`, `GPTQQuantizer`, `GPTQQuantizedLinear`, `quantize_model_gptq`,
  `quantize_model_with_collector`). Documented in `__init__.py` and the spec.
- **Future PEFT-aware quantization will need to handle both PTQ and GPTQ
  layers.** The PEFT slice is already a separate module (`peft.py`); PEFT-aware
  quantization is deferred to a follow-up slice.
- **Asymmetric GPTQ (zero-point based) is unimplemented** in this slice —
  symmetric covers ~95% of production cases and is the Frantar 2022 default.
  `__post_init__` and `_quantize_linear_with_gptq` both raise
  `NotImplementedError` for `sym=False`.

### Neutral

- **Compute dtype is fp32** for Hessian / Cholesky, regardless of the model's
  storage dtype. This matches Frantar 2022 and keeps the algorithm numerically
  stable; users targeting bf16/fp16 storage pay one fp32 promotion cost on the
  calibration pass only.
- **Activation capture uses `register_forward_hook`**, not pre-hooks, because
  we want the layer's actual input (post-reshape) for `X^T X` accumulation.

## Alternatives Considered

### A. Extend `QuantizedLinear` with packed + group_size fields

Rejected. Would force the existing INT8 simple-PTQ test suite to run regression
on every change to GPTQ packing. Branching in `forward()` for
"if packed...elif per-channel...elif grouped..." becomes hard to reason about
and makes the storage layer carry two unrelated concerns.

### B. 4-bit as int8 slots (just mask to [-8, 7])

Rejected. Defeats the purpose of 4-bit quantization (no memory savings);
`compute_model_size` would report wrong sizes and we'd be shipping a
misleading "4-bit" layer.

### C. Single entry point with optional collector

Considered. Would have meant `quantize_model(model, calib_or_collector=...)`
with type-dispatching internals. Rejected because the two entry points have
different ergonomics: standalone users want `Iterator[Tensor]`, trainer users
want to cap batch consumption. A single polymorphic entry hides both
intents and makes testing harder.

### D. Skip CLI, only Python API

Considered for this slice. The Python API covers all programmatic use cases;
the CLI is a thin wrapper. Deferred to a follow-up task — high ROI for
production users but low ROI for unit-testing the algorithm itself.

## References

- Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers" (arXiv:2210.17323).
- AutoGPTQ reference implementation: <https://github.com/AutoGPTQ/AutoGPTQ>.
- Spec: `docs/superpowers/specs/2026-07-22-gptq-integration-design.md`.
- Plan: `docs/superpowers/plans/2026-07-22-gptq-integration.md`.
- Sister ADR on orthogonal modules: see ADR-005 (export-registry) for the
  precedent of "orthogonal paths share calibration infrastructure rather than
  merging implementations."
