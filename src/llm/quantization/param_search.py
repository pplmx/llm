"""Quantization parameter search (ROADMAP 13.2 / QAT research).

Fake/static quantization has knobs — bit width (4/8) and scale granularity
(per-tensor vs per-channel) — and the "right" choice depends on the tensor's
weight/activation distribution. The repo's calibration module
(``calibration.py``) returns absmax-based scales but does not *search* the
configuration space by how well a candidate actually reconstructs the tensor.

This slice answers: **which (bits, granularity) quantizes a given tensor with
the least reconstruction error?** It fake-round-trips the (detached) tensor
under every candidate using the real :class:`FakeQuantize` implementation and
scores each by normalized MSE ``||x - dequant(x)||^2 / ||x||^2``, returning the
best configuration. This is the parameter-search ingredient of QAT / static
rounding, and is CPU-verifiable.

Intended invariants (verified in ``tests/quantization/test_param_search.py``):

- more bits (8 vs 4) never increases reconstruction error;
- per-channel granularity never increases error vs per-tensor (finer scale);
- the reported ``best`` is the argmin over all candidates.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from llm.quantization.fake_quant import FakeQuantize

__all__ = ["param_candidates", "reconstruction_errors", "search_quant_params"]

#: (bits, granularity) candidate space; granularity ``per_channel`` quantizes
#: along row dim 0 (out features), matching the weight-layout convention.
DEFAULT_BITS = (4, 8)
DEFAULT_GRANULARITIES = ("per_tensor", "per_channel")


def param_candidates(
    bits: Iterable[int] = DEFAULT_BITS,
    granularities: Iterable[str] = DEFAULT_GRANULARITIES,
) -> list[tuple[int, str]]:
    """All ``(bits, granularity)`` candidates."""
    return [(b, g) for b in bits for g in granularities]


def _dequantize(x: Tensor, bits: int, per_channel: bool) -> Tensor:
    """Fake round-trip ``x`` through :class:`FakeQuantize` (no grad)."""
    fq = FakeQuantize(bits, per_channel=per_channel, channel_dim=0)
    return fq(x.detach())


def _normalized_mse(x: Tensor, deq: Tensor) -> float:
    denom = float(x.pow(2).mean().clamp_min(1e-12))
    return float((x - deq).pow(2).mean()) / denom


def reconstruction_errors(
    x: Tensor,
    *,
    bits: Iterable[int] = DEFAULT_BITS,
    granularities: Iterable[str] = DEFAULT_GRANULARITIES,
) -> dict[tuple[int, str], float]:
    """Normalized reconstruction MSE for every ``(bits, granularity)`` candidate."""
    return {
        (b, g): _normalized_mse(x, _dequantize(x, b, per_channel=(g == "per_channel")))
        for b in bits
        for g in granularities
    }


def search_quant_params(
    x: Tensor,
    *,
    bits: Iterable[int] = DEFAULT_BITS,
    granularities: Iterable[str] = DEFAULT_GRANULARITIES,
) -> tuple[tuple[int, str], float, dict[tuple[int, str], float]]:
    """Return the ``(best_config, best_mse, all_errors)`` for ``x``.

    ``best_config`` is the ``(bits, granularity)`` with the smallest
    reconstruction error.
    """
    errors = reconstruction_errors(x, bits=bits, granularities=granularities)
    best = min(errors, key=errors.__getitem__)
    return best, errors[best], errors
