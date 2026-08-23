"""Tests for the quantization parameter-search slice (ROADMAP 13.2)."""

from __future__ import annotations

import torch

from llm.quantization.param_search import param_candidates, reconstruction_errors, search_quant_params


def _heterogeneous_weight():
    torch.manual_seed(0)
    rows = torch.randn(4, 16).abs() * torch.tensor([0.1, 1.0, 10.0, 3.0])[:, None]
    return rows


def test_more_bits_never_increase_error():
    w = _heterogeneous_weight()
    errs = reconstruction_errors(w)
    for granularity in ("per_tensor", "per_channel"):
        assert errs[(8, granularity)] <= errs[(4, granularity)] + 1e-9


def test_per_channel_never_increase_error():
    w = _heterogeneous_weight()
    errs = reconstruction_errors(w)
    for bits in (4, 8):
        assert errs[(bits, "per_channel")] <= errs[(bits, "per_tensor")] + 1e-9


def test_best_config_is_argmin():
    w = _heterogeneous_weight()
    best, best_err, errs = search_quant_params(w)
    assert best in errs
    assert best == min(errs, key=errs.get)
    assert errs[best] == best_err


def test_candidate_space():
    cands = param_candidates()
    assert set(cands) == {(4, "per_tensor"), (4, "per_channel"), (8, "per_tensor"), (8, "per_channel")}


def test_search_matches_manual_fakequant_reference():
    """The reported error equals a fresh FakeQuantize round-trip via numpy."""
    from llm.quantization.fake_quant import FakeQuantize

    w = _heterogeneous_weight()
    _, best_err, _ = search_quant_params(w, bits=(8,), granularities=("per_channel",))
    fq = FakeQuantize(8, per_channel=True, channel_dim=0)
    deq = fq(w.detach())
    denom = float(w.pow(2).mean().clamp_min(1e-12))
    manual = float((w - deq).pow(2).mean()) / denom
    assert torch.allclose(torch.tensor(best_err), torch.tensor(manual), atol=1e-6)
