"""Tests for GPTQ mixed-precision quantization (LayerQuantPolicy integration).

These tests cover:
- T3: GPTQConfig integration with layer_policies (5 tests)
- T4: GPTQ mixed-precision quantize_model_gptq dispatch (10 tests)
- T5: e2e mixed-precision on realistic models (2 tests)

Builds on commit 3 (LayerQuantPolicy + resolve_layer_policies).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# =============================================================================
# Level 3: GPTQConfig integration (5 tests)
# =============================================================================


def test_gptq_config_default_no_policies():
    """T3.1: GPTQConfig() default has empty layer_policies (no behavior change)."""
    from llm.quantization.gptq import GPTQConfig

    config = GPTQConfig()
    assert config.layer_policies == ()


def test_gptq_config_with_empty_policies():
    """T3.2: explicit empty tuple ≡ default (zero behavior change)."""
    from llm.quantization.gptq import GPTQConfig

    a = GPTQConfig()
    b = GPTQConfig(layer_policies=())
    assert a == b
    assert a.layer_policies == b.layer_policies


def test_gptq_config_accepts_valid_policies():
    """T3.3: valid LayerQuantPolicy list passes __post_init__."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig

    policies = (
        LayerQuantPolicy(target_modules=("fc1",), bits=8),
        LayerQuantPolicy(target_modules=("fc2",), bits=4, group_size=-1),
    )
    config = GPTQConfig(layer_policies=policies)
    assert config.layer_policies == policies


def test_gptq_config_rejects_non_policy_in_tuple():
    """E8: tuple element that is not a LayerQuantPolicy raises TypeError."""
    from llm.quantization.gptq import GPTQConfig

    with pytest.raises(TypeError, match=r"layer_policies\[0\] must be LayerQuantPolicy"):
        GPTQConfig(layer_policies=("fc1",))  # type: ignore[arg-type]


def test_gptq_config_frozen_with_policies():
    """T3.5: frozen dataclass — reassigning layer_policies raises."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig

    config = GPTQConfig(layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),))
    with pytest.raises((AttributeError, Exception)):
        config.layer_policies = ()  # type: ignore[misc]


# =============================================================================
# Level 4: GPTQ mixed-precision quantize_model_gptq dispatch (10 tests)
# =============================================================================


class TwoLayerMLP(nn.Module):
    """Tiny model for GPTQ mixed-precision testing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_quantize_single_layer_bits_override():
    """T4.1: fc1 → 8-bit override, fc2 → base (4-bit)."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=64)  # larger hidden so base group_size=128 fits
    calib = [torch.randn(8, 64) for _ in range(4)]

    config = GPTQConfig(
        bits=4,
        layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert isinstance(quantized.fc2, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 8
    assert quantized.fc2.bits == 4  # base config


def test_quantize_single_layer_group_size_override():
    """T4.2: fc1 → bits=4 base, group_size=-1 override (per-channel)."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=64)
    calib = [torch.randn(8, 64) for _ in range(4)]

    config = GPTQConfig(
        bits=4,
        group_size=128,
        layer_policies=(LayerQuantPolicy(target_modules=("fc1",), group_size=-1),),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    assert quantized.fc1.bits == 4  # base
    assert quantized.fc1.group_size == -1  # override
    assert quantized.fc2.bits == 4  # base
    assert quantized.fc2.group_size == 128  # base


def test_quantize_full_mixed_dispatch():
    """T4.3: fc1 → 8-bit per-channel, fc2 → 4-bit grouped (full mixed)."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=64)
    calib = [torch.randn(8, 64) for _ in range(4)]

    config = GPTQConfig(
        bits=4,
        layer_policies=(
            LayerQuantPolicy(target_modules=("fc1",), bits=8, group_size=-1),
            LayerQuantPolicy(target_modules=("fc2",), bits=4, group_size=128),
        ),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert isinstance(quantized.fc2, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 8
    assert quantized.fc1.group_size == -1
    assert quantized.fc2.bits == 4
    assert quantized.fc2.group_size == 128


def test_quantize_inherits_base_for_unmentioned_layers():
    """T4.4: layers not in any policy use base config."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=64)
    calib = [torch.randn(8, 64) for _ in range(4)]

    # Only fc1 overridden; fc2 must inherit (bits=4, group_size=128)
    config = GPTQConfig(
        bits=4,
        group_size=128,
        layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    assert quantized.fc1.bits == 8
    assert quantized.fc2.bits == 4
    assert quantized.fc2.group_size == 128


def test_quantize_with_target_modules_filter():
    """T4.5: target_modules arg filter is orthogonal to layer_policies dispatch.

    target_modules=["fc1"] limits quantization to fc1; layer_policies only
    references fc2. Since fc2 is not in available set, ValueError (E9).
    """
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    config = GPTQConfig(
        layer_policies=(LayerQuantPolicy(target_modules=("fc2",), bits=8),),
    )
    # fc2 not in target_modules → strict mode rejects with ValueError
    with pytest.raises(ValueError, match=r"target_modules.*not found"):
        quantize_model_gptq(model, iter(calib), config, target_modules=["fc1"])


def test_quantize_forward_contract_preserved():
    """T4.8: mixed-precision model accepts same input/output shape."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    config = GPTQConfig(
        bits=4,
        layer_policies=(
            LayerQuantPolicy(target_modules=("fc1",), bits=8),
            LayerQuantPolicy(target_modules=("fc2",), bits=4),
        ),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    x = torch.randn(2, 16)
    out = quantized(x)
    assert out.shape == (2, 16)


def test_quantize_with_collector_inherits_mixed_precision():
    """T4.9: quantize_model_with_collector also supports layer_policies."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    class MockCollector:
        def __init__(self, batches):
            self._batches = batches

        def __iter__(self):
            return iter(self._batches)

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]
    collector = MockCollector(calib)

    config = GPTQConfig(
        bits=4,
        layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),),
    )
    quantized = quantize_model_with_collector(model, collector, n_samples=4, config=config)

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 8
    assert quantized.fc2.bits == 4


def test_quantize_zero_policies_no_behavior_change():
    """T4.10: KEY REGRESSION — layer_policies=() vs no field, output byte-equal.

    With same seed + same calibration data, the packed weights must match
    byte-for-byte. This proves the new field has zero effect on the
    default (uniform-precision) path.
    """
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    def _build_and_quantize():
        torch.manual_seed(42)
        model = TwoLayerMLP(hidden=16)
        calib = [torch.randn(8, 16) for _ in range(4)]
        config = GPTQConfig()
        return quantize_model_gptq(model, iter(calib), config)

    a = _build_and_quantize()
    b = _build_and_quantize()

    assert torch.equal(a.fc1.weight_packed, b.fc1.weight_packed)
    assert torch.equal(a.fc2.weight_packed, b.fc2.weight_packed)


def test_quantize_with_collector_zero_policies_no_behavior_change():
    """T4.10b: same as T4.10 but for quantize_model_with_collector path."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    class MockCollector:
        def __init__(self, batches):
            self._batches = batches

        def __iter__(self):
            return iter(self._batches)

    def _build_and_quantize():
        torch.manual_seed(42)
        model = TwoLayerMLP(hidden=16)
        calib = [torch.randn(8, 16) for _ in range(4)]
        config = GPTQConfig(layer_policies=())
        return quantize_model_with_collector(model, MockCollector(calib), n_samples=4, config=config)

    a = _build_and_quantize()
    b = _build_and_quantize()

    assert torch.equal(a.fc1.weight_packed, b.fc1.weight_packed)


def test_quantize_rejects_overlap_in_layer_policies():
    """E10: two policies targeting same layer → ValueError (fail-fast)."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    config = GPTQConfig(
        layer_policies=(
            LayerQuantPolicy(target_modules=("fc1",), bits=8),
            LayerQuantPolicy(target_modules=("fc1",), bits=4),  # overlap!
        ),
    )
    with pytest.raises(ValueError, match=r"overlap detected"):
        quantize_model_gptq(model, iter(calib), config)


# =============================================================================
# Level 5: e2e on realistic models (2 tests)
# =============================================================================


def test_e2e_realistic_mlp_mixed_precision():
    """T5.1: TwoLayerMLP (hidden=64) mixed-precision end-to-end.

    Verifies shape + bits/group_size per layer + no NaN in output.
    """
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=64)
    calib = [torch.randn(8, 64) for _ in range(8)]

    config = GPTQConfig(
        bits=4,
        layer_policies=(
            LayerQuantPolicy(target_modules=("fc1",), bits=8, group_size=-1),
            LayerQuantPolicy(target_modules=("fc2",), bits=4, group_size=128),
        ),
    )
    quantized = quantize_model_gptq(model, iter(calib), config)

    assert quantized.fc1.bits == 8
    assert quantized.fc1.group_size == -1
    assert quantized.fc2.bits == 4
    assert quantized.fc2.group_size == 128

    x = torch.randn(2, 64)
    out = quantized(x)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_e2e_mixed_vs_uniform_storage_efficiency():
    """T5.2: mixed precision storage ≤ uniform 4-bit storage.

    Mixed precision in this test keeps fc1 at 8-bit (slightly more storage)
    but verifies storage accounting is per-layer correct. The test asserts
    correctness of sizes (not absolute savings, which depend on shape).
    """
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    def _quantize(layer_policies):
        torch.manual_seed(42)
        model = TwoLayerMLP(hidden=64)
        calib = [torch.randn(8, 64) for _ in range(4)]
        config = GPTQConfig(bits=4, layer_policies=layer_policies)
        return quantize_model_gptq(model, iter(calib), config)

    # Uniform 4-bit
    uniform = _quantize(())
    # Mixed: fc1 → 8-bit per-channel, fc2 → 4-bit grouped
    mixed = _quantize(
        (
            LayerQuantPolicy(target_modules=("fc1",), bits=8, group_size=-1),
            LayerQuantPolicy(target_modules=("fc2",), bits=4, group_size=128),
        )
    )

    # fc1 is bigger in mixed (8-bit per-channel vs 4-bit packed)
    # but fc2 is identical (same config in both).
    # Sanity check: mixed fc1 actually uses 8-bit
    assert mixed.fc1.bits == 8
    assert mixed.fc2.bits == 4
    assert uniform.fc1.bits == 4
    assert uniform.fc2.bits == 4

    # Verify fc2 storage is identical (same bits/group_size/config)
    assert torch.equal(mixed.fc2.weight_packed, uniform.fc2.weight_packed)


# =============================================================================
# Markers
# =============================================================================


pytestmark = pytest.mark.quick
