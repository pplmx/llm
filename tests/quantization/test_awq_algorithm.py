"""Tests for the AWQ algorithm core (config + activation stats + scale search)."""

from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

# === Config validation ===


def test_awq_config_default_values():
    """Default config is 4-bit, group_size=128, symmetric, n_grid=20."""
    from llm.quantization.awq import AWQConfig

    cfg = AWQConfig()
    assert cfg.bits == 4
    assert cfg.group_size == 128
    assert cfg.sym is True
    assert cfg.n_grid == 20
    assert cfg.clip_ratio is None


def test_awq_config_rejects_invalid_bits():
    """bits must be 4 or 8."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        AWQConfig(bits=16)


def test_awq_config_rejects_negative_group_size():
    """group_size must be -1 (per-channel) or positive."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(ValueError, match="group_size must be -1"):
        AWQConfig(group_size=-128)


def test_awq_config_rejects_zero_group_size():
    """group_size=0 is rejected by config validation."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(ValueError, match="group_size"):
        AWQConfig(group_size=0)


def test_awq_quantizer_rejects_non_divisible_group_size():
    """A group_size that does not divide in_features fails fast at quantizer
    construction (regression for ISS-022) instead of a late packing bug.
    """
    import torch.nn as nn

    from llm.quantization.awq import AWQConfig, AWQQuantizer

    layer = nn.Linear(100, 16)
    with pytest.raises(ValueError, match="must divide in_features"):
        AWQQuantizer(layer, AWQConfig(bits=4, group_size=30, sym=True))


def test_awq_config_rejects_nonpositive_grid():
    """n_grid must be >= 1."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(ValueError, match="n_grid"):
        AWQConfig(n_grid=0)


def test_awq_config_rejects_invalid_clip_ratio():
    """clip_ratio must be in (0, 0.5] or None."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(ValueError, match="clip_ratio"):
        AWQConfig(clip_ratio=0.0)
    with pytest.raises(ValueError, match="clip_ratio"):
        AWQConfig(clip_ratio=0.9)
    assert AWQConfig(clip_ratio=0.2).clip_ratio == 0.2


def test_awq_config_is_frozen():
    """Config is frozen (immutable)."""
    from llm.quantization.awq import AWQConfig

    cfg = AWQConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.bits = 4  # type: ignore[misc]


def test_awq_config_accepts_layer_policies():
    """LayerQuantPolicy tuples are accepted (mixed-precision dispatch)."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.awq import AWQConfig

    cfg = AWQConfig(layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),))
    assert cfg.layer_policies[0].bits == 8


def test_awq_config_rejects_non_policy_element():
    """layer_policies entries must be LayerQuantPolicy instances."""
    from llm.quantization.awq import AWQConfig

    with pytest.raises(TypeError, match="LayerQuantPolicy"):
        AWQConfig(layer_policies=("not-a-policy",))  # type: ignore[arg-type]


# === AWQQuantizer activation accumulation ===


def test_quantizer_accumulates_activation_stats():
    """add_batch accumulates mean-abs activation per input channel."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    layer = nn.Linear(8, 4)
    quantizer = AWQQuantizer(layer, AWQConfig())

    x1 = torch.randn(5, 8)
    x2 = torch.randn(7, 8)
    quantizer.add_batch(x1)
    quantizer.add_batch(x2)

    expected = torch.cat([x1, x2]).abs().mean(dim=0)
    assert torch.allclose(quantizer.act_mean(), expected)


def test_quantizer_flattens_leading_dims():
    """add_batch flattens leading dims before accumulating stats."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    layer = nn.Linear(8, 4)
    quantizer = AWQQuantizer(layer, AWQConfig())
    x = torch.randn(2, 3, 8)
    quantizer.add_batch(x)
    assert quantizer.n_samples == 2 * 3
    assert quantizer.act_mean().shape == (8,)


def test_quantizer_empty_raises():
    """act_mean without calibration data fails loudly."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    layer = nn.Linear(8, 4)
    quantizer = AWQQuantizer(layer, AWQConfig())
    with pytest.raises(RuntimeError, match="No calibration data"):
        quantizer.act_mean()


# === Scale search: the AWQ headline behavior ===


def _layer_error_with_scales(layer: nn.Linear, scales: torch.Tensor, act_mean: torch.Tensor, bits: int) -> float:
    """Activation-weighted reconstruction error of the layer under `scales`."""
    from llm.quantization.awq import _group_quantize_dequant

    w = layer.weight.data.float()
    recon = _group_quantize_dequant(w * scales, bits, 128) / scales
    return (((w - recon) ** 2) * act_mean).sum().item()


def test_awq_search_protects_salient_channels():
    """The headline AWQ property: activation-aware scales beat naive RTN.

    With one strongly salient input channel (large activation magnitude),
    the search should scale that channel's weights up (finer quantization
    relative to the group grid) and materially reduce the activation-weighted
    reconstruction error versus weight-only group quantization.
    """
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    torch.manual_seed(1)
    layer = nn.Linear(16, 16)
    calib = torch.randn(256, 16)
    calib[:, 0] *= 50.0  # channel 0 is salient
    act_mean = calib.abs().mean(0)

    quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=128))
    quantizer.add_batch(calib)
    scale = quantizer.search_scale()

    # Salient channel should be scaled up (>1) and be the most-protected one.
    assert scale[0].item() > 1.0
    assert scale[0].item() >= scale.max().item()

    err_naive = _layer_error_with_scales(layer, torch.ones(16), act_mean, bits=4)
    err_awq = _layer_error_with_scales(layer, scale, act_mean, bits=4)
    assert err_awq < err_naive, f"AWQ ({err_awq:.6f}) should beat naive RTN ({err_naive:.6f})"


def test_awq_search_never_worse_on_uniform_activations():
    """With uniform activation magnitudes the search must not degrade quality."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    torch.manual_seed(2)
    layer = nn.Linear(16, 16)
    calib = torch.randn(256, 16)
    act_mean = calib.abs().mean(0)

    quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=128))
    quantizer.add_batch(calib)
    scale = quantizer.search_scale()

    err_naive = _layer_error_with_scales(layer, torch.ones(16), act_mean, bits=4)
    err_awq = _layer_error_with_scales(layer, scale, act_mean, bits=4)
    assert err_awq <= err_naive + 1e-6


def test_awq_search_matches_packed_dequantization():
    """The search's error model equals the packed layer's actual dequantization.

    `_group_quantize_dequant` is the exact math used both inside the search
    and at packing time, so the searched scale is honest about the final
    layer's behavior.
    """
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, AWQQuantizer, _group_quantize_dequant

    torch.manual_seed(3)
    layer = nn.Linear(32, 16)
    calib = torch.randn(64, 32)
    quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=8))
    quantizer.add_batch(calib)
    scale = quantizer.search_scale()
    packed, scales, effective_gs = quantizer.quantize(scale)

    awq_layer = AWQQuantizedLinear(
        in_features=32,
        out_features=16,
        bias=False,
        weight_packed=packed,
        scales=scales.to(torch.float16),
        bits=4,
        group_size=effective_gs,
        sym=True,
        input_scales=scale.to(torch.float16),
    )

    w = layer.weight.data.float()
    expected = _group_quantize_dequant(w * scale, 4, 8) / scale
    got = awq_layer._unpack_weights().to(torch.float32)
    got = (got - 8.0) * scales.repeat_interleave(8, dim=1).to(torch.float32)
    assert torch.allclose(got, expected, atol=1e-5)


def test_awq_search_per_channel_group_mode():
    """group_size=-1 (per-channel) search runs and returns sane scales."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    torch.manual_seed(4)
    layer = nn.Linear(16, 8)
    calib = torch.randn(64, 16)
    quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=-1))
    quantizer.add_batch(calib)
    scale = quantizer.search_scale()
    assert scale.shape == (16,)
    assert (scale > 0).all()


def test_awq_clip_ratio_bounds_outliers():
    """clip_ratio actually clamps the outlier in the packed reconstruction."""
    from llm.quantization._gptq_layer import _unpack_4bit
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    torch.manual_seed(5)
    layer = nn.Linear(16, 16)
    with torch.no_grad():
        layer.weight[0, 0] = 50.0  # one extreme outlier
    calib = torch.randn(64, 16)

    quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=128, clip_ratio=0.05))
    quantizer.add_batch(calib)
    scale = quantizer.search_scale()
    assert scale.shape == (16,)

    packed, scales, gs = quantizer.quantize(scale)
    unpacked = _unpack_4bit(packed, numel=16 * 16).reshape(16, 16).to(torch.float32) - 8.0
    recon = unpacked * scales.repeat_interleave(gs, dim=1).to(torch.float32)

    # With max=50 and rho=0.05 the clip upper bound is well below 50.
    w = layer.weight.data.float()
    rho = 0.05
    upper = w.max() - rho * (w.max() - w.min())
    assert recon.abs().max().item() <= upper + 1e-3


def test_awq_search_rejects_asymmetric():
    """sym=False is a v1 NotImplementedError (asymmetric AWQ is a follow-up)."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    layer = nn.Linear(8, 8)
    quantizer = AWQQuantizer(layer, AWQConfig(sym=False))
    quantizer.add_batch(torch.randn(16, 8))
    with pytest.raises(NotImplementedError, match="Asymmetric"):
        quantizer.search_scale()


def test_4bit_odd_total_count_fails_fast_per_group_and_per_channel():
    """4-bit packing stores two weights per byte over the whole tensor, so an
    odd TOTAL weight count must fail fast for per-GROUP 4-bit too (round-81
    quant deep-dive F3 / TASK-198; only group_size == -1 was guarded before)."""
    from llm.quantization.awq import AWQConfig, AWQQuantizer

    for group_size in (3, -1):
        layer = nn.Linear(9, 7)
        quantizer = AWQQuantizer(layer, AWQConfig(bits=4, group_size=group_size))
        scale = torch.ones(9)  # per-input-channel activation scale
        with pytest.raises(ValueError, match="even total weight count"):
            quantizer.quantize(scale)
