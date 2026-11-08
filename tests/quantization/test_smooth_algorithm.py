"""Tests for the SmoothQuant algorithm core (config + scale math + alpha search)."""

from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

# === Config validation ===


def test_smooth_config_default_values():
    """Default config is alpha=0.5, INT8 symmetric, per-channel weights."""
    from llm.quantization.smooth import SmoothQuantConfig

    cfg = SmoothQuantConfig()
    assert cfg.alpha == 0.5
    assert cfg.search_alpha is False
    assert cfg.bits == 8
    assert cfg.group_size == -1
    assert cfg.sym is True


def test_smooth_config_rejects_alpha_out_of_range():
    """alpha must be in [0, 1]."""
    from llm.quantization.smooth import SmoothQuantConfig

    with pytest.raises(ValueError, match="alpha must be in"):
        SmoothQuantConfig(alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        SmoothQuantConfig(alpha=1.1)
    assert SmoothQuantConfig(alpha=0.0).alpha == 0.0
    assert SmoothQuantConfig(alpha=1.0).alpha == 1.0


def test_smooth_config_rejects_non_int8_bits():
    """SmoothQuant is an INT8 method in v1 — only bits=8 is accepted."""
    from llm.quantization.smooth import SmoothQuantConfig

    with pytest.raises(ValueError, match="bits must be 8"):
        SmoothQuantConfig(bits=4)


def test_smooth_config_rejects_non_per_channel_group_size():
    """group_size must be -1 (per-channel) in v1."""
    from llm.quantization.smooth import SmoothQuantConfig

    with pytest.raises(ValueError, match="group_size must be -1"):
        SmoothQuantConfig(group_size=128)


def test_smooth_config_is_frozen():
    """Config is frozen (immutable)."""
    from llm.quantization.smooth import SmoothQuantConfig

    cfg = SmoothQuantConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.alpha = 0.5  # type: ignore[misc]


def test_smooth_config_rejects_non_policy_element():
    """layer_policies entries must be LayerQuantPolicy instances."""
    from llm.quantization.smooth import SmoothQuantConfig

    with pytest.raises(TypeError, match="LayerQuantPolicy"):
        SmoothQuantConfig(layer_policies=("not-a-policy",))  # type: ignore[arg-type]


# === Smoothing scale math ===


def test_smoothing_scales_balanced_alpha():
    """alpha=0.5 gives s_j = sqrt(act_max[j] / w_max[j])."""
    from llm.quantization.smooth import _smoothing_scales

    act_max = torch.tensor([4.0, 1.0, 0.0])
    w_max = torch.tensor([1.0, 16.0, 8.0])
    s = _smoothing_scales(act_max, w_max, alpha=0.5)
    assert torch.allclose(s[:2], torch.sqrt(act_max[:2] / w_max[:2]))
    # Dead channel (act_max=0) gets scale 1, not 0.
    assert s[2].item() == 1.0


def test_smoothing_scales_extremes():
    """alpha=0 normalizes weights; alpha=1 normalizes activations."""
    from llm.quantization.smooth import _smoothing_scales

    act_max = torch.tensor([4.0, 9.0])
    w_max = torch.tensor([2.0, 3.0])
    s0 = _smoothing_scales(act_max, w_max, alpha=0.0)
    s1 = _smoothing_scales(act_max, w_max, alpha=1.0)
    # alpha=0: s = 1/w_max → W·s has per-channel max 1.
    assert torch.allclose(s0, 1.0 / w_max)
    # alpha=1: s = act_max → x/s has per-channel max 1.
    assert torch.allclose(s1, act_max)


def test_activation_scale_is_per_tensor_from_stats():
    """The per-tensor activation scale equals max_j(act_max[j]/s_j)/127."""
    from llm.quantization.smooth import _activation_scale, _smoothing_scales

    act_max = torch.tensor([4.0, 16.0])
    w_max = torch.tensor([2.0, 1.0])
    s = _smoothing_scales(act_max, w_max, alpha=0.5)
    assert _activation_scale(act_max, s) == pytest.approx((act_max / s).max().item() / 127.0)


# === Quantizer accumulation ===


def test_quantizer_accumulates_per_channel_max():
    """add_batch tracks the elementwise max abs activation per channel."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    layer = nn.Linear(8, 4)
    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig())
    x1 = torch.randn(5, 8)
    x2 = torch.randn(7, 8)
    quantizer.add_batch(x1)
    quantizer.add_batch(x2)
    expected = torch.cat([x1, x2]).abs().max(dim=0)[0]
    assert torch.allclose(quantizer.act_max, expected)
    assert quantizer.n_samples == 12


def test_quantizer_empty_raises():
    """quantize without calibration data fails loudly."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    layer = nn.Linear(8, 4)
    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig())
    with pytest.raises(RuntimeError, match="No calibration data"):
        quantizer.quantize()


def test_quantizer_search_alpha_retains_batches():
    """search_alpha=True retains batches for candidate evaluation."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    layer = nn.Linear(8, 4)
    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig(search_alpha=True))
    quantizer.add_batch(torch.randn(5, 8))
    assert len(quantizer._batches) == 1


# === Headline: activation outliers ===


def _layer_output_error(layer: nn.Linear, calib: torch.Tensor, alpha: float) -> float:
    """Mean-squared output error of SmoothQuant at a given alpha vs fp32."""
    from llm.quantization.smooth import _eval_layer_error

    w = layer.weight.data.float()
    act_max = calib.abs().max(dim=0)[0]
    return _eval_layer_error(w, act_max, [calib], alpha, layer.bias)


def _naive_int8_output_error(layer: nn.Linear, calib: torch.Tensor) -> float:
    """Output error of INT8 W+A quantization WITHOUT any smoothing (s=1)."""
    w = layer.weight.data.float()
    act_max = calib.abs().max(dim=0)[0]

    w_scale = (w.abs().max(dim=1, keepdim=True)[0] / 127.0).clamp(min=1e-8)
    w_int8 = torch.round(w / w_scale).clamp(-128, 127)
    act_scale = act_max.max().item() / 127.0

    total_err = 0.0
    total_n = 0
    for x in calib.unsqueeze(0):
        x = x.float()
        x_q = torch.clamp(torch.round(x / act_scale), -128, 127) * act_scale
        y_q = torch.nn.functional.linear(x_q, w_int8 * w_scale, layer.bias)
        y_ref = torch.nn.functional.linear(x, w, layer.bias)
        total_err += (y_q - y_ref).pow(2).sum().item()
        total_n += y_ref.numel()
    return total_err / total_n


def test_quantizer_search_alpha_with_fp16_bias():
    """Regression: ``search_alpha=True`` must not crash on a
    layer whose bias is fp16/bf16 (e.g. ``model.half()``).

    ``_eval_layer_error`` casts the weights and activations to fp32 but fed
    ``bias`` through untouched, so ``F.linear(x_fp32, w_fp32, bias_fp16)``
    raised ``RuntimeError: self and mat2 must have the same dtype`` during
    the alpha grid search."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    torch.manual_seed(0)
    layer = nn.Linear(16, 16).half()  # fp16 bias + weights
    calib = torch.randn(8, 16)

    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig(search_alpha=True))
    quantizer.add_batch(calib)

    weight_packed, _weight_scales, act_scale, _input_scales = quantizer.quantize()
    assert weight_packed.numel() == layer.weight.numel()
    assert act_scale.item() > 0


def test_smoothquant_beats_no_smoothing_on_outlier_activations():
    """The headline SmoothQuant property: smoothing cuts INT8 W+A error.

    With one activation outlier channel, per-tensor INT8 activation
    quantization without smoothing has huge error; migrating the difficulty
    into the weights (alpha=0.5) reduces the output reconstruction error.
    """
    torch.manual_seed(1)
    layer = nn.Linear(16, 16)
    calib = torch.randn(256, 16)
    calib[:, 0] *= 30.0  # activation outlier channel

    err_naive = _naive_int8_output_error(layer, calib)
    err_smooth = _layer_output_error(layer, calib, alpha=0.5)
    assert err_smooth < err_naive, f"SmoothQuant ({err_smooth:.6f}) should beat no smoothing ({err_naive:.6f})"


def test_smoothquant_alpha_search_picks_lower_error():
    """search_alpha finds an alpha at least as good as the naive extreme."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    torch.manual_seed(2)
    layer = nn.Linear(16, 16)
    calib = torch.randn(128, 16)
    calib[:, 0] *= 20.0

    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig(search_alpha=True))
    quantizer.add_batch(calib)
    quantizer.quantize()  # must not raise; the search runs over the grid

    err_searched = min(_layer_output_error(layer, calib, a) for a in (0.25, 0.5, 0.75, 1.0))
    err_naive = _naive_int8_output_error(layer, calib)
    assert err_searched < err_naive


def test_smoothquant_never_materially_worse_on_clean_activations():
    """On well-behaved activations smoothing does not degrade quality."""
    torch.manual_seed(3)
    layer = nn.Linear(16, 16)
    calib = torch.randn(128, 16)

    err_naive = _naive_int8_output_error(layer, calib)
    err_smooth = _layer_output_error(layer, calib, alpha=0.5)
    assert err_smooth <= err_naive * 1.5 + 1e-9


def test_smoothquant_rejects_asymmetric():
    """sym=False is a v1 NotImplementedError."""
    from llm.quantization.smooth import SmoothQuantConfig, SmoothQuantQuantizer

    layer = nn.Linear(8, 8)
    quantizer = SmoothQuantQuantizer(layer, SmoothQuantConfig(sym=False))
    quantizer.add_batch(torch.randn(16, 8))
    with pytest.raises(NotImplementedError, match="Asymmetric"):
        quantizer.quantize()


def test_zero_activation_calibration_does_not_produce_nan():
    """A layer whose calibration activations are ALL zero yields act_scale=0;
       forward must degrade to a finite near-zero output, not NaN (/
    — the dead-layer guard GPTQ already has)."""
    from llm.quantization.smooth import SmoothQuantConfig, _quantize_linear_with_smoothquant

    layer = nn.Linear(8, 4)
    zero_calib = [torch.zeros(4, 8) for _ in range(2)]
    qlayer = _quantize_linear_with_smoothquant(layer, zero_calib, SmoothQuantConfig())

    out_zero = qlayer(torch.zeros(3, 8))
    assert torch.isfinite(out_zero).all()
    assert torch.allclose(out_zero, torch.zeros(3, 4), atol=1e-6)

    # A live input also stays finite (act_scale clamped to a small floor).
    out_live = qlayer(torch.randn(3, 8))
    assert torch.isfinite(out_live).all()


def test_vanishing_channel_input_scales_do_not_produce_nan():
    """A nearly-dead channel (act_max tiny but nonzero, e.g. 1e-20) is not the
    exact-zero 'dead' mask, so under alpha=1.0 _smoothing_scales yields a
    sub-fp16 input scale that flushed to 0 in the packed fp16 buffer — and a
    live forward carrying an exact 0.0 on that channel hit 0/0 -> NaN. The
    fp16-safe floor must keep the forward finite."""
    from llm.quantization.smooth import SmoothQuantConfig, _quantize_linear_with_smoothquant

    layer = nn.Linear(8, 4)
    calib = [torch.zeros(4, 8)]
    calib[0][:, 7] = 1e-20  # nearly-dead channel, not exactly dead
    qlayer = _quantize_linear_with_smoothquant(layer, calib, SmoothQuantConfig(alpha=1.0))

    # Live forward: the nearly-dead channel emits exact 0.0 activation.
    x = torch.randn(3, 8)
    x[:, 7] = 0.0
    out = qlayer(x)
    assert torch.isfinite(out).all()


def test_smoothing_preserves_the_layer_output_exactly():
    """The core SmoothQuant identity: ``x @ Wᵀ == (x/s) @ (s·W)ᵀ``.

    Everything else in this module rests on it — the activation side divides by
    ``s`` and the weight side multiplies by ``s``, so the layer must compute the
    same function before quantization. If the ``s`` formula were wrong (say the
    alpha exponents swapped), the scales would still look plausible per-channel
    while every layer silently computed something different, and no existing test
    would notice because they only check the scale values themselves.
    """
    import torch

    from llm.quantization.smooth import _smoothing_scales

    torch.manual_seed(0)
    weight = torch.randn(16, 32) * 0.05
    activations = torch.randn(4, 32)
    act_max = activations.abs().amax(dim=0)
    w_max = weight.abs().amax(dim=0)

    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = _smoothing_scales(act_max, w_max, alpha)
        original = activations @ weight.T
        smoothed = (activations / s) @ (weight * s).T
        assert torch.allclose(original, smoothed, atol=1e-5, rtol=1e-5), (
            f"alpha={alpha}: smoothing changed the layer output by {(original - smoothed).abs().max().item():.2e}"
        )


def test_act_scale_floor_is_representable_in_fp16():
    """``_ACT_SCALE_FLOOR`` must survive the fp16 round trip exactly.

    The quantizer stores ``act_scale`` / ``input_scales`` as fp16. A floor that
    is not fp16-exact would either flush to zero (dividing by it NaNs the
    forward) or round to a different value than the floored scale, which is the
    failure mode the constant exists to prevent.
    """
    import torch

    from llm.quantization.smooth import _ACT_SCALE_FLOOR

    stored = torch.tensor(_ACT_SCALE_FLOOR).to(torch.float16).to(torch.float32)
    assert stored.item() == _ACT_SCALE_FLOOR, f"floor {_ACT_SCALE_FLOOR} is not fp16-exact (became {stored.item()})"
    assert _ACT_SCALE_FLOOR > 0.0
    assert torch.finfo(torch.float16).tiny == _ACT_SCALE_FLOOR, (
        "the floor should be the fp16 smallest normal so it is the largest exact value that still never flushes"
    )


def test_dead_channels_get_unit_scale_not_zero():
    """A channel with no activation or no weight must not be scaled.

    ``s = a**alpha / w**(1-alpha)`` collapses to 0 or inf on a dead channel;
    the implementation substitutes 1 so the channel passes through untouched
    instead of dividing by zero in the forward.
    """
    import torch

    from llm.quantization.smooth import _smoothing_scales

    act_max = torch.tensor([1.0, 0.0, 2.0, 0.0, 0.5])
    w_max = torch.tensor([0.1, 0.2, 0.0, 0.0, 0.3])
    dead = (act_max == 0) | (w_max == 0)

    for alpha in (0.0, 0.5, 1.0):
        s = _smoothing_scales(act_max, w_max, alpha)
        assert torch.isfinite(s).all(), f"alpha={alpha}: non-finite scale {s.tolist()}"
        assert (s > 0).all(), f"alpha={alpha}: non-positive scale {s.tolist()}"
        assert torch.allclose(s[dead], torch.ones_like(s[dead])), (
            f"alpha={alpha}: dead channels should pass through unscaled, got {s[dead].tolist()}"
        )


def test_smoothquant_reduces_activation_outliers_at_alpha_half():
    """The point of the transform: smoothed activations are flatter per channel.

    With ``alpha=0.5`` the per-channel activation range after dividing by ``s``
    should be compressed relative to the raw range — that compression is why
    INT8 activation quantization works at all. A wrong ``s`` direction (dividing
    where it should multiply) would leave the outliers untouched or worse.
    """
    import torch

    from llm.quantization.smooth import _smoothing_scales

    torch.manual_seed(0)
    weight = torch.randn(16, 32) * 0.05
    # One channel dominates, as in real transformer activations.
    activations = torch.randn(8, 32)
    activations[:, 0] *= 50.0

    act_max = activations.abs().amax(dim=0)
    w_max = weight.abs().amax(dim=0)
    s = _smoothing_scales(act_max, w_max, 0.5)

    raw_spread = (act_max / act_max.min()).max().item()
    smoothed_spread = ((act_max / s) / (act_max / s).min()).max().item()
    assert smoothed_spread < raw_spread, (
        f"alpha=0.5 should flatten the per-channel activation range: "
        f"raw {raw_spread:.2f} -> smoothed {smoothed_spread:.2f}"
    )
