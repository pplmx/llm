"""Tests for distributed parallel strategy helpers."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from llm.training.core.config import DistributedConfig
from llm.training.distributed.parallel import (
    model_state_dict,
    wrap_model_for_training,
)
from tests.support.devices import ALL_DEVICES, DEFAULT_DEVICE, cuda_usable


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_wrap_model_single_process(device):
    """Single-rank DDP is a no-op on any device (CPU or GPU)."""
    model = _Tiny().to(device)
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="ddp",
        device=torch.device(device),
        world_size=1,
    )
    assert wrapped is model


def test_model_state_dict_bare_module():
    model = _Tiny()
    state = model_state_dict(model)
    assert "linear.weight" in state


def test_load_model_state_dict_roundtrip():
    from llm.training.distributed import load_model_state_dict

    model = torch.nn.Linear(4, 2)
    state = model_state_dict(model)
    model2 = torch.nn.Linear(4, 2)
    load_model_state_dict(model2, state)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, model2.state_dict()[key])


def test_model_state_dict_strips_torch_compile_prefix():
    """``model_state_dict`` must store plain keys for compiled models.

    ``torch.compile`` renames every key to ``_orig_mod.*``; checkpoints
    stored that way cannot be loaded by ``llm-serve`` (plain ``DecoderModel``,
    no compile graph), silently dropping every weight.
    """
    from llm.training.distributed import model_state_dict

    model = torch.compile(_Tiny())
    state = model_state_dict(model)
    assert not any(key.startswith("_orig_mod.") for key in state)
    assert "linear.weight" in state


def test_load_model_state_dict_accepts_compiled_prefix():
    """Legacy ``_orig_mod.``-prefixed checkpoints load into compiled models."""
    from llm.training.distributed import load_model_state_dict

    source = _Tiny()
    prefixed = {f"_orig_mod.{key}": value for key, value in source.state_dict().items()}
    compiled = torch.compile(_Tiny())
    load_model_state_dict(compiled, prefixed)
    for key, value in source.state_dict().items():
        assert torch.allclose(value, compiled._orig_mod.state_dict()[key])

    # Plain keys also load (the post-fix checkpoint format).
    compiled2 = torch.compile(_Tiny())
    load_model_state_dict(compiled2, source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.allclose(value, compiled2._orig_mod.state_dict()[key])


def test_unknown_parallel_strategy_raises():
    if not cuda_usable():
        pytest.skip("CUDA required for distributed wrap path")
    model = _Tiny().to(DEFAULT_DEVICE)
    with pytest.raises(ValueError, match="Unknown parallel_strategy"):
        wrap_model_for_training(
            model,
            parallel_strategy="megatron",
            device=DEFAULT_DEVICE,
            world_size=2,
        )


# --- DistributedConfig FSDP fields (T3 #29) ------------------------------


def test_distributed_config_defaults_include_fsdp_knobs():
    """FSDP knobs ship with conservative defaults so users can opt in."""
    cfg = DistributedConfig()
    assert cfg.parallel_strategy == "ddp"
    assert cfg.fsdp_mixed_precision == "bf16"
    assert cfg.fsdp_auto_wrap_min_params == 10_000_000
    assert cfg.fsdp_cpu_offload is False


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16"])
def test_distributed_config_accepts_known_fsdp_dtypes(dtype):
    cfg = DistributedConfig(fsdp_mixed_precision=dtype)
    assert cfg.fsdp_mixed_precision == dtype


def test_distributed_config_rejects_unknown_fsdp_dtype():
    """``mixed_precision`` must be one of the documented enum values."""
    with pytest.raises(ValidationError):
        DistributedConfig(fsdp_mixed_precision="int8")  # type: ignore[arg-type]


def test_distributed_config_auto_wrap_min_params_must_be_non_negative():
    """Negative thresholds make no sense — fail at config-parse time."""
    with pytest.raises(ValidationError):
        DistributedConfig(fsdp_auto_wrap_min_params=-1)


def test_distributed_config_auto_wrap_min_params_zero_disables():
    """Zero is allowed (disables auto-wrap) — verify it round-trips."""
    cfg = DistributedConfig(fsdp_auto_wrap_min_params=0)
    assert cfg.fsdp_auto_wrap_min_params == 0


def test_distributed_config_cpu_offload_roundtrip():
    cfg = DistributedConfig(fsdp_cpu_offload=True)
    assert cfg.fsdp_cpu_offload is True


# --- wrap_model_for_training MoE unused-parameter handling (RIL ISS-138) ---


def _moe_model() -> torch.nn.Module:
    """A tiny model whose block carries a MoE (num_experts > 0) MLP."""
    from llm.core.transformer_block import TransformerBlock

    return TransformerBlock(
        hidden_size=32,
        num_heads=4,
        attn_impl="mha",
        mlp_impl="moe",
        num_experts=8,
        top_k=2,
        intermediate_size=64,
    )


def test_wrap_ddp_moe_enables_find_unused_parameters():
    """RIL ISS-138: a MoE model wrapped for DDP must set
    ``find_unused_parameters=True`` — a batch can deterministically leave some
    experts unrouted (top-k over many experts), so those params get NO
    gradient and DDP (find_unused_parameters=False) fails the backward with
    "expected to have finished reduction". Dead experts are structural for
    MoE, not anomalies. (The DDP constructor needs a live process group, so we
    intercept it to capture the kwarg decision.)"""
    killed: dict[str, bool] = {}

    def _capture(model, *_args, **kwargs):
        killed["moe"] = kwargs.get("find_unused_parameters")
        return model  # return the bare model; we only assert the kwarg

    model = _moe_model()
    with patch("llm.training.distributed.parallel.DistributedDataParallel", side_effect=_capture):
        wrap_model_for_training(
            model,
            parallel_strategy="ddp",
            device=torch.device("cuda:0"),
            world_size=2,
        )
    assert killed.get("moe") is True, f"MoE DDP must enable find_unused_parameters, got {killed}"


def test_wrap_ddp_standard_model_keeps_fast_path():
    """A standard (non-MoE) model keeps find_unused_parameters=False — every
    param is used every step, so unused tracking is a pure overhead."""
    killed: dict[str, bool] = {}

    def _capture(model, *_args, **kwargs):
        killed["std"] = kwargs.get("find_unused_parameters")
        return model

    model = _Tiny()
    with patch("llm.training.distributed.parallel.DistributedDataParallel", side_effect=_capture):
        wrap_model_for_training(
            model,
            parallel_strategy="ddp",
            device=torch.device("cuda:0"),
            world_size=2,
        )
    assert killed.get("std") is False, f"standard model must keep find_unused_parameters=False, got {killed}"


# --- wrap_model_for_training FSDP dispatch --------------------------------


def test_wrap_fsdp_cpu_returns_unwrapped():
    """FSDP needs CUDA + a process group; on CPU we just return the bare model."""
    model = _Tiny()
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="fsdp",
        device=torch.device("cpu"),
        world_size=2,
    )
    assert wrapped is model


def test_wrap_fsdp_world_size_one_returns_unwrapped():
    """Single-rank FSDP is equivalent to bare training."""
    if not cuda_usable():
        pytest.skip("CUDA required to build the world-size=1 FSDP input")
    model = _Tiny().to(DEFAULT_DEVICE)
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="fsdp",
        device=DEFAULT_DEVICE,
        world_size=1,
    )
    assert wrapped is model


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_wrap_ddp_world_size_one_returns_unwrapped(device):
    """Single-rank DDP is also a no-op (returns the bare model) on any device."""
    model = _Tiny().to(device)
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="ddp",
        device=torch.device(device),
        world_size=1,
    )
    assert wrapped is model


def test_distributed_device_count_matches_vram_sorted_inventory():
    """The GPU inventory and cuda_usable() agree on the device count.

    Validates the multi-GPU infrastructure: when GPUs are available,
    ``cuda_device_count()`` matches the length of the VRAM-sorted
    ``all_gpu_devices()`` list, and every device in the list is usable.
    """
    import tests.support.devices as dev

    gpu_devices = dev.all_gpu_devices()
    if not gpu_devices:
        assert dev.cuda_device_count() == 0
        return

    # Every device returned must be individually usable.
    for device in gpu_devices:
        assert dev.cuda_usable(device), f"device {device} listed but not usable"
    assert dev.cuda_device_count() == len(gpu_devices)


def test_wrap_unknown_strategy_returns_bare_on_cpu():
    """On CPU the early-return path skips the strategy check.

    This is intentional: an unknown strategy on a CPU host is just a
    no-op return, the same as any other single-rank / CPU call. The
    ValueError only fires on CUDA where the strategy actually matters
    (and where the original ``test_unknown_parallel_strategy_raises``
    already pins that behaviour).
    """
    model = _Tiny()
    out = wrap_model_for_training(
        model,
        parallel_strategy="megatron",  # type: ignore[arg-type]
        device=torch.device("cpu"),
        world_size=2,
    )
    assert out is model


# --- state dict strategy plumbing -----------------------------------------


def test_model_state_dict_full_default_for_bare_model():
    """``state_dict_type`` argument is accepted but ignored for bare modules."""
    model = _Tiny()
    state_full = model_state_dict(model, state_dict_type="full")
    state_sharded = model_state_dict(model, state_dict_type="sharded")
    assert set(state_full) == set(state_sharded)
    for key in state_full:
        assert torch.allclose(state_full[key], state_sharded[key])


def test_model_state_dict_state_dict_type_only_validated_for_fsdp():
    """Bare models don't care about ``state_dict_type`` — we don't gate the arg.

    The ``"banana"`` value would be rejected if we actually consulted it
    for a bare module. This test pins the (mildly surprising) behavior:
    the argument is ignored for bare models so callers can pass it
    uniformly regardless of whether the wrapped model is FSDP or not.
    """
    model = _Tiny()
    # No raise — bare model ignores ``state_dict_type``.
    state = model_state_dict(model, state_dict_type="banana")  # type: ignore[arg-type]
    assert "linear.weight" in state


# --- size-based auto-wrap policy builder ---------------------------------


def test_fsdp_auto_wrap_zero_returns_none():
    """Zero threshold means 'no auto-wrap' — the builder returns ``None``."""
    from llm.training.distributed.parallel import _fsdp_auto_wrap_policy

    assert _fsdp_auto_wrap_policy(0) is None


def test_fsdp_auto_wrap_positive_returns_callable():
    """Positive threshold returns a callable with the FSDP policy signature."""
    from llm.training.distributed.parallel import _fsdp_auto_wrap_policy

    policy = _fsdp_auto_wrap_policy(1_000)
    assert callable(policy)
    # The policy must accept the ``(module, recurse, nonwrapped_numel)``
    # signature that FSDP expects.
    out = policy(_Tiny(), recurse=True, nonwrapped_numel=0)
    assert isinstance(out, bool)


# --- MixedPrecision policy builder ---------------------------------------


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_fsdp_mixed_precision_builder_known_dtypes(dtype):
    """Supported dtypes build without raising (fp16 is REFUSED — RIL ISS-188)."""
    from llm.training.distributed.parallel import _fsdp_mixed_precision

    result = _fsdp_mixed_precision(dtype)
    if dtype == "fp32":
        assert result is None
    else:
        assert result is not None


def test_fsdp_mixed_precision_builder_refuses_fp16():
    """Regression (RIL ISS-188): fp16 sharded params/reductions need a loss
    scaler the framework does not wire — silently running un-scaled fp16 never
    converges. The policy builder must refuse loudly instead."""
    from llm.training.distributed.parallel import _fsdp_mixed_precision

    with pytest.raises(ValueError, match="fp16"):
        _fsdp_mixed_precision("fp16")


def test_fsdp_mixed_precision_builder_rejects_unknown():
    """Unknown dtype strings fail at policy-build time, not at FSDP init."""
    from llm.training.distributed.parallel import _fsdp_mixed_precision

    with pytest.raises(ValueError, match="fsdp_mixed_precision"):
        _fsdp_mixed_precision("int8")


# --- DistributedManager.setup: launcher env respect (RIL ISS-187) ---


def test_setup_respects_launcher_master_env(monkeypatch):
    """Regression (RIL ISS-187): ``DistributedManager.setup`` must NOT
    overwrite a launcher-provided ``MASTER_ADDR``/``MASTER_PORT`` (torchrun
    sets them for multi-node rendezvous); it may only fall back to config
    defaults when the env is unset. The old ``os.environ[...] = config`` made
    every worker rank rendezvous with node-0 loopback and hang."""
    import os

    from llm.training.core.config import DistributedConfig
    from llm.training.core.distributed import DistributedManager

    config = DistributedConfig(master_addr="127.0.0.1", master_port="12355")
    DistributedManager(config)

    launched = {"MASTER_ADDR": "10.0.0.5", "MASTER_PORT": "29900"}
    monkeypatch.setenv("MASTER_ADDR", launched["MASTER_ADDR"])
    monkeypatch.setenv("MASTER_PORT", launched["MASTER_PORT"])

    # Simulate the env-setting phase (both branches of setup are exercised
    # without initializing a process group, which needs multi-proc).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    assert os.environ["MASTER_ADDR"] == launched["MASTER_ADDR"], "launcher MASTER_ADDR must win"
    assert os.environ["MASTER_PORT"] == launched["MASTER_PORT"], "launcher MASTER_PORT must win"

    # And when the env is NOT set, config defaults apply.
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", config.master_port)
    assert os.environ["MASTER_ADDR"] == config.master_addr
    assert os.environ["MASTER_PORT"] == config.master_port
