"""Tests for distributed parallel strategy helpers."""

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from llm.training.core.config import DistributedConfig
from llm.training.distributed.parallel import (
    model_state_dict,
    wrap_model_for_training,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_wrap_model_cpu_single_process():
    model = _Tiny()
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="ddp",
        device=torch.device("cpu"),
        world_size=1,
    )
    assert wrapped is model


def test_model_state_dict_bare_module():
    model = _Tiny()
    state = model_state_dict(model)
    assert "linear.weight" in state


def test_load_model_state_dict_roundtrip():
    from llm.training.distributed import load_model_state_dict, model_state_dict

    model = torch.nn.Linear(4, 2)
    state = model_state_dict(model)
    model2 = torch.nn.Linear(4, 2)
    load_model_state_dict(model2, state)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, model2.state_dict()[key])


def test_unknown_parallel_strategy_raises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for distributed wrap path")
    model = _Tiny().cuda()
    with pytest.raises(ValueError, match="Unknown parallel_strategy"):
        wrap_model_for_training(
            model,
            parallel_strategy="megatron",
            device=torch.device("cuda:0"),
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
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to build the world-size=1 FSDP input")
    model = _Tiny().cuda()
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="fsdp",
        device=torch.device("cuda:0"),
        world_size=1,
    )
    assert wrapped is model


def test_wrap_ddp_world_size_one_returns_unwrapped():
    """Single-rank DDP is also a no-op (returns the bare model)."""
    model = _Tiny()
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="ddp",
        device=torch.device("cpu"),
        world_size=1,
    )
    assert wrapped is model


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


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16"])
def test_fsdp_mixed_precision_builder_known_dtypes(dtype):
    """All documented dtypes build without raising."""
    from llm.training.distributed.parallel import _fsdp_mixed_precision

    result = _fsdp_mixed_precision(dtype)
    if dtype == "fp32":
        assert result is None
    else:
        assert result is not None


def test_fsdp_mixed_precision_builder_rejects_unknown():
    """Unknown dtype strings fail at policy-build time, not at FSDP init."""
    from llm.training.distributed.parallel import _fsdp_mixed_precision

    with pytest.raises(ValueError, match="fsdp_mixed_precision"):
        _fsdp_mixed_precision("int8")
