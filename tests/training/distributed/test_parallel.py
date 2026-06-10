"""Tests for distributed parallel strategy helpers."""

import pytest
import torch
import torch.nn as nn

from llm.training.distributed.parallel import model_state_dict, wrap_model_for_training


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
