"""Tests for :func:`llm.training.core.distributed.broadcast_parameters`.

The DDP init contract requires every rank to start from identical weights;
``broadcast_parameters`` syncs rank-0's parameters and buffers right after a
fresh model build so a multi-GPU run does not optimise a different model per
rank (regression: `DistributedManager.setup` seeds ``42 + rank`` per rank, so
`build_model()` RNG-initialises differently on each rank and DDP only ever
averages gradients of *different* models).
"""

from __future__ import annotations

from unittest import mock

import torch
import torch.nn as nn

from llm.training.core.distributed import broadcast_parameters


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.register_buffer("running_mean", torch.zeros(2))

    def forward(self, x):
        return self.linear(x)


def test_noop_without_process_group(monkeypatch):
    """No live process group -> must not raise and must not touch weights."""
    monkeypatch.setattr("llm.training.core.distributed.dist.is_available", lambda: False)
    model = _Tiny()
    before = model.linear.weight.clone()
    broadcast_parameters(model)  # should not raise
    assert torch.equal(model.linear.weight, before)


def test_noop_when_single_rank_group():
    """Single-rank process group has nothing to sync."""
    model = _Tiny()
    before = model.linear.weight.clone()
    with (
        mock.patch("llm.training.core.distributed.dist.is_available", return_value=True),
        mock.patch("llm.training.core.distributed.dist.is_initialized", return_value=True),
        mock.patch("llm.training.core.distributed.dist.get_world_size", return_value=1),
        mock.patch("llm.training.core.distributed.dist.broadcast") as broadcast,
    ):
        broadcast_parameters(model)
    broadcast.assert_not_called()
    assert torch.equal(model.linear.weight, before)


def test_broadcasts_each_param_and_buffer_from_src():
    """On a live multi-rank group every parameter and buffer is sent from src."""
    model = _Tiny()
    with (
        mock.patch("llm.training.core.distributed.dist.is_available", return_value=True),
        mock.patch("llm.training.core.distributed.dist.is_initialized", return_value=True),
        mock.patch("llm.training.core.distributed.dist.get_world_size", return_value=2),
        mock.patch("llm.training.core.distributed.dist.broadcast") as broadcast,
    ):
        broadcast_parameters(model, src=0)

    params = list(model.parameters())
    buffers = list(model.buffers())
    assert broadcast.call_count == len(params) + len(buffers)
    # Weight + bias parameters, then the running_mean buffer, all from src=0.
    sent_weights = [call.args[0] for call in broadcast.call_args_list]
    assert torch.equal(sent_weights[0], model.linear.weight)
    assert torch.equal(sent_weights[1], model.linear.bias)
    assert torch.equal(sent_weights[2], model.running_mean)
    for call in broadcast.call_args_list:
        assert call.kwargs["src"] == 0
