"""PP + TP co-parallel runtime (RIL TASK-216 / DEC-052, DEC-053).

3D parallel, data-parallel size 1 (pure pipeline + tensor-parallel within each
stage): every rank owns one pipeline stage whose transformer blocks are
tensor-parallelised over the stage's TP group.

torch's ``ScheduleGPipe`` cannot drive this layout: ``PipelineStage`` forbids
``group_size > num_stages`` and derives P2P peers from ``stage_index % group_size``
(DEC-053). So this module provides a small, correct GPipe-style driver (one
microbatch, no overlap) that routes the activation / input-gradient P2P
between pipeline stages rank-to-rank at a fixed ``tp_rank`` within a column.

Layout (dp_size == 1): ``rank = pp_rank * tp_size + tp_rank``, so within a
column the same ``tp_rank`` of stage ``i`` and stage ``i+1`` are ``tp_size``
ranks apart - the P2P partner of rank ``r`` is ``r - tp_size`` (previous stage)
and ``r + tp_size`` (next stage).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn

from llm.training.distributed.pipeline import lm_shift_loss


class PPTPRuntime:
    """GPipe-style PP+TP pipeline driver for data-parallel size 1.

    Each rank holds exactly one tensor-parallelised stage module. ``step`` runs a
    forward pass routed stage-to-stage, computes the LM (shifted) loss on the last
    stage, back-propagates through it, and routes the input gradient back
    stage-to-stage so every stage accumulates its own parameter gradients.
    ``eval`` runs the forward only.

    Args:
        rank: Global rank.
        pp_size: Number of pipeline stages.
        tp_size: Tensor-parallel size (ranks per stage).
        stage: This rank's TP-sharded ``_PipelineStage``.
        device: Device the stage runs on.
    """

    def __init__(self, *, rank: int, pp_size: int, tp_size: int, stage: nn.Module, device: torch.device) -> None:
        if pp_size < 2:
            raise ValueError("PP+TP needs pp_size >= 2")
        if tp_size < 1:
            raise ValueError("PP+TP needs tp_size >= 1")
        self.rank = rank
        self.pp_rank = rank // tp_size
        self.tp_rank = rank % tp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.stage = stage
        self.device = device
        self.group = dist.group.WORLD
        self.prev_rank = rank - tp_size if self.pp_rank > 0 else None
        self.next_rank = rank + tp_size if self.pp_rank < pp_size - 1 else None
        self._fwd_in: torch.Tensor | None = None
        self._fwd_out: torch.Tensor | None = None
        self._loss: torch.Tensor | None = None

    def _recv(self, src: int, *, require_grad: bool) -> torch.Tensor:
        buf: list[torch.Tensor | None] = [None]
        dist.recv_object_list(buf, src=src, group=self.group)
        t = buf[0]
        if t is None:
            raise RuntimeError(f"PP+TP rank {self.rank}: received None from stage P2P (src {src})")
        t = t.detach()
        if require_grad:
            t = t.requires_grad_(True)
        return t.to(self.device)

    def _send(self, t: torch.Tensor, dst: int) -> None:
        dist.send_object_list([t.detach()], dst=dst, group=self.group)

    def step(self, input_ids, *, target, criterion, scale=1.0, losses=None):
        if self.pp_rank == 0:
            h = self.stage(input_ids)
            self._fwd_out = h
            nxt = self.next_rank
            if nxt is not None:
                self._send(h, nxt)
        else:
            prv = cast(int, self.prev_rank)
            h = self._recv(prv, require_grad=True)
            self._fwd_in = h
            out = self.stage(h)
            self._fwd_out = out
            if self.pp_rank == self.pp_size - 1:
                loss = lm_shift_loss(out, target, criterion) / scale
                self._loss = loss
                if losses is not None:
                    losses.append(loss.detach())
            else:
                nxt = cast(int, self.next_rank)
                self._send(out, nxt)

        if self.pp_rank == self.pp_size - 1:
            if self._loss is None:
                raise RuntimeError("PP+TP: last stage stepped without a loss")
            self._loss.backward()
            fin: torch.Tensor = cast(torch.Tensor, self._fwd_in)
            g = cast(torch.Tensor, fin.grad)
            prv = self.prev_rank
            if prv is not None:
                self._send(g, prv)
        elif self.pp_rank == 0:
            nxt = cast(int, self.next_rank)
            ig = self._recv(nxt, require_grad=False)
            self._fwd_out = cast(torch.Tensor, self._fwd_out)
            self._fwd_out.backward(ig)
        else:
            nxt = cast(int, self.next_rank)
            ig = self._recv(nxt, require_grad=False)
            self._fwd_out = cast(torch.Tensor, self._fwd_out)
            self._fwd_out.backward(ig)
            self._fwd_in = cast(torch.Tensor, self._fwd_in)
            prv = cast(int, self.prev_rank)
            self._send(cast(torch.Tensor, self._fwd_in.grad), prv)
        return self._loss

    def eval(self, input_ids, *, target, criterion, evals=None):
        with torch.no_grad():
            if self.pp_rank == 0:
                h = self.stage(input_ids)
                nxt = self.next_rank
                if nxt is not None:
                    self._send(h, nxt)
            else:
                h = self._recv(cast(int, self.prev_rank), require_grad=False)
                out = self.stage(h)
                if self.pp_rank == self.pp_size - 1:
                    loss = lm_shift_loss(out, target, criterion)
                    if evals is not None:
                        evals.append(loss.detach())
                    return loss
                self._send(out, cast(int, self.next_rank))
        return None

    def broadcast_last_loss(self, loss: torch.Tensor) -> torch.Tensor:
        t = loss.detach().to(self.device)
        # The loss lives on the LAST stage's ranks. group=WORLD, so group_src is
        # an actual GLOBAL rank: the first rank of the last stage,
        # (pp_size - 1) * tp_size (its TP peers compute the identical loss).
        src = (self.pp_size - 1) * self.tp_size
        dist.broadcast(t, group_src=src, group=dist.group.WORLD)
        return t
