"""Tensor parallelism for the LLM training engine (Megatron-style, v1).

``apply_tensor_parallel`` mutates a ``DecoderModel`` **in place**: every
parameter-carrying Linear is replaced by a rank-local slice (a
:class:`ColumnParallelLinear` or :class:`RowParallelLinear`) and the model is
tagged with ``_tp`` metadata so the checkpoint helpers in
:mod:`llm.training.distributed.parallel` can gather/scatter full state dicts.

v1 semantics — replicated data, no data-parallel dimension:

* Every rank runs the **same** microbatches (the engine feeds each rank the
  full batch instead of sharding it), so every rank produces the same loss
  and the same gradient for the replicated parameters (embedding). Column
  and row parallel slices are disjoint per rank, so each rank's optimizer
  step moves only its own shards — all ranks stay in lockstep without any
  extra gradient all-reduce (the collectives below balance the shared
  rows instead).
* Column-parallel linears (``qkv_proj``, MLP ``fc1``/``gate_proj``, and the
  ``lm_head``) partition the OUTPUT (heads / intermediate / vocab) across
  ranks. Their **input gradient** is a per-rank partial sum over the local
  output slice, so it must be all-reduced — a :class:`_BackwardAllReduce`
  wrapper sits on the input path.
* Row-parallel linears (attention ``out_proj``, MLP ``fc2``) partition the
  INPUT (hidden / intermediate); their forward output must be all-reduced —
  a :class:`_ForwardAllReduce` on the result.
* The ``lm_head`` is column-parallel in vocab and its logits are
  all-gathered so cross-entropy sees the full vocabulary on every rank
  (identical loss).
* Norms / RoPE / ALiBi buffers are replicated and untouched.

Checkpoints: ``model_state_dict`` on a TP model gathers every rank's shard
into a **full** plain state dict (mirroring FSDP ``FULL_STATE_DICT``) so the
checkpoint manager and ``llm-serve`` need no changes; ``load_model_state_dict``
slices a full state dict back into each rank's shards in place (default
``assign=False`` keeps parameter object identity, so a live optimizer stays
valid).

Scope guards (rejected loudly, not silently wrong): only the ``mha``
attention backend, no MoE, no ALiBi, and every partitioned axis must divide
evenly by the tensor-parallel size. flash/sdpa/mla attention backends,
paged/cache inference paths and serving are out of scope for v1 (this is a
training strategy).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

from llm.models.decoder import DecoderModel


class _ForwardAllReduce(torch.autograd.Function):
    """Forward: all-reduce ``x`` across the group; backward: pass grad through.

    Used on row-parallel outputs: the full output is the elementwise sum of
    every rank's partial, and the derivative of that sum with respect to each
    partial is the identity.
    """

    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, *grad_outputs: Any):  # type: ignore[override]
        return grad_outputs[0], None


class _BackwardAllReduce(torch.autograd.Function):
    """Forward: identity; backward: all-reduce the incoming gradient.

    Sits on the INPUT of a column-parallel linear. Each rank's backward
    produces ``dL/dx`` summed only over its own output slice; the true
    gradient is the sum over every rank's slice, so the incoming grad is
    all-reduced.
    """

    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, *grad_outputs: Any):  # type: ignore[override]
        grad_output = grad_outputs[0]
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _AllGatherScatter(torch.autograd.Function):
    """Forward: all-gather the last dimension; backward: plain SCATTER.

    Reconstructs the full logits from the per-rank vocab slices. The backward
    is a scatter ONLY (no reduce-scatter/sum): in replicated-data TP every
    rank already computes the identical full ``dL/dlogits``, and each rank's
    contribution to the gathered output is its own ``vocab/tp`` slice — the
    sum-over-ranks that ``torch.distributed.nn.functional.all_gather`` applies
    in its backward would double (x tp) every upstream gradient (RIL TASK-200 /
    DEC-045 — caught by the numeric-parity test).
    """

    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        ctx.group = group
        pieces = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(pieces, x.contiguous(), group=group)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):  # type: ignore[override]
        grad_output = grad_outputs[0]
        rank = dist.get_rank(ctx.group)
        world = dist.get_world_size(ctx.group)
        part = grad_output.shape[-1] // world
        return grad_output[..., rank * part : (rank + 1) * part].contiguous(), None


class ColumnParallelLinear(nn.Module):
    """``nn.Linear`` with the output dimension partitioned across the group.

    ``weight`` holds this rank's row slice ``[out_local, in_full]``. The
    input is replicated; the result is a per-rank slice. The input gradient
    is all-reduced via :class:`_BackwardAllReduce` so upstream replicated
    weights (norms, embedding) see the full gradient.

    Attributes:
        full_shape: the ORIGINAL (pre-partition) weight shape, kept for
            state-dict gather/scatter.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        group: dist.ProcessGroup,
        full_shape: tuple[int, ...] | None = None,
        full_index_list: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        # ``full_shape`` is the ORIGINAL (pre-partition) weight shape — the
        # caller passes it because this object only ever sees the rank slice.
        self.full_shape = tuple(full_shape if full_shape is not None else weight.shape)
        # ``full_index_list`` maps, for EVERY rank, that rank's LOCAL row ->
        # FULL row of the original weight. ``None`` for a plain contiguous
        # partition (local rows are a contiguous block of the full tensor,
        # ordered by rank). A non-None value handles block-interleaved
        # partitions (the fused QKV, where the local slice is
        # [q_rank, k_rank, v_rank] and gathers must scatter every rank's piece
        # back into the q/k/v blocks of the full tensor — RIL TASK-200).
        self.full_index_list = full_index_list
        self.weight = nn.Parameter(weight.detach().clone().contiguous())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone().contiguous())
        else:
            self.register_parameter("bias", None)
        self._group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            x = _BackwardAllReduce.apply(x, self._group)
        return functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """``nn.Linear`` with the input dimension partitioned across the group.

    ``weight`` holds this rank's column slice ``[out_full, in_local]``. The
    input is this rank's contiguous input slice; the partial result is
    all-reduced (:class:`_ForwardAllReduce`) to the full output.

    Attributes:
        full_shape: the ORIGINAL (pre-partition) weight shape.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        group: dist.ProcessGroup,
        full_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()
        # ``full_shape`` is the ORIGINAL (pre-partition) weight shape — the
        # caller passes it because this object only ever sees the rank slice.
        self.full_shape = tuple(full_shape if full_shape is not None else weight.shape)
        self.weight = nn.Parameter(weight.detach().clone().contiguous())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone().contiguous())
        else:
            self.register_parameter("bias", None)
        self._group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All-reduce the UNBIASED partial: the bias belongs to the full
        # output and is added once AFTER the reduce — adding it per-rank
        # would multiply it by the TP size (moot with zero-initialised
        # biases, a real wrong-answer with filled ones).
        partial = functional.linear(x, self.weight, None)
        out = _ForwardAllReduce.apply(partial, self._group)
        if self.bias is not None:
            out = out + self.bias
        return out


class VocabParallelHead(nn.Module):
    """Column-parallel ``lm_head`` plus an all-gather of the logits.

    Each rank computes its ``vocab/tp`` slice of the logits; the full logits
    are reconstructed with an autograd-aware ``all_gather`` (backward
    scatters the gradient back to the local slice) so cross-entropy on the
    full vocabulary is identical on every rank.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        group: dist.ProcessGroup,
        full_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.vocab_partition = weight.shape[0]
        # ORIGINAL (pre-partition) head shape; the caller passes it because
        # this object only ever sees the vocab/tp slice.
        self.full_shape = tuple(full_shape if full_shape is not None else weight.shape)
        self.weight = nn.Parameter(weight.detach().clone().contiguous())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone().contiguous())
        else:
            self.register_parameter("bias", None)
        self._group = group

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Column-parallel head over vocab — the ONLY exception to "no reduce":
        # there is no column-parallel layer between here and the replicated
        # final_norm, so without an input-grad all-reduce every rank would
        # backprop only its own vocab-slice contribution into the shared norm
        # weights, giving a /tp-scaled (and wrong) gradient (RIL TASK-200).
        if hidden_states.requires_grad:
            hidden_states = _BackwardAllReduce.apply(hidden_states, self._group)
        local = functional.linear(hidden_states, self.weight, self.bias)
        logits = _AllGatherScatter.apply(local.contiguous(), self._group)
        if logits.shape[-1] != self.full_shape[0]:
            # All-gather concatenates in rank order — validated at apply time,
            # but the order dependence is subtle enough to assert loudly.
            raise RuntimeError(
                f"VocabParallelHead built logits of width {logits.shape[-1]}, expected full vocab {self.full_shape[0]}"
            )
        return logits


class _TPState:
    """Mutation metadata attached to a TP model (``model._tp``).

    Not an ``nn.Module``/buffer — it never appears in ``state_dict()``. It
    carries the process group, this rank's index, and the partition map used
    by the state-dict gather/scatter helpers.

    ``partition[key]`` is the axis of the FULL tensor along which rank slices
    concatenate (``0`` for column-partitioned linears: rows/heads/intermediate/
    vocab and their column-sliced biases; ``1`` for row-partitioned linears:
    only the weight's input axis is split), or ``None`` when every rank holds
    a full replicated copy (norms, rope tables, embedding, row-partitioned
    biases).
    """

    def __init__(self, model: nn.Module, group: dist.ProcessGroup, world_size: int, rank: int):
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.partition: dict[str, int | None] = {}
        # key -> per-rank local-row -> full-row index for block-interleaved
        # partitions (only the fused QKV column linears today), a list of
        # ``world_size`` tensors; absent = plain slice.
        self.full_index: dict[str, list[torch.Tensor]] = {}
        for key, module in model.named_modules():
            if isinstance(module, ColumnParallelLinear):
                self.partition[key + ".weight"] = 0
                if getattr(module, "bias", None) is not None:
                    self.partition[key + ".bias"] = 0
                if module.full_index_list is not None:
                    self.full_index[key + ".weight"] = module.full_index_list
                    if getattr(module, "bias", None) is not None:
                        self.full_index[key + ".bias"] = module.full_index_list
            elif isinstance(module, VocabParallelHead):
                self.partition[key + ".weight"] = 0
                if getattr(module, "bias", None) is not None:
                    self.partition[key + ".bias"] = 0
            elif isinstance(module, RowParallelLinear):
                self.partition[key + ".weight"] = 1
                if getattr(module, "bias", None) is not None:
                    self.partition[key + ".bias"] = None

    # --- checkpoint helpers -------------------------------------------.

    def gather_full_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Rebuild the full (un-sharded) state dict from the per-rank shards.

        Every rank reconstructs the full dict (each rank's shards are
        identical across ranks given replicated data; the ``all_gather`` for
        plain-slice keys keeps the collective ordering uniform across ranks),
        and the checkpoint manager writes rank 0's. Replicated tensors
        (``axis None``) are taken from this rank's local copy.
        """
        local = model.state_dict()
        full: dict[str, torch.Tensor] = {}
        for key, tensor in local.items():
            axis = self.partition.get(key)
            if axis is None:
                full[key] = tensor.detach().clone()
                continue
            idx_all = self.full_index.get(key)
            if idx_all is not None:
                # All-gather EVERY rank's piece, then scatter each piece into
                # the full tensor at THAT rank's row positions. (A single
                # rank's own rows are only a part of the full tensor — unlike
                # replicated data, per-rank shard VALUES differ.)
                pieces = [torch.empty_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(pieces, tensor.contiguous(), group=self.group)
                full_shape = list(tensor.shape)
                full_shape[axis] = max(int(idx.max().item()) for idx in idx_all) + 1
                full_t = torch.zeros(full_shape, dtype=tensor.dtype, device=tensor.device)
                for r, piece in enumerate(pieces):
                    full_t.index_copy_(axis, idx_all[r].to(tensor.device), piece)
                full[key] = full_t
                continue
            pieces = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(pieces, tensor.contiguous(), group=self.group)
            full[key] = torch.cat(pieces, dim=axis)
        return full

    def scatter_load_state_dict(self, model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
        """Slice a full state dict into this rank's shards and load them.

        Loads with the default ``assign=False`` (in-place copy), so parameter
        object identity is preserved and a live optimizer/scheduler remains
        valid across a resume.
        """
        sharded: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            axis = self.partition.get(key)
            if axis is None:
                sharded[key] = tensor.detach().clone()
                continue
            idx_all = self.full_index.get(key)
            if idx_all is not None:
                sharded[key] = (
                    tensor.index_select(axis, idx_all[self.rank].to(tensor.device)).detach().clone().contiguous()
                )
                continue
            part = tensor.shape[axis] // self.world_size
            start = self.rank * part
            sl = [slice(None)] * tensor.ndim
            sl[axis] = slice(start, start + part)
            sharded[key] = tensor[tuple(sl)].detach().clone().contiguous()
        model.load_state_dict(sharded)


def _even(n: int, tp_size: int, what: str) -> None:
    if n % tp_size != 0:
        raise ValueError(
            f"Tensor parallelism requires {what} ({n}) to divide evenly by tp_size={tp_size}; "
            "adjust model dims or tp_size."
        )


def _replace(module: nn.Module, attr: str, new: nn.Module) -> None:
    setattr(module, attr, new)


def apply_tensor_parallel(
    model: DecoderModel,
    *,
    process_group: dist.ProcessGroup | None = None,
) -> DecoderModel:
    """Partition ``model`` for tensor-parallel training, in place.

    Args:
        model: A ``DecoderModel`` (may be wrapped in DDP/compile — only the
            bare decoder underneath is transformed). Must use the ``mha``
            attention backend with the standard MLP (MoE rejected).
        process_group: The TP group. Defaults to the default ``WORLD`` group
            (the whole world is one TP group in v1).

    Returns:
        The same ``model`` object, mutated in place (identity preserved so
        downstream ``nn.Module``-traversal code and the training loop are
        unaffected), tagged with ``.tp`` metadata.

        The model is NOT wrapped — this is a pure in-place rewiring. Use
        ``parallel_strategy="tp"`` via :func:`wrap_model_for_training`.

    Raises:
        ValueError / NotImplementedError: for unsupported configurations or
            axes that do not divide evenly by the TP size.
    """
    group: Any = process_group if process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0

    if world_size <= 1:
        # Single-rank TP is a no-op identity (matches DDP/FSDP world_size<=1).
        return model

    if not isinstance(model, DecoderModel):
        raise TypeError(
            f"apply_tensor_parallel expects a llm.models.DecoderModel, got {type(model).__name__}. "
            "Tensor parallelism v1 supports the DecoderModel architecture only."
        )

    # --- Scope guards ------------------------------------------------.
    tp_size = world_size
    # The registry returns attention/MLP as generic nn.Module; cast the whole
    # decoder to Any for the transform so attribute access stays unchecked (the
    # architecture is validated by the isinstance guard above and the scope
    # checks below).
    m: Any = model
    if getattr(m, "use_alibi", False):
        raise NotImplementedError("Tensor parallelism v1 does not support ALiBi (use_alibi=True).")
    for i, block in enumerate(m.transformer_blocks):
        attn_cls = type(block.self_attn).__name__
        if attn_cls != "MultiHeadAttention":
            raise NotImplementedError(
                f"Tensor parallelism v1 supports only attn_impl='mha' (block {i} is {attn_cls}); "
                "flash/sdpa/mla attention backends are out of scope for the TP milestone."
            )
        if getattr(block.mlp, "num_experts", None):
            raise NotImplementedError("Tensor parallelism with MoE (expert parallelism) is not implemented in v1.")

    # --- Partition axes must divide evenly ----------------------------.
    num_heads = m.num_heads
    num_kv_heads = getattr(m.transformer_blocks[0].self_attn, "num_kv_heads", num_heads)
    _even(num_heads, tp_size, "num_heads")
    _even(num_kv_heads, tp_size, "num_kv_heads")
    _even(m.lm_head.weight.shape[0], tp_size, "vocab_size")
    for block in m.transformer_blocks:
        _even(block.self_attn.qkv_proj.weight.shape[0], tp_size, "the fused QKV projection")
        _even(block.mlp.fc1.weight.shape[0], tp_size, "the MLP intermediate width")
        if getattr(block.mlp, "gate_proj", None) is not None:
            _even(block.mlp.gate_proj.weight.shape[0], tp_size, "the MLP intermediate width")

    # --- 1. Attention: head-partition QKV, row-partition output proj ---.
    for block in m.transformer_blocks:
        attn = block.self_attn
        qkv = attn.qkv_proj
        out = attn.out_proj
        qkv_w = qkv.weight.detach()
        qkv_b = qkv.bias.detach() if qkv.bias is not None else None
        out_w = out.weight.detach()
        out_b = out.bias.detach() if out.bias is not None else None
        # The fused QKV projection lays its rows out as [q, k, v] blocks, so a
        # plain contiguous column slice would hand each rank a mix of OTHER
        # ranks' q / k / v rows — the local model then splits its slice as
        # [q_local, kv_local, kv_local] and reads garbage (query rows as K/V).
        # Slice each of q, k, v SEPARATELY and reassemble as
        # [q_rank, k_rank, v_rank] so the local split stays correct (RIL
        # TASK-200 / DEC-045 — this was a real wrong-answer bug, caught by the
        # TP-vs-reference numeric-parity test).
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        q_rows = q_size // tp_size
        kv_rows = kv_size // tp_size
        q_slice = qkv_w[rank * q_rows : (rank + 1) * q_rows]
        k_slice = qkv_w[q_size + rank * kv_rows : q_size + (rank + 1) * kv_rows]
        v_slice = qkv_w[q_size + kv_size + rank * kv_rows : q_size + kv_size + (rank + 1) * kv_rows]
        qkv_slice = torch.cat([q_slice, k_slice, v_slice], dim=0)
        full_index_list = [
            torch.cat(
                [
                    torch.arange(r * q_rows, (r + 1) * q_rows),
                    torch.arange(q_size + r * kv_rows, q_size + (r + 1) * kv_rows),
                    torch.arange(q_size + kv_size + r * kv_rows, q_size + kv_size + (r + 1) * kv_rows),
                ]
            )
            for r in range(tp_size)
        ]
        if qkv_b is not None:
            q_b = qkv_b[rank * q_rows : (rank + 1) * q_rows]
            k_b = qkv_b[q_size + rank * kv_rows : q_size + (rank + 1) * kv_rows]
            v_b = qkv_b[q_size + kv_size + rank * kv_rows : q_size + kv_size + (rank + 1) * kv_rows]
            qkv_b_slice = torch.cat([q_b, k_b, v_b])
        else:
            qkv_b_slice = None
        _replace(
            attn,
            "qkv_proj",
            ColumnParallelLinear(
                qkv_slice, qkv_b_slice, group=group, full_shape=tuple(qkv_w.shape), full_index_list=full_index_list
            ),
        )
        # out_proj: [hidden, hidden] row-partitioned over its INPUT (the
        # per-rank head output slice). Keep the full-output bias.
        out_slice = out_w[:, rank * (out_w.shape[1] // tp_size) : (rank + 1) * (out_w.shape[1] // tp_size)]
        _replace(attn, "out_proj", RowParallelLinear(out_slice, out_b, group=group, full_shape=tuple(out_w.shape)))
        # Patch the per-rank attention geometry so the local reshape and
        # q/k/v splits agree with the sliced qkv projection.
        attn.num_heads = num_heads // tp_size
        attn.num_kv_heads = num_kv_heads // tp_size
        attn.hidden_size = attn.num_heads * attn.head_dim
        attn.kv_dim = attn.num_kv_heads * attn.head_dim

    # --- 2. MLP: column gate/up, row down ------------------------------.
    for block in m.transformer_blocks:
        mlp = block.mlp
        for name in ("fc1", "gate_proj"):
            lin = getattr(mlp, name, None)
            if lin is None:
                continue
            w = lin.weight.detach()
            b = lin.bias.detach() if lin.bias is not None else None
            part = w.shape[0] // tp_size
            w_slice = w[rank * part : (rank + 1) * part]
            b_slice = b[rank * part : (rank + 1) * part] if b is not None else None
            _replace(mlp, name, ColumnParallelLinear(w_slice, b_slice, group=group, full_shape=tuple(w.shape)))
        fc2 = mlp.fc2
        w2 = fc2.weight.detach()
        b2 = fc2.bias.detach() if fc2.bias is not None else None
        part = w2.shape[1] // tp_size
        w2_slice = w2[:, rank * part : (rank + 1) * part]
        _replace(mlp, "fc2", RowParallelLinear(w2_slice, b2, group=group, full_shape=tuple(w2.shape)))

    # --- 3. lm_head: column-parallel vocab + logits all-gather ---------.
    head = m.lm_head
    w = head.weight.detach()
    b = head.bias.detach() if head.bias is not None else None
    part = w.shape[0] // tp_size
    w_slice = w[rank * part : (rank + 1) * part]
    b_slice = b[rank * part : (rank + 1) * part] if b is not None else None
    _replace(m, "lm_head", VocabParallelHead(w_slice, b_slice, group=group, full_shape=tuple(w.shape)))

    # --- 4. Tag the model so checkpoint helpers are TP-aware -----------.
    m._tp = _TPState(m, group, world_size, rank)  # type: ignore[attr-defined]
    return model
