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

Scope guards (rejected loudly, not silently wrong): the ``mha``, ``flash_attn``
and ``mla`` attention backends and MoE expert parallelism are supported; ALiBi
is rejected, and every partitioned axis must divide evenly by the
tensor-parallel size. ``sdpa`` is a functional (``core/attn/sdpa.py``) used by
every supported backend, not a registered ``attn_impl`` — TP covers it
transitively. Paged/cache inference paths and serving are out of scope for v1
(this is a training strategy).
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


class _ExpertParallelMoE(nn.Module):
    """MoE layer under tensor parallelism: expert parallelism (TASK-207).

    Replaces ``block.mlp`` (a ``MoE``) in place. The GATE stays a full
    replicated ``nn.Linear`` (every rank computes identical routing — the
    gate is tiny, so duplicating it beats sharding); the EXPERTS are split
    across ranks by expert index: rank ``r`` owns ``experts[r*n_local :
    (r+1)*n_local]`` (each expert is held ENTIRELY on its owner — the expert
    dimension, not the MLP hidden dims, is the shard axis, so the local
    expert linears stay plain ``nn.Linear``).

    Forward restructures the dense MoE (``out[i] = sum_k w[i,k] * f_{e_k}(x[i])``)
    for sharded experts:

    * both the GATE and the EXPERTS run on ``_BackwardAllReduce(x)`` (their
      per-rank gradients are PARTIAL — each rank only holds a subset of the
      experts, so it can only backprop those experts' contributions through
      both the expert outputs AND the gate weight-marginal path; the input
      all-reduce sums them into the full gradient for the attention);
    * the per-rank partial output (only tokens routed to a local expert
      contribute) is all-reduced — ``_ForwardAllReduce`` — completing the
      ``sum_k`` over every rank's experts (matches the dense reference modulo
      fp summation order).

    Keeps ``num_experts`` = TOTAL expert count (the full gate output dim) so
    attribute-based MoE detection (DDP ``find_unused_parameters``,
    ``parallel.py:uses_moe``) and checkpointing see a normal MoE block.
    """

    def __init__(
        self,
        gate: nn.Module,
        local_experts: list[nn.Module],
        *,
        group: dist.ProcessGroup,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        rank: int,
        n_local: int,
    ):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(local_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self._group = group
        self._expert_offset = rank * n_local  # this rank's first global expert
        self._n_local = n_local

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.hidden_size)  # [N, hidden]

        # 1. Input gradient all-reduce. ``x`` is the replicated attention
        # output; both the gate and the expert outputs backprop only into
        # THIS rank's expert subset, so their partial gradients must be
        # summed across ranks (column-parallel input rule) — see docstring.
        x_ep = x if not x.requires_grad else _BackwardAllReduce.apply(x, self._group)

        # 2. Full gate (replicated): identical routing on every rank.
        gate_logits = self.gate(x_ep)  # [N, num_experts]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = torch.softmax(top_k_logits, dim=-1, dtype=x.dtype)

        # 3. Only tokens routed to a local expert contribute on this rank.
        lo, hi = self._expert_offset, self._expert_offset + self._n_local
        partial = torch.zeros_like(x)
        # Zero-contributing term that ALWAYS connects ``x_ep`` to the output:
        # on a rank whose local experts receive no routed tokens (collapsed
        # routing / small microbatches are the norm for MoE, ~every rank dead
        # some step), the ``index_add_`` below never runs and ``x_ep`` would
        # otherwise drop off the autograd graph — then THIS rank would skip
        # ``_BackwardAllReduce.backward``'s all-reduce while a peer that got
        # hits enters it, deadlocking the backward. ``* 0.0`` keeps the value
        # (and this rank's expert gradient) exactly zero.
        partial = partial + x_ep * 0.0
        hit = (top_k_indices >= lo) & (top_k_indices < hi)  # [N, k]
        if hit.any():
            rows, cols = hit.nonzero(as_tuple=True)
            local_idx = top_k_indices[rows, cols] - lo
            weights = expert_weights[rows, cols].unsqueeze(-1)
            for j in range(self._n_local):
                sel = local_idx == j
                if not sel.any():
                    continue
                tokens = rows[sel]
                partial.index_add_(0, tokens, self.experts[j](x_ep[tokens]) * weights[sel])

        # 4. Complete the expert sum across ranks and restore the shape.
        return _ForwardAllReduce.apply(partial, self._group).view(original_shape)


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
        # TP + data-parallel 2D (TASK-202): the strided group of ranks holding
        # the SAME shard across TP groups, averaged at step boundaries.
        # ``None`` in pure TP v1 (whole world = one TP group, no DP dimension).
        self.dp_group: dist.ProcessGroup | None = None
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
        # Expert-parallel MoE blocks (TASK-207): ``prefix`` -> (total_experts,
        # n_local), where ``prefix`` is the MoE module's own name (e.g.
        # "transformer_blocks.0.mlp"), so its state-dict keys live under
        # ``prefix.experts.{li}.**``. Each rank owns experts
        # ``[rank*n_local, (rank+1)*n_local)``; the local ModuleList keys map
        # to global expert ``rank*n_local + li``. The MoE gate is a replicated
        # ``nn.Linear`` — deliberately absent from ``partition`` so it is
        # treated as a replicated param by gather/scatter (full copy) and
        # averaged across the TP group by ``allreduce_dp_grads`` (identical on
        # every rank, so AVG keeps the replicas in lockstep without changing
        # the value).
        self.expert_shards: dict[str, tuple[int, int]] = {}
        # Replicated MoE routers: their WEIGHT gradient is a per-rank PARTIAL
        # (each rank backprops only its own experts' contributions through the
        # routing marginals), so ``allreduce_dp_grads`` must SUM it over the
        # TP group — unlike other replicated params (norms, embedding) whose
        # gradients are already complete via the input all-reduces and only
        # get averaged to keep replicas in lockstep.
        self.gate_params: set[str] = set()
        for name, module in model.named_modules():
            if isinstance(module, _ExpertParallelMoE):
                self.expert_shards[name] = (module.num_experts, module._n_local)
                self.gate_params.add(name + ".gate.weight")
                if getattr(module.gate, "bias", None) is not None:
                    self.gate_params.add(name + ".gate.bias")

    def is_expert_param(self, key: str) -> bool:
        """True when ``key`` is an expert-sharded parameter (one rank only).

        Expert params straddle the partition rule: they are sharded (a given
        expert lives on exactly one rank), but not by the usual axis — so
        ``partition`` cannot describe them. Used by ``allreduce_dp_grads`` to
        skip the TP-group average (neighbours hold DIFFERENT experts) while
        still reducing over the DP group.
        """
        return any(key.startswith(prefix + ".experts.") for prefix in self.expert_shards)

    def is_gate_param(self, key: str) -> bool:
        """True when ``key`` is a replicated MoE router (gate) parameter.

        Unlike other replicated params, the gate does not get its full
        gradient from the input all-reduces — its weight gradient is a
        per-rank partial that ``allreduce_dp_grads`` must SUM over the TP
        group (see ``gate_params`` above).
        """
        return key in self.gate_params

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
            if self.is_expert_param(key):
                continue  # local-indexed expert keys — rebuilt below
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
        # Expert-parallel shards: every rank holds a DIFFERENT subset (its own
        # n_local experts), so the full tensor for global expert ``g`` comes
        # from owner rank ``g // n_local`` (local copy ``g % n_local``). The
        # all-gather is rank-major, so rebuilding ``experts.{g}`` in ascending
        # order yields a full dict identical on every rank.
        self._gather_expert_shards(full, local)
        return full

    def _gather_expert_shards(self, full: dict[str, torch.Tensor], local: dict[str, torch.Tensor]) -> None:
        for prefix, (total, n_local) in self.expert_shards.items():
            marker = prefix + ".experts."
            by_suffix: dict[str, dict[int, torch.Tensor]] = {}
            for key, tensor in local.items():
                if not key.startswith(marker):
                    continue
                li_str, suffix = key[len(marker) :].split(".", 1)
                by_suffix.setdefault(suffix, {})[int(li_str)] = tensor
            for suffix, tensors_by_li in by_suffix.items():
                tensors = [tensors_by_li[li] for li in range(n_local)]
                block = tensors[0].unsqueeze(0) if n_local == 1 else torch.stack(tensors)
                pieces = [torch.empty_like(block) for _ in range(self.world_size)]
                dist.all_gather(pieces, block.contiguous(), group=self.group)
                for g in range(total):
                    owner, li = divmod(g, n_local)
                    full[f"{prefix}.experts.{g}.{suffix}"] = pieces[owner][li].detach().clone()

    def scatter_load_state_dict(self, model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
        """Slice a full state dict into this rank's shards and load them.

        Loads with the default ``assign=False`` (in-place copy), so parameter
        object identity is preserved and a live optimizer/scheduler remains
        valid across a resume.
        """
        sharded: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            match = _expert_prefix_of(self.expert_shards, key)
            if match is not None:
                # Global expert ``g`` lives on owner ``g // n_local``; only the
                # owning rank keeps it, renumbered to its local index.
                prefix, rest = match
                _total, n_local = self.expert_shards[prefix]
                g = int(rest.split(".", 1)[0])
                owner, li = divmod(g, n_local)
                if owner == self.rank:
                    suffix = rest.split(".", 1)[1]
                    sharded[f"{prefix}.experts.{li}.{suffix}"] = tensor.detach().clone()
                continue
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


def _expert_prefix_of(expert_shards: dict[str, tuple[int, int]], key: str) -> tuple[str, str] | None:
    """Return ``(prefix, rest)`` when ``key`` lives under an expert shard.

    ``rest`` is everything after ``prefix + ".experts."`` (e.g. ``"3.fc1.weight"``
    for global expert 3); ``None`` when the key is not expert-sharded.
    """
    for prefix in expert_shards:
        marker = prefix + ".experts."
        if key.startswith(marker):
            return prefix, key[len(marker) :]
    return None


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
            bare decoder underneath is transformed). Supported backends: the
            ``mha`` / ``flash_attn`` / ``mla`` attention and MoE expert
            parallelism (every partitioned axis must divide evenly by the TP
            size; ALiBi is rejected).
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
        # v1 supports backends that share the MHA projection surface: fused
        # QKV (column-parallel over heads) + row-parallel out_proj with
        # per-rank head geometry patchable via num_heads/num_kv_heads/head_dim/
        # kv_dim. FlashAttention declares exactly that surface (RIL ISS-137),
        # so the transform below applies unchanged; the kv-cache / paged
        # decoding differences live in forward, not in the weights.
        # MultiLatentAttention (TASK-206) uses a different parameter layout
        # (latent_query/value/output projections + [K,V]-block input_kv_proj),
        # dispatched to a dedicated branch — see ``_mha_flash_attn_slice`` /
        # ``_mla_slice``. sdpa is a FUNCTIONAL (core/attn/sdpa.py), not a
        # registered attn_impl: every backend above runs its attention through
        # it, so TP covers the sdpa kernel transitively (TASK-209). MoE blocks
        # get expert parallelism (TASK-207): the gate is replicated and the
        # experts are split across ranks by expert index (guarded below).
        if attn_cls not in ("MultiHeadAttention", "FlashAttention", "MultiLatentAttention"):
            raise NotImplementedError(
                f"Tensor parallelism v1 supports attn_impl in {{'mha', 'flash_attn', 'mla'}} "
                f"(block {i} is {attn_cls}); this attention backend is out of scope for the TP milestone."
            )
        # ``num_experts`` is the total expert count (the full gate output dim).
        # Expert parallelism requires it to split evenly across ranks — the
        # actual slicing happens in the MLP section below.
        if getattr(block.mlp, "num_experts", None):
            _even(block.mlp.num_experts, tp_size, "num_experts")

    # --- Partition axes must divide evenly ----------------------------.
    num_heads = m.num_heads
    num_kv_heads = getattr(m.transformer_blocks[0].self_attn, "num_kv_heads", num_heads)
    _even(num_heads, tp_size, "num_heads")
    _even(num_kv_heads, tp_size, "num_kv_heads")
    _even(m.lm_head.weight.shape[0], tp_size, "vocab_size")
    for block in m.transformer_blocks:
        attn_cls = type(block.self_attn).__name__
        if attn_cls == "MultiLatentAttention":
            # MLA K/V blocks: input_kv_proj output is [K | V] each num_heads
            # heads; head-slicing requires each block divisible (same
            # constraint the fused QKV enforces per q/k/v fragment).
            _even(block.self_attn.input_kv_proj.weight.shape[0] // 2, tp_size, "the MLA K/V head blocks")
        else:
            _even(block.self_attn.qkv_proj.weight.shape[0], tp_size, "the fused QKV projection")
        if getattr(block.mlp, "num_experts", None):
            # MoE: the expert dimension (not an MLP hidden width) is the shard
            # axis — already checked for even division in the scope loop.
            continue
        _even(block.mlp.fc1.weight.shape[0], tp_size, "the MLP intermediate width")
        if getattr(block.mlp, "gate_proj", None) is not None:
            _even(block.mlp.gate_proj.weight.shape[0], tp_size, "the MLP intermediate width")

    # --- 1. Attention: head-partition projections, row-partition output ---.
    for block in m.transformer_blocks:
        attn = block.self_attn
        if type(attn).__name__ == "MultiLatentAttention":
            # MLA layout (TASK-206): the head axis threads through four
            # projections, each sliced per-head:
            #   * latent_q_proj [hidden, latent_dim] and latent_output_proj
            #     [hidden, latent_dim] are column-parallel over their OUTPUT
            #     hidden (= num_heads * head_dim, heads contiguous — a plain
            #     row slice);
            #   * input_kv_proj [2*hidden, hidden] is column-parallel over its
            #     output split into [K | V] blocks (each num_heads heads) —
            #     same block-interleave trap as the fused QKV, so K/V must be
            #     sliced separately and reassembled [k_rank, v_rank] with a
            #     block-aware full_index;
            #   * latent_v_proj [latent_dim, hidden] and out_proj [hidden,
            #     hidden] are row-parallel over their INPUT hidden (each rank
            #     holds its own head slice of the hidden axis).
            # ``latents`` [1, num_latents, latent_dim] and ``latent_dim`` are
            # REPLICATED (latent_dim is a latent axis, not a head axis).
            #
            # Q projection: column over output hidden (heads contiguous).
            lq_w = attn.latent_q_proj.weight.detach()
            lq_b = attn.latent_q_proj.bias.detach() if attn.latent_q_proj.bias is not None else None
            part = lq_w.shape[0] // tp_size
            lq_slice = lq_w[rank * part : (rank + 1) * part]
            lq_b_slice = lq_b[rank * part : (rank + 1) * part] if lq_b is not None else None
            _replace(
                attn,
                "latent_q_proj",
                ColumnParallelLinear(lq_slice, lq_b_slice, group=group, full_shape=tuple(lq_w.shape)),
            )
            # K/V projection: column over 2*hidden, block-interleaved [K | V].
            kv_w = attn.input_kv_proj.weight.detach()
            kv_b = attn.input_kv_proj.bias.detach() if attn.input_kv_proj.bias is not None else None
            kv_block = kv_w.shape[0] // 2  # each of K and V holds ``num_heads`` heads
            kv_rows = kv_block // tp_size
            k_slice = kv_w[rank * kv_rows : (rank + 1) * kv_rows]
            v_slice = kv_w[kv_block + rank * kv_rows : kv_block + (rank + 1) * kv_rows]
            kv_slice = torch.cat([k_slice, v_slice], dim=0)
            kv_full_index = [
                torch.cat(
                    [
                        torch.arange(r * kv_rows, (r + 1) * kv_rows),
                        torch.arange(kv_block + r * kv_rows, kv_block + (r + 1) * kv_rows),
                    ]
                )
                for r in range(tp_size)
            ]
            if kv_b is not None:
                k_b = kv_b[rank * kv_rows : (rank + 1) * kv_rows]
                v_b = kv_b[kv_block + rank * kv_rows : kv_block + (rank + 1) * kv_rows]
                kv_b_slice = torch.cat([k_b, v_b])
            else:
                kv_b_slice = None
            _replace(
                attn,
                "input_kv_proj",
                ColumnParallelLinear(
                    kv_slice,
                    kv_b_slice,
                    group=group,
                    full_shape=tuple(kv_w.shape),
                    full_index_list=kv_full_index,
                ),
            )
            # Latent-dim projection: row over input hidden (per-rank head slice
            # ~> full latent_dim). Keep the full-output bias.
            lv_w = attn.latent_v_proj.weight.detach()
            lv_b = attn.latent_v_proj.bias.detach() if attn.latent_v_proj.bias is not None else None
            lv_slice = lv_w[:, rank * (lv_w.shape[1] // tp_size) : (rank + 1) * (lv_w.shape[1] // tp_size)]
            _replace(
                attn,
                "latent_v_proj",
                RowParallelLinear(lv_slice, lv_b, group=group, full_shape=tuple(lv_w.shape)),
            )
            # Output projection: column over output hidden back into heads.
            lo_w = attn.latent_output_proj.weight.detach()
            lo_b = attn.latent_output_proj.bias.detach() if attn.latent_output_proj.bias is not None else None
            lo_part = lo_w.shape[0] // tp_size
            lo_slice = lo_w[rank * lo_part : (rank + 1) * lo_part]
            lo_b_slice = lo_b[rank * lo_part : (rank + 1) * lo_part] if lo_b is not None else None
            _replace(
                attn,
                "latent_output_proj",
                ColumnParallelLinear(lo_slice, lo_b_slice, group=group, full_shape=tuple(lo_w.shape)),
            )
            # Final out_proj: row over input hidden. Keep the full-output bias.
            out_w = attn.out_proj.weight.detach()
            out_b = attn.out_proj.bias.detach() if attn.out_proj.bias is not None else None
            out_slice = out_w[:, rank * (out_w.shape[1] // tp_size) : (rank + 1) * (out_w.shape[1] // tp_size)]
            _replace(attn, "out_proj", RowParallelLinear(out_slice, out_b, group=group, full_shape=tuple(out_w.shape)))
            # Patch the per-rank geometry so the local views/splits agree with
            # the sliced projections (MLA has no GQA: num_kv_heads == num_heads,
            # and no ``kv_dim`` attribute to patch).
            attn.num_heads = num_heads // tp_size
            attn.num_kv_heads = num_kv_heads // tp_size
            attn.hidden_size = attn.num_heads * attn.head_dim
            continue
        # --- Fused-QKV backends (mha / flash_attn) ---.
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
        if getattr(mlp, "num_experts", None):
            # MoE expert parallelism (TASK-207): the gate is replicated (full
            # ``nn.Linear`` — identical routing on every rank, averaged across
            # the TP group like the norms), and only this rank's slice of
            # experts survives. Each local expert is a complete MLP (full
            # fc1/fc2) — the EXPERT index is the shard axis, so no intra-expert
            # slicing. ``_ExpertParallelMoE`` keeps ``num_experts`` = total to
            # stay attribute-compatible (DDP MoE detection, checkpoints).
            n_local = mlp.num_experts // tp_size
            local_experts = [mlp.experts[rank * n_local + i] for i in range(n_local)]
            _replace(
                block,
                "mlp",
                _ExpertParallelMoE(
                    mlp.gate,
                    local_experts,
                    group=group,
                    num_experts=mlp.num_experts,
                    top_k=mlp.top_k,
                    hidden_size=mlp.hidden_size,
                    rank=rank,
                    n_local=n_local,
                ),
            )
            continue
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
