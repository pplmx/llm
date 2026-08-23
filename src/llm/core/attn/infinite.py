"""Infini-Attention compressive-memory slice (ROADMAP 15.1 / research).

Infini-Attention (Munkhdalai et al., "Leave No Context Behind", 2024) augments
causal dot-product attention with a *compressive memory*: a key-value matrix
``M`` of shape ``[d_k, d_v]`` plus a normalizer ``z`` of shape ``[d_k]`` that
accumulate every past key/value through a non-negative feature map. At each
step a query retrieves a context vector

    A_mem(q) = (phi(q)ᵀ M) / (phi(q)ᵀ z),

which is exactly the linear-attention weighted average of all values written to
memory so far. The local (causal) dot-product head and the memory head are
combined with a learned scalar gate ``beta``:

    o = sigmoid(beta) * A_dot + (1 - sigmoid(beta)) * A_mem.

This module is a self-contained, CPU-verifiable research slice in the same
spirit as the other ``core/attn`` long-context tools (StreamingLLM, Longformer,
BigBird). It implements the memory state machine and retrieval; wiring it into
a live ``attn_impl``/model forward is a documented follow-up (the memory makes
context effectively unbounded, a separate integration concern).

CPU parity invariants verified in ``tests/core/attn/test_infinite.py``:

- a fresh (empty) memory retrieves exactly ``0``, and with the gate pushed to
  the dot-attention side the output equals the dense causal attention over the
  current segment — i.e. Infini-Attention is a *superset* of dense attention;
- memory retrieval equals the explicit linear-attention reference
  ``sum_i phi(q.k_i) v_i / sum_i phi(q.k_i)`` over the accumulated prefix;
- every state machine op is differentiable (backward yields finite grads).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "InfiniMemory",
    "feature_map",
    "infinite_attention",
    "memory_retrieval_attention",
]

#: Default beta gate: sigmoid(beta) blends the dot and memory heads. At 0 the
#: two heads are weighted 50/50; ``beta -> +inf`` uses only dot attention,
#: ``beta -> -inf`` uses only memory.
DEFAULT_BETA = 0.0


def feature_map(x: Tensor) -> Tensor:
    """Non-negative feature map ``elu(x, alpha=1.0) + 1`` (linear-attention).

    ``elu(x) + 1 >= 0`` for all ``x``, which keeps normalizers non-negative and
    avoids the alternating-sign instability of a raw ``1 + x`` kernel when keys
    are projected to ``x < -1`` (standard Infini-Attention choice).
    """
    return torch.nn.functional.elu(x, alpha=1.0) + 1.0


class InfiniMemory:
    """Stateful compressive memory: ``M`` (``[d_k, d_v]``) + normalizer ``z`` (``[d_k]``).

    Accumulates every written key/value through :func:`feature_map`. Each write
    is linear (``outer(phi_k, v)``), so a whole prefix fits in one chunkless
    ``[d_k, d_v]`` state regardless of context length — the unbounded-memory
    property Infini-Attention is designed for.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        self.d_k = d_k
        self.d_v = d_v
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.M = torch.zeros(d_k, d_v, dtype=dtype, device=self.device)
        self.z = torch.zeros(d_k, dtype=dtype, device=self.device)

    def reset(self) -> None:
        self.M.zero_()
        self.z.zero_()

    def update(self, k: Tensor, v: Tensor) -> None:
        """Compress a chunk ``k [T, d_k]``, ``v [T, d_v]`` into the memory state."""
        phi = feature_map(k)  # [T, d_k]
        # M += phiᵀ V   (d_k x d_v);  z += sum_t phi (d_k)
        self.M = self.M + phi.transpose(0, 1) @ v
        self.z = self.z + phi.sum(dim=0)

    def retrieve(self, q: Tensor) -> Tensor:
        """Return the memory head for query ``q``: ``[d_k]`` -> ``[1, d_v]``.

        An empty memory (``z == 0``) contributes exactly ``0`` (no contexts yet),
        matching the dense-only behaviour of a fresh Infini-Attention head.
        """
        q = q.reshape(-1)  # [d_k]
        phi_q = feature_map(q)  # [d_k]
        denom = float(phi_q @ self.z)
        if denom == 0.0:
            return torch.zeros(1, self.d_v, dtype=self.M.dtype, device=self.M.device)
        ctx = (phi_q @ self.M).unsqueeze(0) / denom  # [1, d_v]
        return ctx


def memory_retrieval_attention(q: Tensor, memory: InfiniMemory) -> Tensor:
    """Convenience: the memory context vector for ``q`` (``[d_k]`` or ``[1,d_k]``)."""
    return memory.retrieve(q)


def infinite_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    memory: InfiniMemory | None = None,
    beta: float = DEFAULT_BETA,
    causal: bool = True,
) -> tuple[Tensor, InfiniMemory]:
    """One forward over a segment ``Q/K/V [T, d]`` combining causal dot attention
    with the compressive-memory head.

    Memory is updated *after* each position's retrieval (so it sees only the
    past), and both the dot and memory heads are combined via ``sigmoid(beta)``.
    A ``None`` memory is created on the query's device/dtype. Returns
    ``(outputs [T, d_v], memory)``.
    """
    seq_len = q.shape[0]
    d = q.shape[-1]
    device = q.device
    dtype = q.dtype
    if memory is None:
        memory = InfiniMemory(d, v.shape[-1], dtype=dtype, device=device)

    dot_scale = d**-0.5
    mask_out = None
    if causal:
        mask_out = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )  # True=block future

    gate = float(torch.sigmoid(torch.tensor(beta, dtype=dtype)))  # sigmoid(beta) weight of dot head

    outs: list[Tensor] = []
    for t in range(seq_len):
        # Memory head over the accumulated past (positions < t).
        a_mem = memory.retrieve(q[t])  # [1, d_v]
        # Local causal dot-product head over the current segment, positions <= t.
        scores = (q[t : t + 1] @ k.transpose(0, 1)) * dot_scale  # [1, T]
        if mask_out is not None:
            scores = scores.masked_fill(mask_out[t : t + 1], float("-inf"))
        a_dot = torch.softmax(scores, dim=-1) @ v  # [1, d_v]

        o = gate * a_dot + (1.0 - gate) * a_mem
        outs.append(o)
        # Persist the current key/value so later positions can retrieve it.
        memory.update(k[t : t + 1], v[t : t + 1])

    return torch.cat(outs, dim=0), memory
