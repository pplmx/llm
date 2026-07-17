"""Attention-layer protocols shared by all ``attn/`` implementations.

Defines the optional extension points adapters (e.g. Prefix Tuning)
rely on. Implementations only need to honor the protocols they
genuinely support — anything else raises ``NotImplementedError`` at
adapter construction time so the failure mode is loud, not silent.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class PrefixCapableAttention(Protocol):
    """Attention impls that accept an optional ``prefix_kv`` to prepend.

    The prefix K/V are concatenated to the projected K/V along the
    sequence dimension **before** the attention compute and **after**
    any KV-cache write — so the cache only holds the dynamically-
    generated tokens and the prefix is recomputed (or folded to a
    static buffer) on every forward.

    Implementations that don't support prefix tuning (MLA, Flash
    Attention fused kernels, etc.) should leave ``prefix_kv`` as an
    accepted-but-ignored arg or raise ``NotImplementedError``. The
    ``PrefixTuningAttention`` wrapper guards construction with an
    explicit ``isinstance`` check so the failure is at adapter build
    time, not at first forward.

    Args:
        prefix_kv: ``(prefix_k, prefix_v)`` tuple of shape
            ``[B, num_kv_heads, prefix_len, head_dim]``, or ``None``
            to skip prefix injection (the default).
    """

    def forward(
        self,
        hidden_states: Tensor,
        prefix_kv: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> Tensor: ...
