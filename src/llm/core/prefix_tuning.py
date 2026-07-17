"""Prefix Tuning module (parameter-efficient fine-tuning, T2 PEFT).

Wraps a frozen :class:`MultiHeadAttention` with a trainable prefix that
gets prepended to K and V at every forward pass. The prefix lives in a
small latent space (``prefix_small``) and is projected into the K/V
dimensions by two reparameterization MLPs — that's the "reparameterized"
Prefix Tuning from Li & Liang 2021, which stabilises training versus
directly learning the K/V prefix.

Forward keeps the base MHA entirely frozen — only ``prefix_small`` and
the reparameterization MLPs receive gradients. After training,
:func:`fold_reparameterization` collapses the MLPs into static
``prefix_k`` / ``prefix_v`` buffers for inference (one fewer matmul per
step, no trainable params at serve time, no risk of training-mode
behaviour leaking into deployment).

Foundation slice: only MHA is supported. MLA / Flash / SDPA attention
variants intentionally raise ``TypeError`` at construction time so the
failure is loud — see :class:`llm.core.attn.base.PrefixCapableAttention`
for the protocol other backends would need to implement.

Reference: Li & Liang, 2021 — *Prefix-Tuning: Optimizing Continuous
Prompts for Generation*, arXiv:2101.00190.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn

from llm.core.attn.mha import MultiHeadAttention


class PrefixTuningAttention(nn.Module):
    """Wrap a frozen ``MultiHeadAttention`` with trainable prefix K/V.

    The trainable parameters are:

    - ``prefix_small``: ``(prefix_len, reparam_hidden)`` — small latent
      prefix. Lower-rank than the full K/V dimension so the search space
      is bounded.
    - ``_reparam_k`` / ``_reparam_v``: ``nn.Linear(reparam_hidden, kv_dim)``
      MLPs that project the small prefix into the K and V spaces.

    Forward computes prefix K/V via the reparam MLPs (or reads the
    static buffers if :func:`fold_reparameterization` has been called),
    expands to ``[B, num_kv_heads, prefix_len, head_dim]``, and dispatches
    to ``base_attn(x, prefix_kv=...)``. The base MHA is frozen at
    construction so only prefix parameters receive gradients.

    Args:
        base_attn: The frozen :class:`MultiHeadAttention` to wrap. Must
            be a real MHA — Flash / MLA / SDPA backends intentionally
            raise so the failure is loud (the foundation slice only
            supports MHA).
        prefix_len: Number of prefix tokens to prepend to each layer's
            K and V. Typical values: 10–200.
        reparam_hidden: Width of the reparam MLP's hidden dim. Defaults
            to ``kv_dim``. Smaller values reduce trainable parameters
            at the cost of expressivity.
    """

    def __init__(
        self,
        base_attn: MultiHeadAttention,
        prefix_len: int,
        reparam_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_attn, MultiHeadAttention):
            raise TypeError(
                f"PrefixTuningAttention requires a MHA (MultiHeadAttention) base; "
                f"got {type(base_attn).__name__}. Flash / MLA / SDPA don't accept "
                f"prefix_kv yet — see llm.core.attn.base.PrefixCapableAttention."
            )

        self.base_attn = base_attn
        self.prefix_len = prefix_len
        self.num_kv_heads = base_attn.num_kv_heads
        self.head_dim = base_attn.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.reparam_hidden = (
            reparam_hidden if reparam_hidden is not None else self.kv_dim
        )

        # Trainable parameters.
        # prefix_small is the "small latent" prefix — lives in
        # ``(prefix_len, reparam_hidden)`` and gets projected into K/V
        # space by the reparam MLPs below.
        self.prefix_small = nn.Parameter(
            torch.empty(prefix_len, self.reparam_hidden)
        )
        self._reparam_k = nn.Linear(self.reparam_hidden, self.kv_dim, bias=True)
        self._reparam_v = nn.Linear(self.reparam_hidden, self.kv_dim, bias=True)

        # Init: prefix_small and both reparam weight matrices use Kaiming
        # uniform so gradients flow at step 1 (a zero-init for the
        # reparam would make ``d_pk / d_prefix_small = 0`` and stall the
        # prefix path until the reparam learned from somewhere else —
        # a chicken-and-egg problem that delays convergence). Biases are
        # zero. The initial prefix contribution is small but non-zero;
        # the optimizer (typically Adam) drives it toward whatever the
        # task loss asks for.
        nn.init.kaiming_uniform_(self.prefix_small, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._reparam_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._reparam_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self._reparam_k.bias)
        nn.init.zeros_(self._reparam_v.bias)

        # Freeze base MHA — only prefix params train.
        for p in self.base_attn.parameters():
            p.requires_grad = False

    # --- Internal helpers --------------------------------------------------

    def _project_prefix(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prefix K/V via the reparam MLPs (training path)."""
        pk = self._reparam_k(self.prefix_small)
        pv = self._reparam_v(self.prefix_small)
        return pk, pv

    def _expand_to_attn_shape(
        self, pk: torch.Tensor, pv: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape ``[prefix_len, kv_dim]`` → ``[B, num_kv_heads, prefix_len, head_dim]``.

        The buffer / reparam output lives in 2D ``[prefix_len, kv_dim]``.
        Attention expects 4D ``[B, num_kv_heads, prefix_len, head_dim]``;
        we broadcast across the batch dim (the prefix is shared across
        the batch — every sequence in a batch sees the same prefix).
        """
        # [prefix_len, kv_dim] -> [1, num_kv_heads, prefix_len, head_dim]
        pk = pk.view(1, self.num_kv_heads, self.prefix_len, self.head_dim)
        pv = pv.view(1, self.num_kv_heads, self.prefix_len, self.head_dim)
        # Broadcast across the batch dim.
        pk = pk.expand(batch_size, -1, -1, -1)
        pv = pv.expand(batch_size, -1, -1, -1)
        return pk, pv

    # --- Forward -----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch to the base MHA with the prefix K/V prepended.

        Forwards every keyword argument (``attn_mask``, ``is_causal``,
        ``kv_cache``, ``use_cache``, ``batch_indices``, ``start_pos``,
        ``paged_kv_cache``, ``layer_idx``) to the base MHA. Any caller-
        supplied ``prefix_kv`` is silently overridden — the wrapper owns
        prefix construction.
        """
        kwargs.pop("prefix_kv", None)

        batch_size = hidden_states.shape[0]
        if hasattr(self, "prefix_k") and hasattr(self, "prefix_v"):
            # Folded: static buffers (no reparam MLPs in play).
            pk, pv = self._expand_to_attn_shape(
                self.prefix_k, self.prefix_v, batch_size
            )
        else:
            pk, pv = self._project_prefix()
            pk, pv = self._expand_to_attn_shape(pk, pv, batch_size)

        return self.base_attn(
            hidden_states,
            prefix_kv=(pk, pv),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (mirror llm.core.lora / llm.core.adalora surface)
# ---------------------------------------------------------------------------


def apply_prefix_tuning(
    model: nn.Module,
    prefix_len: int,
    reparam_hidden: int | None = None,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Wrap every matching ``MultiHeadAttention`` in ``model`` with prefix tuning.

    Mirrors :func:`llm.core.lora.apply_lora` and
    :func:`llm.core.adalora.apply_adalora` so swapping LoRA → AdaLoRA →
    Prefix Tuning in user code is a one-import change.

    Args:
        model: The model to adapt. Modified in-place.
        prefix_len: Number of prefix tokens per attention layer.
        reparam_hidden: Hidden dim of the reparam MLP. ``None`` → defaults
            to ``kv_dim`` (the base attention's full K/V dimension).
        target_modules: List of module-name substring patterns. If
            ``None``, every ``MultiHeadAttention`` is wrapped. Otherwise
            only modules whose qualified name contains any of the
            patterns are wrapped.

    Returns:
        The same model, modified in-place.
    """
    if target_modules is None:
        target_modules = []

    def should_apply(name: str) -> bool:
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention) and should_apply(name):
            replacements.append((name, module))

    for name, module in replacements:
        wrapper = PrefixTuningAttention(
            base_attn=module,
            prefix_len=prefix_len,
            reparam_hidden=reparam_hidden,
        )
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    return model


def get_prefix_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every trainable prefix parameter (``prefix_small`` + reparam MLPs).

    Trainers pass this to the optimizer so only the prefix path is
    updated — the base MHA weights stay frozen. Yields 5 parameters per
    wrapped attention:

    - ``prefix_small``
    - ``_reparam_k.weight``, ``_reparam_k.bias``
    - ``_reparam_v.weight``, ``_reparam_v.bias``
    """
    for module in model.modules():
        if isinstance(module, PrefixTuningAttention):
            yield module.prefix_small
            yield module._reparam_k.weight
            yield module._reparam_k.bias
            yield module._reparam_v.weight
            yield module._reparam_v.bias


def fold_reparameterization(model_or_attn: nn.Module) -> nn.Module:
    """Collapse reparam MLPs into static prefix buffers for inference.

    After fold:

    - ``prefix_small``, ``_reparam_k``, ``_reparam_v`` are removed from
      the wrapper (so the optimizer no longer references them and the
      model state_dict stops carrying them).
    - ``prefix_k``, ``prefix_v`` are registered as buffers with the
      final per-layer K/V values. They are not trainable.
    - Forward path skips the reparam MLPs and reads the buffers directly.

    Idempotent: calling on an already-folded wrapper is a no-op (the
    existing buffers are preserved exactly).

    Works on either a top-level model (walks every wrapper) or a single
    :class:`PrefixTuningAttention` directly. Models with no prefix
    wrappers are left untouched.

    Args:
        model_or_attn: A model containing wrapped attention modules, or
            a single :class:`PrefixTuningAttention`.

    Returns:
        The same object, modified in-place.
    """
    for module in model_or_attn.modules():
        if not isinstance(module, PrefixTuningAttention):
            continue
        # Idempotent: skip already-folded wrappers.
        if hasattr(module, "prefix_k") and hasattr(module, "prefix_v"):
            continue
        # Compute final K/V via reparam MLPs (no_grad so the buffers are
        # constants — they're meant to be the deployment-time values).
        with torch.no_grad():
            pk = module._reparam_k(module.prefix_small).detach().clone()
            pv = module._reparam_v(module.prefix_small).detach().clone()
        # Register as buffers (not Parameters — they're static post-fold).
        module.register_buffer("prefix_k", pk)
        module.register_buffer("prefix_v", pv)
        # Drop the trainable reparam path. Using delattr (rather than
        # setting to None) ensures nn.Module removes the entries from
        # ``_parameters`` / ``_modules`` so the optimizer and state_dict
        # stop referencing them.
        delattr(module, "_reparam_k")
        delattr(module, "_reparam_v")
        delattr(module, "prefix_small")

    return model_or_attn
