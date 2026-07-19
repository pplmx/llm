"""Pfeiffer Adapter (Pfeiffer et al. 2020).

Parameter-efficient fine-tuning via bottleneck modules inserted **only
after FFN/MLP** layers, not after attention projections. The variant
was introduced in Pfeiffer et al., 2020 — *AdapterHub: A Framework for
Adapting Transformers*, arXiv:2007.07779 — and is the production
default in AdapterHub / HuggingFace PEFT, roughly half the parameters
of Houlsby 2019 at comparable accuracy on most tasks.

The wrapper class is the same :class:`llm.core.adapter.AdapterLinear`
used by Houlsby — there is no new tensor type. The only difference
between Houlsby and Pfeiffer is **which linears get wrapped**:

- Houlsby: every ``nn.Linear`` (attention + MLP)
- Pfeiffer: only the FFN / MLP linears (default ``fc1`` + ``fc2``,
  matching :class:`llm.core.mlp.MLP`)

Per the original paper, the adapter is the same bottleneck residual:

    ``y = base_linear(x) + up(activation(down(base_linear(x))))``

with ``up`` zero-initialized so the adapter is the identity at step 1
(no chicken-and-egg training stall).

Per-layer trainable cost: ``2 * out_features * bottleneck_dim +
out_features + bottleneck_dim`` — identical to Houlsby's per-layer
cost. The parameter savings come from wrapping **fewer layers**, not
from a smaller per-layer footprint. A 2-layer transformer block with
hidden=4096 and bottleneck=64 has:

- Houlsby: 4 attention linears + 2 MLP linears = 6 wrappers
  ≈ 6 * (2 * 4096 * 64 + 4096 + 64) ≈ 3.2M trainable per block
- Pfeiffer: 2 MLP linears = 2 wrappers
  ≈ 2 * (2 * 4096 * 64 + 4096 + 64) ≈ 1.05M trainable per block

The helper API mirrors the Houlsby one (``merge_*`` / ``unmerge_*`` /
``get_*_parameters`` / ``count_*_parameters`` / ``disable_*` /
``enable_*``) so swapping ``adapter`` → ``pfeiffer_adapter`` in user
code is a one-import change. Internally the helpers delegate to the
Houlsby implementations — Pfeiffer wrappers **are**
:class:`llm.core.adapter.AdapterLinear` instances, so there's nothing
to distinguish at runtime.

Reference: Pfeiffer et al., 2020 — *AdapterHub: A Framework for
Adapting Transformers*, arXiv:2007.07779. The Compacter (Kronecker
decomposition) and MAD-X (cross-lingual modular) variants are
deliberate follow-ups; this slice ships the simple FFN-only variant.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn

from llm.core.adapter import (
    apply_adapter,
    count_adapter_parameters,
    disable_adapter,
    enable_adapter,
    get_adapter_parameters,
    merge_adapter,
    unmerge_adapter,
)

# Default target filter: only the FFN / MLP linears (matches the
# standard :class:`llm.core.mlp.MLP` layer names). Users with custom
# MLP modules can pass a custom ``target_modules`` list to
# :func:`apply_pfeiffer_adapter`.
DEFAULT_PFEIFFER_TARGETS: list[str] = ["fc1", "fc2"]


def apply_pfeiffer_adapter(
    model: nn.Module,
    bottleneck_dim: int = 64,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply Pfeiffer Adapter — bottleneck residual only after FFN/MLP.

    Args:
        model: The model to adapt (modified in-place).
        bottleneck_dim: Width of the adapter's hidden dim. Forwarded
            to :class:`llm.core.adapter.AdapterLinear`. Defaults to
            64 (the Houlsby 2019 paper convention, also used by the
            Pfeiffer 2020 reproductions).
        target_modules: List of module-name substring patterns used to
            pick which ``nn.Linear`` modules get wrapped. If ``None``
            (default), the standard FFN/MLP filter ``["fc1", "fc2"]``
            is used — matching the layer names in
            :class:`llm.core.mlp.MLP`. Pass a custom list to wrap a
            different subset (e.g. ``["q_proj", "v_proj"]`` is
            invalid here — Pfeiffer is FFN-only — but you can point
            at non-standard MLP layer names in custom architectures).

    Returns:
        The same ``model`` (modified in-place; chainable).

    Note:
        Internally this is a thin delegate to
        :func:`llm.core.adapter.apply_adapter` with the FFN-only
        target filter. The wrapper class is
        :class:`llm.core.adapter.AdapterLinear` — Pfeiffer IS
        Houlsby-on-MLP-only, so no new wrapper code is needed.

        Unlike LoRA / IA³ / Prefix Tuning, ``bottleneck_dim`` is the
        knob (rather than ``rank``) and the ``up`` projection is
        zero-initialized so the wrapper is the identity transform at
        step 1.
    """
    if target_modules is None:
        target_modules = list(DEFAULT_PFEIFFER_TARGETS)
    return apply_adapter(model, bottleneck_dim=bottleneck_dim, target_modules=target_modules)


def merge_pfeiffer_adapter(model: nn.Module) -> nn.Module:
    """No-op for Pfeiffer — kept for API parity with LoRA / IA³.

    Delegates to :func:`llm.core.adapter.merge_adapter`. The ``up``
    projection being zero means the adapter contributes nothing to
    the output unless the user trained it, so there's nothing to
    fold into the base weight.
    """
    return merge_adapter(model)


def unmerge_pfeiffer_adapter(model: nn.Module) -> nn.Module:
    """No-op for Pfeiffer — mirror of :func:`merge_pfeiffer_adapter`."""
    return unmerge_adapter(model)


def get_pfeiffer_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every trainable Pfeiffer parameter.

    Delegates to :func:`llm.core.adapter.get_adapter_parameters`.
    Since both Houlsby and Pfeiffer produce :class:`AdapterLinear`
    wrappers, this helper yields parameters from **every** adapter
    wrapper in the model (Pfeiffer alone, Houlsby alone, or both
    coexisting). For Pfeiffer-only the result is identical to
    ``get_adapter_parameters``.
    """
    return get_adapter_parameters(model)


def count_pfeiffer_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts.

    Delegates to :func:`llm.core.adapter.count_adapter_parameters`.
    Same caveat as :func:`get_pfeiffer_parameters` — counts every
    adapter wrapper, not just Pfeiffer-targeted ones. Users who mix
    Pfeiffer and Houlsby on the same model should call the per-
    method helpers selectively.
    """
    return count_adapter_parameters(model)


def disable_pfeiffer_adapter(model: nn.Module) -> None:
    """Disable Pfeiffer adapters by zeroing the ``up`` projection.

    Delegates to :func:`llm.core.adapter.disable_adapter`. After
    this call every wrapper's ``up.weight`` and ``up.bias`` are
    zero, making the wrapper mathematically the identity on top of
    the base. The pre-disable up-projection is snapshotted under
    ``_original_up_weight`` / ``_original_up_bias`` so
    :func:`enable_pfeiffer_adapter` can restore it.
    """
    disable_adapter(model)


def enable_pfeiffer_adapter(model: nn.Module) -> None:
    """Re-enable Pfeiffer adapters after :func:`disable_pfeiffer_adapter`.

    Delegates to :func:`llm.core.adapter.enable_adapter`. No-op if
    ``disable_pfeiffer_adapter`` was never called (the snapshot
    attribute is the sentinel).
    """
    enable_adapter(model)


__all__ = [
    "DEFAULT_PFEIFFER_TARGETS",
    "apply_pfeiffer_adapter",
    "count_pfeiffer_parameters",
    "disable_pfeiffer_adapter",
    "enable_pfeiffer_adapter",
    "get_pfeiffer_parameters",
    "merge_pfeiffer_adapter",
    "unmerge_pfeiffer_adapter",
]
