"""Public types for the PEFT registry (T2 PEFT #43).

The :class:`PEFTMethod` dataclass is the contract every PEFT method —
built-in or third-party plugin — must satisfy to register with
:data:`llm.core.peft.registry.PEFT_REGISTRY`.

Built-in PEFT methods expose **asymmetric** API surfaces:

- ``lora`` / ``adalora`` / ``ia3`` / ``adapter``: apply / get_parameters /
  count_parameters / merge / unmerge / disable / enable — the full set
- ``bitfit``: apply / get_parameters / count_parameters — no merge
  (biases are kept at inference, no fold step)
- ``qlora``: apply / get_parameters — no merge (NF4 quantized base
  cannot be re-folded into a float tensor)
- ``prefix_tuning``: apply / get_parameters — inference-time fold is
  :func:`llm.core.prefix_tuning.fold_reparameterization`, not the
  merge/unmerge protocol

The dataclass accommodates all of these by making ``get_parameters`` /
``count_parameters`` / ``merge`` / ``unmerge`` / ``disable`` / ``enable``
:data:`Optional`. Callers that hit a ``None`` helper get a loud
``NotImplementedError`` (see :mod:`llm.core.peft.registry`) instead of a
silent skip — the failure mode is the same as the per-method
``apply_*`` raising ``TypeError`` on a non-MHA base.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    pass


class TargetModuleFilter(StrEnum):
    """What kind of submodules a PEFT method targets.

    Used as metadata only — the actual filter logic lives in the
    per-method ``apply_*`` function (which already accepts a
    ``target_modules`` substring list). The enum lets introspection /
    docs report "this method wraps Linear layers" vs "this method
    wraps Multi-Head Attention" without importing the method module.

    Inherits from :class:`enum.StrEnum` so the values serialize
    naturally to JSON strings (e.g. in the docs build or in
    ``metadata.json`` snapshots).
    """

    LINEAR = "linear"
    MHA = "mha"
    ANY = "any"


@dataclass(frozen=True)
class PEFTMethod:
    """The contract every PEFT method registers with the registry.

    Attributes:
        name: Unique registry name (e.g. ``"lora"``, ``"adalora"``,
            ``"prefix_tuning"``). Matches the registry key.
        apply: ``(model, **kwargs) -> nn.Module`` — wraps the model
            in-place (per the existing ``apply_*`` convention) and
            returns it for chainability. **Required.**
        get_parameters: ``(model) -> Iterator[nn.Parameter]`` — yields
            exactly the trainable parameters added by this method.
            ``None`` means the method doesn't expose a parameter
            iterator (callers should fall back to
            ``[p for p in model.parameters() if p.requires_grad]``).
        count_parameters: ``(model) -> tuple[int, int]`` — returns
            ``(trainable, total)`` for the wrapped model. ``None``
            means the method doesn't expose a count helper.
        merge: ``(model) -> nn.Module`` — inference-time fold of the
            adapter into the base weight. ``None`` for methods that
            don't fold (bitfit / qlora / prefix_tuning).
        unmerge: ``(model) -> nn.Module`` — reverse the merge.
            ``None`` when merge is ``None``.
        disable: ``(model) -> None`` — disable the adapter (e.g. for
            ablation studies). ``None`` when not supported.
        enable: ``(model) -> None`` — re-enable a previously disabled
            adapter. ``None`` when not supported.
        requires_callback: Whether the method needs a periodic trainer
            callback. Currently only ``adalora`` sets this to ``True``
            (the :class:`AdaLoRAPruningCallback`).
        target_module_filter: What kind of submodule the method
            wraps. ``"linear"`` for LoRA / AdaLoRA / IA³ / Adapter /
            QLoRA (wrap ``nn.Linear``), ``"mha"`` for Prefix Tuning
            (wrap ``MultiHeadAttention``), ``"any"`` for BitFit (just
            toggles ``requires_grad``).

    Notes:
        The dataclass is ``frozen=True`` — methods are registered
        once at module import and never mutated. ``apply`` and the
        helpers are stored as raw callables, not bound to the dataclass,
        so ``is`` identity comparisons with the per-module functions
        succeed (``PEFT_REGISTRY.get("lora").apply is apply_lora``).
    """

    name: str
    apply: Callable[..., nn.Module]
    get_parameters: Callable[[nn.Module], Iterator[nn.Parameter]] | None = None
    count_parameters: Callable[[nn.Module], tuple[int, int]] | None = None
    merge: Callable[[nn.Module], nn.Module] | None = None
    unmerge: Callable[[nn.Module], nn.Module] | None = None
    disable: Callable[[nn.Module], None] | None = None
    enable: Callable[[nn.Module], None] | None = None
    requires_callback: bool = False
    target_module_filter: TargetModuleFilter = TargetModuleFilter.LINEAR


__all__ = ["PEFTMethod", "TargetModuleFilter"]
