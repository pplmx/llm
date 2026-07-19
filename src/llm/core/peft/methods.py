"""Built-in PEFT method registrations (T2 PEFT #43).

Each entry is a thin wrapper around the existing module-level
``apply_*`` / ``merge_*`` / etc. functions in ``llm.core.{lora, qlora,
adalora, prefix_tuning, ia3, bitfit, adapter}``. The wrappers exist so
the registry can hold a uniform :class:`PEFTMethod` record for every
built-in — no behaviour is duplicated, and the per-method API surface
(asymmetric: lora has merge, bitfit doesn't, prefix_tuning has
``fold_reparameterization`` instead of merge, etc.) is faithfully
recorded via the dataclass's ``Optional`` fields.

This module is imported lazily by :func:`ensure_methods_registered` —
not at package import time — so the PEFT registry stays opt-in and a
user who never touches PEFT pays no import cost.
"""

from __future__ import annotations

from llm.core.adalora import (
    apply_adalora,
    count_adalora_parameters,
    disable_adalora,
    enable_adalora,
    get_adalora_parameters,
    merge_adalora,
    unmerge_adalora,
)
from llm.core.adapter import (
    apply_adapter,
    count_adapter_parameters,
    disable_adapter,
    enable_adapter,
    get_adapter_parameters,
    merge_adapter,
    unmerge_adapter,
)
from llm.core.bitfit import (
    apply_bitfit,
    count_bitfit_parameters,
    get_bitfit_parameters,
)
from llm.core.ia3 import (
    apply_ia3,
    count_ia3_parameters,
    disable_ia3,
    enable_ia3,
    get_ia3_parameters,
    merge_ia3,
    unmerge_ia3,
)
from llm.core.lora import (
    apply_lora,
    count_lora_parameters,
    disable_lora,
    enable_lora,
    get_lora_parameters,
    merge_lora,
    unmerge_lora,
)
from llm.core.peft.types import PEFTMethod, TargetModuleFilter
from llm.core.pfeiffer_adapter import (
    apply_pfeiffer_adapter,
    count_pfeiffer_parameters,
    disable_pfeiffer_adapter,
    enable_pfeiffer_adapter,
    get_pfeiffer_parameters,
    merge_pfeiffer_adapter,
    unmerge_pfeiffer_adapter,
)
from llm.core.prefix_tuning import apply_prefix_tuning, get_prefix_parameters
from llm.core.qlora import apply_qlora, get_qlora_parameters

# ---------------------------------------------------------------------------
# Built-in PEFTMethod records
# ---------------------------------------------------------------------------
#
# Naming: keys are short, lowercase, and match the per-method helper
# suffix (e.g. ``apply_lora`` → ``"lora"``, ``apply_prefix_tuning`` →
# ``"prefix_tuning"``). Keys are referenced from user configs via
# ``TrainingConfig.peft_method``.
#
# ``requires_callback`` is True only for AdaLoRA — it's the one method
# that needs a periodic :class:`AdaLoRAPruningCallback` to drive its
# rank schedule.
#
# ``target_module_filter`` is metadata for introspection / docs. The
# actual filtering logic lives in each per-method ``apply_*`` function
# (which accepts a ``target_modules`` substring list).


_BUILTIN_METHODS: list[PEFTMethod] = [
    PEFTMethod(
        name="lora",
        apply=apply_lora,
        get_parameters=get_lora_parameters,
        count_parameters=count_lora_parameters,
        merge=merge_lora,
        unmerge=unmerge_lora,
        disable=disable_lora,
        enable=enable_lora,
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
    PEFTMethod(
        name="qlora",
        apply=apply_qlora,
        get_parameters=get_qlora_parameters,
        # NF4-quantized base weight cannot be re-folded into a float
        # tensor — QLoRA exposes apply + get_parameters only.
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
    PEFTMethod(
        name="adalora",
        apply=apply_adalora,
        get_parameters=get_adalora_parameters,
        count_parameters=count_adalora_parameters,
        merge=merge_adalora,
        unmerge=unmerge_adalora,
        disable=disable_adalora,
        enable=enable_adalora,
        requires_callback=True,  # AdaLoRAPruningCallback
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
    PEFTMethod(
        name="prefix_tuning",
        apply=apply_prefix_tuning,
        get_parameters=get_prefix_parameters,
        # Inference-time fold is ``fold_reparameterization`` — not the
        # merge/unmerge protocol. The registry doesn't proxy that helper
        # because it's a one-shot fold into static buffers, not a
        # base-weight mutation.
        target_module_filter=TargetModuleFilter.MHA,
    ),
    PEFTMethod(
        name="ia3",
        apply=apply_ia3,
        get_parameters=get_ia3_parameters,
        count_parameters=count_ia3_parameters,
        merge=merge_ia3,
        unmerge=unmerge_ia3,
        disable=disable_ia3,
        enable=enable_ia3,
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
    PEFTMethod(
        name="bitfit",
        apply=apply_bitfit,
        get_parameters=get_bitfit_parameters,
        count_parameters=count_bitfit_parameters,
        # BitFit's biases are simply part of the model at serve time —
        # no merge / disable / enable helpers exist on the per-method
        # API.
        target_module_filter=TargetModuleFilter.ANY,
    ),
    PEFTMethod(
        name="adapter",
        apply=apply_adapter,
        get_parameters=get_adapter_parameters,
        count_parameters=count_adapter_parameters,
        merge=merge_adapter,
        unmerge=unmerge_adapter,
        disable=disable_adapter,
        enable=enable_adapter,
        # Houlsby 2019 bottleneck residual — the up projection is zero
        # at init, so ``merge_adapter`` is a documented no-op (the
        # base weight doesn't change). The registry still surfaces it
        # for API parity with LoRA's apply/merge pattern.
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
    PEFTMethod(
        name="pfeiffer_adapter",
        apply=apply_pfeiffer_adapter,
        get_parameters=get_pfeiffer_parameters,
        count_parameters=count_pfeiffer_parameters,
        merge=merge_pfeiffer_adapter,
        unmerge=unmerge_pfeiffer_adapter,
        disable=disable_pfeiffer_adapter,
        enable=enable_pfeiffer_adapter,
        # Pfeiffer 2020 — adapter ONLY after FFN/MLP, not after
        # attention. Same wrapper class as Houlsby
        # (``AdapterLinear``) but a different default ``target_modules``
        # filter (``["fc1", "fc2"]``). Roughly half the parameters of
        # Houlsby at comparable accuracy; the production default in
        # AdapterHub / HuggingFace PEFT.
        target_module_filter=TargetModuleFilter.LINEAR,
    ),
]


def iter_builtin_methods() -> list[PEFTMethod]:
    """Return the list of built-in :class:`PEFTMethod` records.

    Returned by value (not a generator) so callers can iterate
    multiple times — used by :func:`ensure_methods_registered` to
    populate the registry idempotently.
    """
    return list(_BUILTIN_METHODS)


__all__ = ["iter_builtin_methods"]
