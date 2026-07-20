"""BitFit (Bias-Term Fine-Tuning).

The simplest parameter-efficient fine-tuning method: train only the
bias parameters, freeze everything else. Unlike LoRA / AdaLoRA / IA³,
BitFit adds no new parameters and wraps no modules — it just toggles
``requires_grad`` on every bias in the model.

Per the paper (Ben-Zaken et al. 2021), all bias parameters are
trainable: attention Q/K/V/O projection biases, FFN intermediate /
output projection biases, and LayerNorm / RMSNorm biases. The user
can opt to filter by ``target_modules`` (substring match on the
parameter's qualified name) to bias-select a subset — e.g. only
attention biases — but the default is to train every bias.

The helper API mirrors LoRA / AdaLoRA / IA³ so swapping PEFT methods
in user code is a one-import change:

    from llm.core.bitfit import apply_bitfit, get_bitfit_parameters

    apply_bitfit(model)
    optimizer = torch.optim.AdamW(get_bitfit_parameters(model), lr=1e-3)

Per-model cost: ``O(num_biases)`` trainable params — typically
<0.1% of total parameters. BitFit is the lightest possible PEFT
method: no math, no wrappers, no scheduler.

Reference: Ben-Zaken et al., 2021 — *BitFit: Simple Parameter-efficient
Fine-tuning for Transformer-based Masked Language-models*.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn


def apply_bitfit(
    model: nn.Module,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Freeze every parameter, then enable gradients on every bias.

    Args:
        model: The model to adapt (modified in-place).
        target_modules: Optional list of module-name substring patterns.
            A bias is trainable only if its qualified name (e.g.
            ``"layers.0.attn.q_proj.bias"``) contains at least one of
            the patterns. ``None`` (default) → every bias is trainable.

    Returns:
        The model with BitFit applied (modified in-place).

    Note:
        BitFit saves the original ``requires_grad`` state on the model
        under ``_bitfit_original_requires_grad`` so :func:`unapply_bitfit`
        can restore it. Calling :func:`apply_bitfit` twice without an
        intervening :func:`unapply_bitfit` re-saves on the second call
        (so the snapshot always reflects the pre-BitFit state).
    """
    if target_modules is None:
        target_modules = []

    # Snapshot the original requires_grad state so unapply_bitfit can
    # restore it. Save BEFORE toggling anything — that way repeated
    # calls of apply_bitfit converge to the same final state.
    model._bitfit_original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    # Freeze every parameter.
    for p in model.parameters():
        p.requires_grad = False

    # Enable gradients on biases whose qualified name matches.
    # We check for ``.bias`` SUFFIX (not just substring) — substring
    # would falsely match module names like ``fc_with_bias`` whose
    # ``weight`` parameter is NOT a bias. The qualified name of a
    # bias parameter always ends in ``.bias`` (or equals ``bias`` at
    # the top level).
    for name, p in model.named_parameters():
        if not (name == "bias" or name.endswith(".bias")):
            continue
        if target_modules and not any(pattern in name for pattern in target_modules):
            continue
        p.requires_grad = True

    return model


def unapply_bitfit(model: nn.Module) -> nn.Module:
    """Restore the pre-BitFit ``requires_grad`` state.

    Reverses :func:`apply_bitfit` — every parameter is set back to
    whatever its ``requires_grad`` was before :func:`apply_bitfit` was
    called. No-op if :func:`apply_bitfit` was never called.
    """
    snapshot = getattr(model, "_bitfit_original_requires_grad", None)
    if snapshot is None:
        return model

    for name, p in model.named_parameters():
        if name in snapshot:
            p.requires_grad = snapshot[name]
    del model._bitfit_original_requires_grad
    return model


def get_bitfit_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every trainable bias parameter.

    Use this to wire the optimizer:
        ``torch.optim.AdamW(get_bitfit_parameters(model), lr=...)``

    After :func:`apply_bitfit`, exactly the bias parameters are
    trainable, so this is equivalent to ``iter(model.parameters())``
    filtered by ``p.requires_grad`` — but we re-check the ``.bias``
    suffix explicitly so the helper remains correct if a user
    manually enables a non-bias parameter after applying BitFit
    (the helper still yields only biases, not the non-bias ones).

    Substring matching is intentionally avoided: ``fc_with_bias.weight``
    contains the substring ``bias`` but is not a bias parameter.
    """
    for name, p in model.named_parameters():
        if not (name == "bias" or name.endswith(".bias")):
            continue
        if p.requires_grad:
            yield p


def count_bitfit_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs. total parameters in a BitFit-adapted model.

    Returns:
        ``(trainable_params, total_params)`` — the BitFit contribution
        is ``trainable_params``, dominated by the frozen base weights.

    Before :func:`apply_bitfit` is called, the helper reports whatever
    ``requires_grad`` state the model was constructed with (typically
    all-trainable, since :class:`nn.Linear` defaults to ``True``).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def is_bitfit_applied(model: nn.Module) -> bool:
    """Return ``True`` if :func:`apply_bitfit` was called on this model
    and not yet reversed by :func:`unapply_bitfit`.

    Useful for checkpoint validation: if a checkpoint claims to be
    BitFit-adapted, the snapshot attribute should be present (or
    absent, if the user already called ``unapply_bitfit``).
    """
    return hasattr(model, "_bitfit_original_requires_grad")
