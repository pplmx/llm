"""IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).

A parameter-efficient fine-tuning method that wraps a frozen
``nn.Linear`` with a single trainable vector that multiplicatively
scales the output. IA³ is the multiplicative counterpart to LoRA's
additive design — instead of adding ``ΔW · x`` it scales the existing
output ``W · x`` element-wise on the output dimension.

Per-layer cost: ``out_features`` trainable parameters (vs. LoRA's
``rank * (in_features + out_features)`` and AdaLoRA's ``init_rank *
(in_features + out_features)``). At ``out_features=4096`` and
``rank=8``, ``in_features=4096`` that is ~4k vs. ~65k parameters per
adapted linear — typically two orders of magnitude fewer than LoRA.

Forward:
    ``y = (W · x + b) * l``
    where ``l`` is a learned vector of shape ``(out_features,)``
    broadcast across batch and sequence dims.

Initialization:
    ``l ← ones`` so the wrapped layer starts as the identity transform
    — the model's existing forward is preserved at step 1, which
    avoids the chicken-and-egg problem that a zero-init would cause.

Merge for inference:
    ``W ← W * l[None, :]`` and ``b ← b * l``. The learned vector folds
    into the base weight and can be discarded — no extra params at
    serve time, no extra matmul. Symmetric ``unmerge_weights`` reverses
    the merge so the same model can be checkpointed pre- and post-merge.

Reference: Liu et al., 2022 — *Few-Shot Parameter-Efficient Fine-Tuning
is Better and Cheaper than In-Context Learning*, arXiv:2205.05638
(aka "T-Few"). The paper applies IA³ to attention K/V/output and FFN
intermediate projections — the helper API mirrors LoRA so swapping
``apply_lora`` for ``apply_ia3`` is a one-import change.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn


class IA3Linear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable multiplicative scale.

    Args:
        base_layer: The original ``nn.Linear`` to adapt (frozen at
            construction). The base layer's weight is **not** merged
            on init — the wrapper just multiplies the output of the
            base layer by the learned ``ia3_l`` vector.
        init_scale: Initial value of the multiplicative scale. Defaults
            to ``1.0`` so the wrapper starts as the identity transform
            — the model behaves identically to the base at step 1.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer

        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # One learned multiplier per output channel. Broadcasts over
        # batch and sequence dims at forward time.
        self.ia3_l = nn.Parameter(torch.full((out_features,), init_scale, device=device, dtype=dtype))

        # Freeze the base layer so only ``ia3_l`` is trainable.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen base output, multiplicatively scaled."""
        base_output = self.base_layer(x)
        # ``ia3_l`` has shape (out_features,). The base output is
        # (..., out_features) — broadcasting handles the rest.
        return base_output * self.ia3_l

    def merge_weights(self) -> None:
        """Merge the multiplicative scale into the base weight for inference.

        After this call, ``self.base_layer.weight`` already contains
        the scale (``W * l[None, :]``) and ``self.ia3_l`` is set to
        ones so the wrapper is the identity on top of the already-scaled
        base. ``unmerge_weights`` reverses the operation, restoring
        both the original ``ia3_l`` snapshot and the pre-merge base
        weight — useful for checkpoint roundtrip where the same model
        needs to be saved both pre- and post-merge.
        """
        with torch.no_grad():
            # Save the original scale so ``unmerge_weights`` can
            # restore it after dividing the base weight back out.
            self._merged_ia3_l = self.ia3_l.detach().clone()
            # Fold the scale into the base weight.
            # ``weight`` has shape ``(out_features, in_features)``; we
            # scale each output-channel row by the matching ``ia3_l``
            # entry, which means multiplying on dim 0 with a
            # ``(out_features, 1)`` broadcast.
            self.base_layer.weight.mul_(self.ia3_l.unsqueeze(1))
            if self.base_layer.bias is not None:
                self.base_layer.bias.mul_(self.ia3_l)
            # Zero-out the active scale — the wrapper is now identity
            # on top of the already-folded weight, so forward still
            # produces the same output as pre-merge.
            self.ia3_l.fill_(1.0)

    def unmerge_weights(self) -> None:
        """Reverse :meth:`merge_weights` — restores both the saved
        ``ia3_l`` snapshot and the pre-merge base weight. No-op if
        :meth:`merge_weights` was never called.
        """
        with torch.no_grad():
            if not hasattr(self, "_merged_ia3_l"):
                return
            # Restore the active scale to its pre-merge value, then
            # divide it back out of the base weight.
            self.ia3_l.copy_(self._merged_ia3_l)
            del self._merged_ia3_l
            self.base_layer.weight.div_(self.ia3_l.unsqueeze(1))
            if self.base_layer.bias is not None:
                self.base_layer.bias.div_(self.ia3_l)

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable IA³ parameters (just ``ia3_l.numel()``)."""
        return self.ia3_l.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"trainable={self.ia3_l.numel()}"
        )


def apply_ia3(
    model: nn.Module,
    init_scale: float = 1.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply IA³ to specified linear layers in a model.

    Args:
        model: The model to adapt (modified in-place).
        init_scale: Initial value of the multiplicative scale (passed
            through to :class:`IA3Linear`).
        target_modules: List of module-name substring patterns. If
            ``None`` (default), every ``nn.Linear`` is wrapped — same
            default as ``apply_lora``. Pass e.g. ``["q_proj", "v_proj"]``
            to wrap only attention projections.

    Returns:
        The model with IA³ applied (modified in-place).
    """
    if target_modules is None:
        target_modules = []

    def should_apply(name: str) -> bool:
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply(name):
            replacements.append((name, module))

    for name, module in replacements:
        ia3_layer = IA3Linear(module, init_scale=init_scale)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ia3_layer)

    return model


def merge_ia3(model: nn.Module) -> nn.Module:
    """Merge every IA³ scale into the wrapped base weight.

    After this, the model is identical to the original at inference
    but no longer has trainable IA³ params. Reversible via
    :func:`unmerge_ia3`.
    """
    for module in model.modules():
        if isinstance(module, IA3Linear):
            module.merge_weights()
    return model


def unmerge_ia3(model: nn.Module) -> nn.Module:
    """Reverse :func:`merge_ia3` — restores the trained ``ia3_l`` as the
    active scale. Useful for checkpoint roundtrip.
    """
    for module in model.modules():
        if isinstance(module, IA3Linear):
            module.unmerge_weights()
    return model


def get_ia3_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every IA³ trainable parameter — one ``ia3_l`` per wrapper.

    Use this to wire the optimizer:
        ``torch.optim.Adam(get_ia3_parameters(model), lr=...)``
    """
    for module in model.modules():
        if isinstance(module, IA3Linear):
            yield module.ia3_l


def count_ia3_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs. total parameters in an IA³-adapted model.

    Returns:
        ``(trainable_params, total_params)`` — the IA³ contribution
        is ``trainable_params``, dominated by the frozen base weights.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def disable_ia3(model: nn.Module) -> None:
    """Disable IA³ at inference — sets every ``ia3_l`` to all-ones so
    the wrapper is the identity transform. Use this when you want to
    evaluate the base model behaviour without un-wrapping.
    """
    for module in model.modules():
        if isinstance(module, IA3Linear):
            module._original_ia3_l = module.ia3_l.detach().clone()
            with torch.no_grad():
                module.ia3_l.fill_(1.0)


def enable_ia3(model: nn.Module) -> None:
    """Re-enable IA³ after :func:`disable_ia3` — restores the saved
    ``ia3_l`` snapshot. No-op if ``disable_ia3`` was never called.
    """
    for module in model.modules():
        if isinstance(module, IA3Linear) and hasattr(module, "_original_ia3_l"):
            with torch.no_grad():
                module.ia3_l.copy_(module._original_ia3_l)
            del module._original_ia3_l
