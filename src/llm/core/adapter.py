"""Adapter Layers (Houlsby et al. 2019).

Parameter-efficient fine-tuning via bottleneck modules inserted into
transformer blocks. The adapter is a small feed-forward block with
a residual connection - the base Linear is frozen, and only the
adapter parameters train.

Per the original paper, adapters are inserted:

    ``x → Linear → activation → Linear → + residual → output``

i.e. a down-projection (to a small bottleneck dim), a non-linearity,
and an up-projection back to the hidden dim, summed with the base
output. The up-projection is zero-initialized so the adapter is the
identity transform at step 1 - no chicken-and-egg training stall.

Per-layer trainable cost: ``hidden_size x bottleneck_dim +
bottleneck_dim x hidden_size + bottleneck_dim + hidden_size``
(down weight + up weight + down bias + up bias). At
``hidden_size=4096, bottleneck_dim=64`` that's
``2 x 4096 x 64 + 64 + 4096 ≈ 528k`` params per adapted Linear - vs.
LoRA's ``rank x (in + out) = 8 x (4096 + 4096) ≈ 65k`` and IA³'s
``4096``. Adapters are usually bigger than LoRA / IA³ but smaller
than full fine-tuning.

Forward:
    ``y = base_linear(x) + up(activation(down(base_linear(x))))``

Initialization:
    - ``down``: Kaiming uniform (standard Linear init).
    - ``up``: zeros - so the adapter contributes 0 to the output at
      step 1, and the wrapper is the identity on top of the base.
    - ``activation``: ``nn.ReLU`` (the original paper uses ``ReLU``;
      later work uses ``GELU`` or ``Tanh`` - picked here to match
      Houlsby 2019).

The helper API (``apply_adapter`` / ``merge_adapter`` /
``unmerge_adapter`` / ``get_adapter_parameters`` /
``count_adapter_parameters`` / ``disable_adapter`` /
``enable_adapter``) mirrors LoRA / IA³ / BitFit so swapping PEFT
methods in user code is a one-import change. Note that
``merge_adapter`` is a near no-op for adapters - the up-projection
being zero means the adapter contributes nothing, so merging the
adapter into the base would just add zeros. The function is kept
for API parity.

Reference: Houlsby et al., 2019 - *Parameter-Efficient Transfer
Learning for NLP*, arXiv:1902.00751. The bottleneck-only-after-FFN
variant (Pfeiffer et al. 2020) and the Compacter / MAD-X
decompositions are deliberate follow-ups.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn


class AdapterLinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a bottleneck adapter on the output.

    Args:
        base_layer: The original ``nn.Linear`` to adapt (frozen at
            construction).
        bottleneck_dim: Width of the adapter's hidden dim. Smaller
            values reduce trainable parameters; typical values are
            8-256 depending on the hidden size.
        activation: Non-linearity class (defaults to ``nn.ReLU`` to
            match Houlsby 2019).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        bottleneck_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        self.base_layer = base_layer
        self.bottleneck_dim = bottleneck_dim

        hidden = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Down-project: hidden → bottleneck.
        self.down = nn.Linear(hidden, bottleneck_dim, device=device, dtype=dtype)
        self.activation = activation()
        # Up-project: bottleneck → hidden. Zero-init so the adapter
        # is the identity transform at step 1 (no chicken-and-egg
        # training stall - the up output is zero, so the wrapper
        # matches the base output at step 1).
        self.up = nn.Linear(bottleneck_dim, hidden, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Freeze the base layer so only the adapter trains.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen base output + residual adapter output."""
        base_output = self.base_layer(x)
        adapter_output = self.up(self.activation(self.down(base_output)))
        return base_output + adapter_output

    def merge_weights(self) -> None:
        """No-op for adapters.

        Unlike LoRA / IA³ the adapter has no math to fold into the
        base weight - the up-projection being zero means the adapter
        contributes zero to the output. ``merge_weights`` is kept
        for API parity with the other PEFT helpers; it does nothing.
        """
        # Intentionally empty - see docstring.

    def unmerge_weights(self) -> None:
        """No-op for adapters (mirror of :meth:`merge_weights`).

        Kept for API parity; nothing to undo.
        """
        # Intentionally empty - see docstring.

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable adapter parameters (down + up weights + biases)."""
        return self.down.weight.numel() + self.down.bias.numel() + self.up.weight.numel() + self.up.bias.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"bottleneck={self.bottleneck_dim}, "
            f"trainable={self.trainable_parameters}"
        )


def apply_adapter(
    model: nn.Module,
    bottleneck_dim: int = 64,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply adapter bottleneck modules to specified linear layers.

    Args:
        model: The model to adapt (modified in-place).
        bottleneck_dim: Width of the adapter's hidden dim (passed
            through to :class:`AdapterLinear`).
        target_modules: List of module-name substring patterns. If
            ``None`` (default), every ``nn.Linear`` is wrapped. Pass
            e.g. ``["q_proj", "v_proj"]`` to wrap only attention
            projections.

    Returns:
        The model with adapters applied (modified in-place).
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
        adapter = AdapterLinear(module, bottleneck_dim=bottleneck_dim)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], adapter)

    return model


def merge_adapter(model: nn.Module) -> nn.Module:
    """No-op for adapters - kept for API parity with LoRA / IA³.

    Unlike LoRA / IA³, the adapter has no math to fold into the base
    weight. The up-projection is zero-initialized, so the adapter
    contributes zero to the output unless the user trained it.

    This function is provided so that ``apply_adapter`` /
    ``merge_adapter`` / ``unmerge_adapter`` follow the same call
    pattern as the other PEFT helpers.
    """
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            module.merge_weights()
    return model


def unmerge_adapter(model: nn.Module) -> nn.Module:
    """No-op for adapters - mirror of :func:`merge_adapter`."""
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            module.unmerge_weights()
    return model


def get_adapter_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every trainable adapter parameter - down + up weights + biases
    per wrapper, nothing from the base Linear.
    """
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            yield module.down.weight
            yield module.down.bias
            yield module.up.weight
            yield module.up.bias


def count_adapter_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs. total parameters in an adapter-adapted model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def disable_adapter(model: nn.Module) -> None:
    """Disable adapters by zeroing the up-projection (so the adapter
    contributes zero to the output).

    Saves the up-projection weight / bias under ``_original_up_weight``
    / ``_original_up_bias`` so :func:`enable_adapter` can restore them.
    """
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            module._original_up_weight = module.up.weight.detach().clone()
            module._original_up_bias = module.up.bias.detach().clone()
            with torch.no_grad():
                module.up.weight.zero_()
                module.up.bias.zero_()


def enable_adapter(model: nn.Module) -> None:
    """Re-enable adapters after :func:`disable_adapter` - restores the
    saved up-projection snapshot. No-op if ``disable_adapter`` was
    never called.
    """
    for module in model.modules():
        if isinstance(module, AdapterLinear) and hasattr(module, "_original_up_weight"):
            with torch.no_grad():
                module.up.weight.copy_(module._original_up_weight)
                module.up.bias.copy_(module._original_up_bias)
            del module._original_up_weight
            del module._original_up_bias
