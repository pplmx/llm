"""Small ``nn.Module`` surgery helpers shared by the quantization algorithms.

Every algorithm that swaps a layer for a quantized/decomposed/pruned
replacement needs the same two-line operation: walk a dotted module path and
``setattr`` the replacement onto its real parent. Seven modules (``awq``,
``fp8``, ``gptq``, ``lowrank``, ``prune``, ``ptq``, ``smooth``) each carried a
byte-identical private copy, so a fix to the traversal (say, handling a module
registered under a numeric ``ModuleList`` index reached through ``getattr`` on a
non-``Module`` object) would have had to be applied seven times and would drift
the moment one was missed.
"""

from __future__ import annotations

from torch import nn


def replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the descendant of ``parent`` addressed by dotted ``name``.

    ``name`` is the full dotted path produced by ``model.named_modules()``
    (e.g. ``"blocks.0.attn.q_proj"``). Every segment but the last is resolved
    with ``getattr``; the last is assigned with ``setattr`` on its immediate
    parent, so the replacement is visible to the next ``named_modules()`` walk
    without touching the outer model object.

    Args:
        parent: Root module the dotted path is relative to.
        name: Dotted attribute path to the module being replaced.
        new_module: Replacement to assign at that path.

    Raises:
        AttributeError: If an intermediate path segment does not exist.
    """
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


__all__ = ["replace_module"]
