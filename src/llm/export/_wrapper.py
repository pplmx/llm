"""Shared helpers for export backends.

Right now the only shared piece is the cache-contract wrapper used
by every trace-based export target (``torch.onnx.export``,
``torch.jit.trace``). Both exporters need the model to be called
with ``use_cache=False`` and to return a single tensor so the
tracer doesn't record KV-cache boolean conditionals or shape
expressions.

This module is intentionally tiny — it only holds what two or
more backends need. Anything specific to a single backend stays
in that backend's file.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExportCacheWrapper(nn.Module):
    """Wrap a model so trace-based exporters see a clean contract.

    Forces ``use_cache=False`` (avoiding KV-cache tracer branching)
    and unwraps the ``(logits, kv_cache)`` tuple to just ``logits``
    so the traced graph's output is a single tensor.

    The class is shared across every trace-based backend. ``script``
    backends don't need it, but using it is harmless — the wrapper
    is just a thin ``nn.Module`` subclass.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, use_cache=False)
        if isinstance(output, tuple):
            return output[0]
        return output


__all__ = ["ExportCacheWrapper"]
