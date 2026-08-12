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


def model_vocab_size(model: nn.Module) -> int | None:
    """Return the model's vocab size (embedding row count), or ``None``.

    Trace-based exporters build a random token-id dummy input and must
    bound it by the REAL vocabulary — a hardcoded ``randint(0, 100)``
    crashes with ``IndexError`` inside the embedding for any model with
    ``vocab_size < 100`` (RIL ISS-058). This helper resolves the vocab
    from the common embedding layouts (``DecoderModel`` and friends);
    returns ``None`` when the model exposes no recognizable embedding so
    callers keep their historical default.
    """
    embedding = getattr(model, "embedding_layer", None)
    token_embeddings = getattr(embedding, "token_embeddings", None)
    num_embeddings = getattr(token_embeddings, "num_embeddings", None)
    if isinstance(num_embeddings, int):
        return num_embeddings
    # Fallbacks for models that expose an HF-style ``get_input_embeddings``.
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            emb = getter()
        except Exception:  # noqa: BLE001 - defensive; vocab is best-effort
            return None
        num = getattr(emb, "num_embeddings", None)
        if isinstance(num, int):
            return num
    return None


def dummy_token_ids(
    model: nn.Module,
    shape: tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a random token-id dummy input bounded by the model's vocab.

    Args:
        model: The model being exported.
        shape: ``(batch_size, seq_len)``.
        device: Torch device for the tensor.

    Uses :func:`model_vocab_size`; falls back to the historical ``100``
    upper bound when the vocab can't be resolved (keeps existing
    behaviour for models without a discoverable embedding).
    """
    batch_size, seq_len = shape
    upper = model_vocab_size(model)
    if upper is None:
        upper = 100
    # ``randint(0, upper)`` yields ids in ``[0, upper)`` — always < vocab.
    return torch.randint(0, upper, (batch_size, seq_len), device=device)


__all__ = ["ExportCacheWrapper", "dummy_token_ids", "model_vocab_size"]
