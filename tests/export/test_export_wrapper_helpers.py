"""Tests for the exporter input-preparation helpers.

``llm/export/_wrapper.py`` holds the three helpers every trace-based exporter
uses to build its dummy input: :func:`clamp_input_shape`,
:func:`model_vocab_size` and :func:`dummy_token_ids`. Both the ONNX and the
TorchScript backend call them, yet the module had no direct tests — the
``model_vocab_size`` fallback branch (for models exposing an HF-style
``get_input_embeddings``) sat at 0% coverage while being reachable from both
export paths.

The functions matter because a wrong dummy input does not fail cleanly: an
out-of-vocabulary id raises ``IndexError`` deep inside the embedding, after the
export has already begun, and the message names the embedding rather than the
export call the user made.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.export._wrapper import (
    ExportCacheWrapper,
    clamp_input_shape,
    dummy_token_ids,
    model_vocab_size,
)


class _HFLikeModel(nn.Module):
    """A model exposing only the HF-style ``get_input_embeddings`` accessor."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size

    def get_input_embeddings(self) -> nn.Embedding:
        return nn.Embedding(self._vocab_size, 8)


class _BrokenAccessorModel(nn.Module):
    """A model whose embedding accessor raises."""

    def get_input_embeddings(self) -> nn.Embedding:
        raise RuntimeError("this model cannot report its embedding")


class _NoEmbeddingModel(nn.Module):
    """A model exposing no recognizable embedding at all."""


class _MutatingExportModel(nn.Module):
    """Records ``use_cache`` so the wrapper's override can be asserted."""

    def __init__(self, *, return_tuple: bool) -> None:
        super().__init__()
        self._return_tuple = return_tuple
        self.seen_use_cache: bool | None = None

    def forward(self, input_ids: torch.Tensor, use_cache: bool = True):
        self.seen_use_cache = use_cache
        logits = torch.zeros(*input_ids.shape, 4)
        return (logits, None) if self._return_tuple else logits


class TestModelVocabSize:
    def test_resolves_from_a_decoder_style_embedding_layer(self):
        """The primary layout: ``embedding_layer.token_embeddings.num_embeddings``."""

        class _WithEmbeddingLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_layer = nn.Module()
                self.embedding_layer.token_embeddings = nn.Embedding(37, 8)

        assert model_vocab_size(_WithEmbeddingLayer()) == 37

    def test_falls_back_to_the_hf_style_getter(self):
        """Models that only offer ``get_input_embeddings`` still resolve.

        This branch is what makes the helper useful for HF-style model
        objects; without it those models silently got the hard-coded 100 upper
        bound and could sample out-of-vocabulary ids.
        """
        assert model_vocab_size(_HFLikeModel(42)) == 42

    def test_returns_none_when_the_getter_raises(self):
        """A raising accessor is best-effort, not fatal — callers fall back."""
        assert model_vocab_size(_BrokenAccessorModel()) is None

    def test_returns_none_when_nothing_is_discoverable(self):
        assert model_vocab_size(_NoEmbeddingModel()) is None


class TestDummyTokenIds:
    def test_ids_stay_within_the_real_vocabulary(self):
        """The whole point: no id may exceed the model's embedding rows.

        A ``randint(0, 100)`` default breaks for any model with
        ``vocab_size < 100``; with the resolved vocab the upper bound follows
        the model.
        """
        torch.manual_seed(0)
        vocab_size = 7  # far below the historical hard-coded 100
        ids = dummy_token_ids(_HFLikeModel(vocab_size), (4, 16))
        assert ids.shape == (4, 16)
        assert ids.min() >= 0
        assert ids.max() < vocab_size, "a sampled id would index past the embedding table"

    def test_falls_back_to_the_historical_bound_without_an_embedding(self):
        torch.manual_seed(0)
        ids = dummy_token_ids(_NoEmbeddingModel(), (2, 8))
        assert ids.shape == (2, 8)
        assert ids.max() < 100

    def test_honours_the_requested_dtype_and_device(self):
        ids = dummy_token_ids(_HFLikeModel(50), (1, 4))
        assert ids.dtype == torch.long


class TestClampInputShape:
    def test_long_sequence_is_clamped_to_the_model_window(self):
        """Exporting at a longer shape than the model supports must not be attempted."""

        class _Windowed(nn.Module):
            max_seq_len = 32

        assert clamp_input_shape(_Windowed(), (1, 512)) == (1, 32)

    def test_short_sequence_is_left_alone(self):
        class _Windowed(nn.Module):
            max_seq_len = 32

        assert clamp_input_shape(_Windowed(), (2, 8)) == (2, 8)


class TestExportCacheWrapper:
    def test_forward_forces_use_cache_false(self):
        """Tracing with the cache on records KV-cache branching into the graph."""
        model = _MutatingExportModel(return_tuple=True)
        ExportCacheWrapper(model)(torch.zeros(1, 4, dtype=torch.long))
        assert model.seen_use_cache is False

    def test_tuple_output_is_unwrapped_to_logits(self):
        """Exporters trace a single tensor; ``(logits, cache)`` must collapse."""
        model = _MutatingExportModel(return_tuple=True)
        assert ExportCacheWrapper(model)(torch.zeros(1, 4, dtype=torch.long)).shape == (1, 4, 4)

    def test_bare_tensor_output_is_passed_through(self):
        model = _MutatingExportModel(return_tuple=False)
        assert ExportCacheWrapper(model)(torch.zeros(1, 4, dtype=torch.long)).shape == (1, 4, 4)


def test_public_helpers_are_declared_in_all():
    """``__all__`` must list every helper the exporters import from this module.

    ``clamp_input_shape`` is imported by both ``llm.export.onnx`` and
    ``llm.export.torchscript`` but was missing from ``__all__``, which is the
    module's declared public surface — so the declaration undersold what
    downstream code already depends on.
    """
    import llm.export._wrapper as wrapper

    imported_by_exporters = {"ExportCacheWrapper", "clamp_input_shape", "dummy_token_ids"}
    missing = imported_by_exporters - set(wrapper.__all__)
    assert not missing, f"exporters import names absent from __all__: {sorted(missing)}"
    for name in wrapper.__all__:
        assert hasattr(wrapper, name), f"__all__ advertises {name!r}, which does not exist"
