"""Tests for ``ServingGenerationService`` backend wiring.

Regression for RIL ISS-150: ``generation_backend=batched`` must be able to
start the REST server. ``api.lifespan`` constructs the service *before*
the engine, so ``from_config`` had to build the batched engine itself —
otherwise ``get_generation_backend("batched", engine=None)`` raised
``ValueError`` at startup and the documented config option could never
serve.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm.serving.config import ServingConfig
from llm.serving.generation_service import ServingGenerationService


@pytest.fixture
def batched_config(device):
    return ServingConfig(
        api_key="test-key",
        device=str(device),
        generation_backend="batched",
    )


def test_from_config_batched_builds_engine_itself(batched_config):
    """``from_config`` with backend=batched and no engine must not raise —
    it builds a ContinuousBatchingEngine from the loaded model."""
    dummy_model = MagicMock()
    dummy_tokenizer = object()

    with (
        patch(
            "llm.serving.generation_service.load_model_and_tokenizer",
            return_value=(dummy_model, dummy_tokenizer),
        ),
        patch(
            "llm.serving.batch_engine.ContinuousBatchingEngine.from_serving_config",
            return_value=MagicMock(),
        ) as mock_engine,
    ):
        service = ServingGenerationService.from_config(batched_config)

    mock_engine.assert_called_once()
    assert getattr(service.backend, "engine", None) is not None


def test_from_config_batched_passes_through_explicit_engine(batched_config):
    """An externally supplied engine is honoured (not clobbered by a new one)."""
    dummy_model = MagicMock()
    dummy_tokenizer = object()
    external_engine = MagicMock()
    external_engine.model = dummy_model

    with (
        patch(
            "llm.serving.generation_service.load_model_and_tokenizer",
            return_value=(dummy_model, dummy_tokenizer),
        ) as mock_load,
        patch(
            "llm.serving.batch_engine.ContinuousBatchingEngine.from_serving_config",
        ) as mock_engine_builder,
    ):
        service = ServingGenerationService.from_config(
            batched_config,
            engine=external_engine,
        )

    mock_engine_builder.assert_not_called()
    mock_load.assert_called_once()
    assert service.backend.engine is external_engine
