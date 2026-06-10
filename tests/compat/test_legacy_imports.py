"""Verify deprecated import shims still resolve."""

import warnings

import pytest


@pytest.mark.parametrize(
    ("import_stmt", "attr"),
    [
        ("from llm.data.data_module import MapDataModule", "MapDataModule"),
        ("from llm.data.loader import TextDataset", "TextDataset"),
        ("from llm.training.registry import TASK_REGISTRY", "TASK_REGISTRY"),
        ("from llm.serving.engine import ContinuousBatchingEngine", "ContinuousBatchingEngine"),
    ],
)
def test_legacy_import_shims_emit_deprecation(import_stmt: str, attr: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        namespace: dict = {}
        exec(import_stmt, namespace)
        assert attr in namespace
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
