"""Pluggable text sources for streaming data pipelines."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class TextSource(abc.ABC):
    """Abstract source of text records for streaming datasets."""

    @abc.abstractmethod
    def iter_texts(self) -> Iterator[str]:
        """Yield non-empty text records."""
        pass


class LocalLineTextSource(TextSource):
    """Stream UTF-8 text line-by-line from a local file."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def iter_texts(self) -> Iterator[str]:
        with self.file_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


class HFStreamTextSource(TextSource):
    """Stream text from a HuggingFace dataset in streaming mode."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        dataset_config: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.dataset_config = dataset_config

    def iter_texts(self) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HF streaming requires the 'datasets' package. Install with: uv sync --group streaming"
            ) from exc

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        for row in dataset:
            text = row.get(self.text_column)
            if isinstance(text, str) and text.strip():
                yield text.strip()


def build_text_source(data_config: Any) -> TextSource:
    """Factory: resolve TextSource from DataConfig."""
    source_type = getattr(data_config, "data_source", "local")

    if source_type == "hf":
        dataset_name = getattr(data_config, "dataset_name", None)
        if not dataset_name:
            raise ValueError("data.dataset_name is required when data_source='hf'")
        return HFStreamTextSource(
            dataset_name=dataset_name,
            split=getattr(data_config, "dataset_split", "train"),
            text_column=getattr(data_config, "text_column", "text"),
            dataset_config=getattr(data_config, "dataset_config", None),
        )

    if not data_config.dataset_path:
        raise ValueError("data.dataset_path is required when data_source='local'")
    return LocalLineTextSource(data_config.dataset_path)
