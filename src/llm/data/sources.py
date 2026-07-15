"""Pluggable text sources for streaming data pipelines."""

from __future__ import annotations

import abc
import hashlib
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from llm.runtime.plugins import load_entry_point_registry
from llm.runtime.registry import Registry

TextSourceBuilder = Callable[[Any], "TextSource"]

SOURCE_REGISTRY: Registry[TextSourceBuilder] = Registry("TextSource")

_sources_registered = False


class TextSource(abc.ABC):
    """Abstract source of text records for streaming datasets."""

    @abc.abstractmethod
    def iter_texts(self, skip: int = 0) -> Iterator[str]:
        """Yield non-empty text records, optionally skipping the first ``skip`` records."""
        pass

    def source_fingerprint(self) -> dict[str, Any]:
        """Return metadata for validating checkpoint resume against the same source."""
        return {"type": self.__class__.__name__}


class LocalLineTextSource(TextSource):
    """Stream UTF-8 text line-by-line from a local file."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def source_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "local",
            "dataset_path": str(self.file_path.resolve()),
        }

    def iter_texts(self, skip: int = 0) -> Iterator[str]:
        skipped = 0
        with self.file_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if skipped < skip:
                    skipped += 1
                    continue
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

    def source_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "hf",
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_split": self.split,
            "text_column": self.text_column,
        }

    def iter_texts(self, skip: int = 0) -> Iterator[str]:
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
        if skip > 0:
            dataset = dataset.skip(skip)

        for row in dataset:
            text = row.get(self.text_column)
            if isinstance(text, str) and text.strip():
                yield text.strip()


_WHITESPACE_RE = re.compile(r"\s+")


def _default_dedup_normalize(text: str) -> str:
    """Strip and collapse internal whitespace runs.

    Intentionally case-sensitive: case-only differences carry semantic
    meaning ("Apple" vs "apple"), and conflating them would silently
    drop legitimate records.
    """
    return _WHITESPACE_RE.sub(" ", text.strip())


class DedupTextSource(TextSource):
    """TextSource wrapper that drops duplicate records by content hash.

    Useful for pretraining data preparation where web-crawl-derived
    corpora contain substantial exact duplicates. The wrapper:

    - hashes the normalized text and drops records whose hash has
      already been yielded this run;
    - optionally loads a pre-populated "seen hashes" file on
      construction so dedup state is shared across runs / shards;
    - optionally appends new hashes to that file so dedup state grows
      monotonically;
    - exposes a stable :meth:`source_fingerprint` that includes the
      inner source's fingerprint plus the dedup strategy, so
      :func:`validate_source_fingerprint` catches configuration drift
      on checkpoint resume.

    Args:
        inner: The wrapped source. Records yielded by
            ``inner.iter_texts`` flow through the dedup filter.
        normalize: Optional callable that normalizes text before
            hashing. Default: :func:`_default_dedup_normalize` (strip +
            collapse internal whitespace runs). **Case-sensitive by
            default.**
        seen_hashes_path: Optional path to a file containing previously
            seen hashes (one per line, hex-encoded). If the file exists
            when the wrapper is constructed, its contents are loaded
            into the seen-set so dedup state survives across runs.
        write_seen_hashes: If True, append new hashes to
            ``seen_hashes_path`` as they are discovered. Requires
            ``seen_hashes_path``. Defaults to False.
        hash_algo: Name of any algorithm accepted by :func:`hashlib.new`
            (e.g. ``"sha256"``, ``"sha1"``, ``"md5"``). Default: SHA-256.

    Example:
        >>> src = LocalLineTextSource("data.txt")
        >>> dedup = DedupTextSource(src, seen_hashes_path="seen.txt")
        >>> unique_texts = list(dedup.iter_texts())
    """

    def __init__(
        self,
        inner: TextSource,
        *,
        normalize: Callable[[str], str] | None = None,
        seen_hashes_path: str | Path | None = None,
        write_seen_hashes: bool = False,
        hash_algo: str = "sha256",
    ):
        self.inner = inner
        self.normalize = normalize if normalize is not None else _default_dedup_normalize
        self.seen_hashes_path = Path(seen_hashes_path) if seen_hashes_path is not None else None
        self.write_seen_hashes = write_seen_hashes
        self.hash_algo = hash_algo
        if write_seen_hashes and self.seen_hashes_path is None:
            raise ValueError("write_seen_hashes=True requires seen_hashes_path to be set")
        # ``hashlib.new`` raises ValueError synchronously for unknown
        # algos; fail fast at construction time so users see the error
        # before iterating.
        hashlib.new(self.hash_algo)
        self._seen: set[str] = set()
        self._load_seen_hashes()

    def _load_seen_hashes(self) -> None:
        if self.seen_hashes_path is None or not self.seen_hashes_path.exists():
            return
        with self.seen_hashes_path.open(encoding="utf-8") as handle:
            self._seen.update(line.strip() for line in handle if line.strip())

    def iter_texts(self, skip: int = 0) -> Iterator[str]:
        # ``skip`` is delegated to the inner source so the
        # ``line_index`` resume semantics used by StreamingTextDataset
        # stay consistent with non-dedup sources.
        for text in self.inner.iter_texts(skip=skip):
            normalized = self.normalize(text)
            digest = hashlib.new(self.hash_algo, normalized.encode("utf-8")).hexdigest()
            if digest in self._seen:
                continue
            self._seen.add(digest)
            if self.write_seen_hashes and self.seen_hashes_path is not None:
                with self.seen_hashes_path.open("a", encoding="utf-8") as handle:
                    handle.write(digest + "\n")
            yield text

    def source_fingerprint(self) -> dict[str, Any]:
        fp: dict[str, Any] = {
            "type": "dedup",
            "inner": self.inner.source_fingerprint(),
            "hash_algo": self.hash_algo,
            "normalize": getattr(self.normalize, "__name__", repr(self.normalize)),
        }
        if self.seen_hashes_path is not None:
            fp["seen_hashes_path"] = str(self.seen_hashes_path.resolve())
        return fp


def _build_local_source(data_config: Any) -> TextSource:
    if not data_config.dataset_path:
        raise ValueError("data.dataset_path is required when data_source='local'")
    return LocalLineTextSource(data_config.dataset_path)


def _build_hf_source(data_config: Any) -> TextSource:
    dataset_name = getattr(data_config, "dataset_name", None)
    if not dataset_name:
        raise ValueError("data.dataset_name is required when data_source='hf'")
    return HFStreamTextSource(
        dataset_name=dataset_name,
        split=getattr(data_config, "dataset_split", "train"),
        text_column=getattr(data_config, "text_column", "text"),
        dataset_config=getattr(data_config, "dataset_config", None),
    )


def _resolve_dedup_kwargs(data_config: Any) -> dict[str, Any]:
    """Pull the dedup-related fields off a DataConfig with safe defaults.

    Returns an empty dict when none of the fields are set so the
    underlying :class:`DedupTextSource` falls back to in-memory exact
    dedup with no on-disk persistence.
    """
    kwargs: dict[str, Any] = {}
    seen_hashes_path = getattr(data_config, "seen_hashes_path", None)
    if seen_hashes_path is not None:
        kwargs["seen_hashes_path"] = seen_hashes_path
    write_seen_hashes = getattr(data_config, "write_seen_hashes", False)
    if write_seen_hashes:
        kwargs["write_seen_hashes"] = True
    hash_algo = getattr(data_config, "hash_algo", None)
    if hash_algo:
        kwargs["hash_algo"] = hash_algo
    return kwargs


def _build_dedup_local_source(data_config: Any) -> TextSource:
    if not getattr(data_config, "dataset_path", None):
        raise ValueError("data.dataset_path is required when data_source='dedup_local'")
    return DedupTextSource(
        LocalLineTextSource(data_config.dataset_path),
        **_resolve_dedup_kwargs(data_config),
    )


def _build_dedup_hf_source(data_config: Any) -> TextSource:
    dataset_name = getattr(data_config, "dataset_name", None)
    if not dataset_name:
        raise ValueError("data.dataset_name is required when data_source='dedup_hf'")
    return DedupTextSource(
        HFStreamTextSource(
            dataset_name=dataset_name,
            split=getattr(data_config, "dataset_split", "train"),
            text_column=getattr(data_config, "text_column", "text"),
            dataset_config=getattr(data_config, "dataset_config", None),
        ),
        **_resolve_dedup_kwargs(data_config),
    )


def ensure_sources_registered() -> None:
    global _sources_registered
    if _sources_registered:
        return

    # Built-in dedup variants are registered explicitly rather than
    # via setuptools entry points so a fresh checkout can use them
    # without an editable install of an external plugin. Third-party
    # dedup strategies (MinHash / SimHash / ...) are still welcome via
    # the ``llm.data_sources`` entry-point group.
    SOURCE_REGISTRY.register("dedup_local", _build_dedup_local_source)
    SOURCE_REGISTRY.register("dedup_hf", _build_dedup_hf_source)

    load_entry_point_registry("llm.data_sources", SOURCE_REGISTRY)
    _sources_registered = True


def build_text_source(data_config: Any) -> TextSource:
    """Resolve TextSource from DataConfig via SOURCE_REGISTRY."""
    ensure_sources_registered()
    source_type = getattr(data_config, "data_source", "local")
    return SOURCE_REGISTRY.get(source_type)(data_config)


def source_fingerprint_from_config(data_config: Any) -> dict[str, Any]:
    """Build a stable fingerprint for the configured text source without loading data."""
    return build_text_source(data_config).source_fingerprint()


def validate_source_fingerprint(expected: dict[str, Any] | None, actual: dict[str, Any]) -> None:
    """Raise if checkpoint source metadata does not match the active DataModule config."""
    if not expected:
        return
    if expected != actual:
        raise ValueError(
            "Streaming checkpoint source fingerprint mismatch. "
            f"expected={expected}, actual={actual}. "
            "Use the same dataset configuration when resuming."
        )
