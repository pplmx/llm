"""Built-in data presets for common pretraining datasets.

The project already ships a streaming data pipeline
(:class:`llm.data.modules.streaming.StreamingTextDataModule` +
:class:`llm.data.sources.HFStreamTextSource`), but every well-known
dataset requires hand-authoring the ``DataConfig`` triple
(``dataset_name``, ``dataset_config``, ``text_column``). This module
ships those triples out of the box so users can pick a dataset by
name instead of looking up the HF identifier every time.

The presets are intentionally **decoupled from the ``datasets``
package**: this module imports nothing from
``llm.data.datasets`` or ``llm.data.modules``, only
:class:`llm.training.core.config.DataConfig`. That keeps the import
cheap on hosts that don't have ``datasets`` installed.

Example:
    >>> from llm.training.core.config import DataConfig
    >>> from llm.data.presets import C4_PRESET, apply_to_config
    >>> cfg = DataConfig(data_source="hf", max_seq_len=2048)
    >>> apply_to_config(cfg, C4_PRESET)
    >>> cfg.dataset_name
    'allenai/c4'
    >>> cfg.dataset_config
    'en'
    >>> cfg.text_column
    'text'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetPreset:
    """A well-known HuggingFace dataset configuration.

    Attributes:
        dataset_name: HuggingFace dataset identifier (e.g.
            ``"allenai/c4"``).
        dataset_config: HF dataset config name (subset /
            ``revision``); ``None`` when the dataset has no subsets.
        dataset_split: Split to stream. ``"train"`` is the default
            for all built-in presets.
        text_column: Name of the text field in each row. Most
            English-text datasets use ``"text"``; RedPajama's
            Wikipedia subset uses ``"raw_content"``.
        description: Human-readable one-liner for CLI / docs.
        preset_name: Canonical short name (lowercase, kebab-case)
            used by :func:`resolve_preset`. Defaults to the
            ``dataset_name`` when not provided.
    """

    dataset_name: str
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    description: str = ""
    preset_name: str = ""

    def __post_init__(self) -> None:
        # ``frozen=True`` + ``field(default=...)`` works for mutable
        # defaults, but a derived string default has to be assigned
        # via ``object.__setattr__`` because we can't mutate ``self``
        # normally. We do it here so callers don't have to repeat
        # the dataset name as the preset name.
        if not self.preset_name:
            object.__setattr__(self, "preset_name", self.dataset_name)


# --- Built-in presets ------------------------------------------------------


C4_PRESET = DatasetPreset(
    dataset_name="allenai/c4",
    dataset_config="en",
    text_column="text",
    description="C4 (Colossal Clean Crawled Corpus) — English web crawl, 'en' config.",
)


THEPILE_PRESET = DatasetPreset(
    dataset_name="monology/pile-uncopyrighted",
    dataset_config=None,
    text_column="text",
    description="The Pile (uncopyrighted subset) — 825 GB English text corpus.",
)


# RedPajama-Data-1T is split into seven well-known subsets, each
# served as a separate HF config. We expose them as a dict keyed by
# the preset name (``"redpajama/arxiv"`` etc.) so callers can pick
# by subset without re-typing the ``togethercomputer/RedPajama-Data-1T``
# id. The dict key equals ``DatasetPreset.preset_name`` so
# :data:`BUILTIN_PRESETS` (which spreads this dict) stays consistent.
_REDPAJAMA_DATASET = "togethercomputer/RedPajama-Data-1T"


REDPAJAMA_PRESETS: dict[str, DatasetPreset] = {
    f"redpajama/{subset}": DatasetPreset(
        dataset_name=_REDPAJAMA_DATASET,
        dataset_config=subset,
        text_column="text",
        preset_name=f"redpajama/{subset}",
        description=f"RedPajama-Data-1T — {subset} subset.",
    )
    for subset in (
        "arxiv",
        "book",
        "common_crawl",
        "c4",
        "github",
        "stackexchange",
        "wikipedia",
    )
}


BUILTIN_PRESETS: dict[str, DatasetPreset] = {
    C4_PRESET.preset_name: C4_PRESET,
    THEPILE_PRESET.preset_name: THEPILE_PRESET,
    **REDPAJAMA_PRESETS,
}


# --- Application helpers ---------------------------------------------------


def apply_to_config(config: Any, preset: DatasetPreset) -> Any:
    """Mutate ``config`` (a :class:`DataConfig`) to bind to ``preset``.

    Sets ``data_source="hf"`` and the four HF fields
    (``dataset_name``, ``dataset_config``, ``dataset_split``,
    ``text_column``). Unrelated fields (``max_seq_len``,
    ``tokenizer_*``, ``val_dataset_path``, …) are left untouched.

    The mutated ``config`` is returned for fluent use:

    .. code-block:: python

        cfg = apply_to_config(DataConfig(...), C4_PRESET)

    Raises:
        TypeError: if ``config`` doesn't expose ``data_source`` as a
            writable attribute (i.e. it's not a ``DataConfig``).
    """
    if not hasattr(config, "data_source"):
        raise TypeError(
            f"apply_to_config expected a DataConfig with a 'data_source' attribute; got {type(config).__name__}"
        )
    config.data_source = "hf"
    config.dataset_name = preset.dataset_name
    config.dataset_config = preset.dataset_config
    config.dataset_split = preset.dataset_split
    config.text_column = preset.text_column
    return config


def resolve_preset(name: str) -> DatasetPreset:
    """Look up a preset by name.

    ``name`` may be:

    - the preset's canonical short name (``"c4"``, ``"the-pile"``,
      ``"redpajama/c4"`` …), or
    - the full HuggingFace dataset id (``"allenai/c4"``).

    Raises:
        KeyError: if no preset matches. The error message includes
            the available preset names so callers can self-correct.
    """
    # Direct preset-name lookup.
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]

    # RedPajama "subset" shorthand: ``"redpajama:arxiv"`` or
    # ``"redpajama/arxiv"`` resolves without the user having to know
    # the exact dict key shape.
    if ":" in name or "/" in name:
        for separator in (":", "/"):
            prefix, _, subset = name.partition(separator)
            if prefix.lower() == "redpajama" and subset:
                key = f"redpajama/{subset}"
                if key in BUILTIN_PRESETS:
                    return BUILTIN_PRESETS[key]
        # Fall through to the unknown-name error below.

    # Fallback: maybe they passed a dataset name directly.
    for preset in BUILTIN_PRESETS.values():
        if preset.dataset_name == name:
            return preset

    available = ", ".join(sorted(BUILTIN_PRESETS))
    raise KeyError(f"unknown data preset {name!r}; available built-ins: {available}")


def list_presets() -> list[DatasetPreset]:
    """Return all built-in presets in stable order (by preset name)."""
    return [BUILTIN_PRESETS[name] for name in sorted(BUILTIN_PRESETS)]
