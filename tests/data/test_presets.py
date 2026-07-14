"""Tests for the data preset module (T3 #28).

Covers:

- :class:`DatasetPreset` construction + immutability
- Built-in presets: ``C4_PRESET``, ``THEPILE_PRESET``,
  ``REDPAJAMA_PRESETS``
- :func:`apply_to_config` mutates only the four HF fields
- :func:`resolve_preset` happy path + unknown name error +
  RedPajama shorthand
- :func:`list_presets` enumerator
- Soft-dependency boundary: presets module imports cleanly without
  the ``datasets`` package installed (just a regression check on
  the import shape).
"""

from __future__ import annotations

import pytest

from llm.data.presets import (
    BUILTIN_PRESETS,
    C4_PRESET,
    REDPAJAMA_PRESETS,
    THEPILE_PRESET,
    DatasetPreset,
    apply_to_config,
    list_presets,
    resolve_preset,
)
from llm.training.core.config import DataConfig


# --- DatasetPreset dataclass -----------------------------------------------


def test_datasetpreset_construction_minimal():
    """Only ``dataset_name`` is required."""
    preset = DatasetPreset(dataset_name="allenai/c4")
    assert preset.dataset_name == "allenai/c4"
    assert preset.dataset_config is None
    assert preset.dataset_split == "train"
    assert preset.text_column == "text"
    assert preset.description == ""
    # ``preset_name`` defaults to ``dataset_name`` (filled in __post_init__).
    assert preset.preset_name == "allenai/c4"


def test_datasetpreset_construction_full():
    preset = DatasetPreset(
        dataset_name="togethercomputer/RedPajama-Data-1T",
        dataset_config="wikipedia",
        dataset_split="train",
        text_column="raw_content",
        description="RedPajama Wikipedia subset",
        preset_name="redpajama/wikipedia",
    )
    assert preset.dataset_name == "togethercomputer/RedPajama-Data-1T"
    assert preset.dataset_config == "wikipedia"
    assert preset.text_column == "raw_content"
    assert preset.description == "RedPajama Wikipedia subset"
    assert preset.preset_name == "redpajama/wikipedia"


def test_datasetpreset_is_frozen():
    """``frozen=True`` prevents accidental preset mutation."""
    preset = DatasetPreset(dataset_name="allenai/c4")
    with pytest.raises(Exception):
        preset.dataset_name = "other"  # type: ignore[misc]


def test_datasetpreset_preset_name_defaults_to_dataset_name():
    """When ``preset_name`` is omitted, it falls back to ``dataset_name``."""
    preset = DatasetPreset(dataset_name="my-org/my-dataset")
    assert preset.preset_name == "my-org/my-dataset"


def test_datasetpreset_preset_name_keeps_explicit_value():
    """Explicit ``preset_name`` survives ``__post_init__``."""
    preset = DatasetPreset(
        dataset_name="allenai/c4", preset_name="c4-short"
    )
    assert preset.preset_name == "c4-short"


# --- Built-in presets ------------------------------------------------------


def test_c4_preset_basics():
    assert C4_PRESET.dataset_name == "allenai/c4"
    assert C4_PRESET.dataset_config == "en"
    assert C4_PRESET.text_column == "text"


def test_thepile_preset_basics():
    assert THEPILE_PRESET.dataset_name == "monology/pile-uncopyrighted"
    # The Pile has no subset/config; ``dataset_config`` stays ``None``.
    assert THEPILE_PRESET.dataset_config is None
    assert THEPILE_PRESET.text_column == "text"


def test_redpajama_presets_subset_count():
    """RedPajama ships seven well-known subsets in the dict."""
    assert len(REDPAJAMA_PRESETS) == 7
    expected_subsets = {
        "redpajama/arxiv",
        "redpajama/book",
        "redpajama/common_crawl",
        "redpajama/c4",
        "redpajama/github",
        "redpajama/stackexchange",
        "redpajama/wikipedia",
    }
    assert set(REDPAJAMA_PRESETS) == expected_subsets


def test_redpajama_presets_share_dataset_name():
    """All RedPajama presets point at the same dataset, differing only in config."""
    for preset in REDPAJAMA_PRESETS.values():
        assert preset.dataset_name == "togethercomputer/RedPajama-Data-1T"
        assert preset.text_column == "text"


def test_redpajama_presets_keys_match_preset_names():
    """Dict keys must equal each preset's ``preset_name`` so the dict spreads cleanly into ``BUILTIN_PRESETS``."""
    for key, preset in REDPAJAMA_PRESETS.items():
        assert key == preset.preset_name
        assert key.startswith("redpajama/")


def test_builtin_presets_index_includes_every_preset():
    """BUILTIN_PRESETS aggregates C4 + The Pile + all RedPajama subsets."""
    assert BUILTIN_PRESETS[C4_PRESET.preset_name] is C4_PRESET
    assert BUILTIN_PRESETS[THEPILE_PRESET.preset_name] is THEPILE_PRESET
    for subset_name, preset in REDPAJAMA_PRESETS.items():
        assert BUILTIN_PRESETS[subset_name] is preset


def test_list_presets_returns_sorted_list():
    """``list_presets()`` returns all built-ins in sorted name order."""
    presets = list_presets()
    assert len(presets) == 1 + 1 + len(REDPAJAMA_PRESETS)
    names = [p.preset_name for p in presets]
    assert names == sorted(names)


# --- apply_to_config -------------------------------------------------------


def test_apply_to_config_fills_hf_fields():
    cfg = DataConfig(data_source="local", max_seq_len=2048)
    apply_to_config(cfg, C4_PRESET)
    assert cfg.data_source == "hf"
    assert cfg.dataset_name == "allenai/c4"
    assert cfg.dataset_config == "en"
    assert cfg.dataset_split == "train"
    assert cfg.text_column == "text"


def test_apply_to_config_preserves_unrelated_fields():
    """``max_seq_len`` / ``tokenizer_*`` / ``val_dataset_path`` must survive."""
    cfg = DataConfig(
        data_source="local",
        max_seq_len=4096,
        tokenizer_type="hf",
        tokenizer_path="gpt2",
        val_dataset_path="/tmp/val.txt",
    )
    apply_to_config(cfg, C4_PRESET)
    assert cfg.max_seq_len == 4096
    assert cfg.tokenizer_type == "hf"
    assert cfg.tokenizer_path == "gpt2"
    assert cfg.val_dataset_path == "/tmp/val.txt"


def test_apply_to_config_returns_config_for_fluent_use():
    cfg = DataConfig(data_source="local")
    out = apply_to_config(cfg, THEPILE_PRESET)
    assert out is cfg
    assert cfg.dataset_name == "monology/pile-uncopyrighted"


def test_apply_to_config_overrides_dataset_config_for_no_subset_datasets():
    """The Pile has ``dataset_config=None`` — applying must clear any prior config."""
    cfg = DataConfig(data_source="hf", dataset_name="old", dataset_config="en")
    apply_to_config(cfg, THEPILE_PRESET)
    assert cfg.dataset_config is None
    assert cfg.dataset_name == "monology/pile-uncopyrighted"


def test_apply_to_config_redpajama_subset():
    preset = REDPAJAMA_PRESETS["redpajama/wikipedia"]
    cfg = DataConfig(data_source="local")
    apply_to_config(cfg, preset)
    assert cfg.dataset_name == "togethercomputer/RedPajama-Data-1T"
    assert cfg.dataset_config == "wikipedia"


def test_apply_to_config_rejects_non_dataconfig():
    """A non-DataConfig object must raise ``TypeError`` rather than crash on attr access."""
    with pytest.raises(TypeError) as exc_info:
        apply_to_config("not a config", C4_PRESET)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "data_source" in msg


# --- resolve_preset --------------------------------------------------------


def test_resolve_preset_by_preset_name():
    assert resolve_preset("allenai/c4") is C4_PRESET
    assert resolve_preset("monology/pile-uncopyrighted") is THEPILE_PRESET


def test_resolve_preset_redpajama_via_canonical_key():
    assert resolve_preset("redpajama/c4") is REDPAJAMA_PRESETS["redpajama/c4"]
    assert (
        resolve_preset("redpajama/wikipedia")
        is REDPAJAMA_PRESETS["redpajama/wikipedia"]
    )


def test_resolve_preset_redpajama_via_shorthand():
    """``"redpajama:github"`` and ``"redpajama/github"`` both resolve."""
    assert (
        resolve_preset("redpajama:github")
        is REDPAJAMA_PRESETS["redpajama/github"]
    )
    assert (
        resolve_preset("redpajama/arxiv")
        is REDPAJAMA_PRESETS["redpajama/arxiv"]
    )


def test_resolve_preset_unknown_raises_with_available_list():
    with pytest.raises(KeyError) as exc_info:
        resolve_preset("does-not-exist")
    msg = str(exc_info.value)
    assert "does-not-exist" in msg
    # The error message lists at least one known preset name so users
    # can self-correct from the error alone.
    assert "redpajama/c4" in msg


def test_resolve_preset_fallback_by_dataset_name():
    """Looking up by the full HF dataset id also works (returns first match by iteration order).

    All RedPajama presets share the same ``dataset_name``; the fallback
    returns the first preset in iteration order. We don't pin which
    one — only that *some* preset with the matching dataset name
    comes back.
    """
    preset = resolve_preset("togethercomputer/RedPajama-Data-1T")
    assert preset.dataset_name == "togethercomputer/RedPajama-Data-1T"
    assert preset in REDPAJAMA_PRESETS.values()


def test_resolve_preset_redpajama_empty_subset_raises():
    """``"redpajama/"`` (empty subset) must NOT silently match the first preset."""
    with pytest.raises(KeyError):
        resolve_preset("redpajama/")


# --- Soft dependency boundary (regression) --------------------------------


def test_presets_module_does_not_import_datasets():
    """``llm.data.presets`` must be importable without ``datasets``.

    Without this guard, every preset import pulls in the HF
    ``datasets`` package transitively, breaking CPU-only hosts
    that only want to look up preset names.
    """
    import llm.data.presets as presets_module

    # The module should NOT have ``datasets`` or ``streaming`` in its
    # module-level imports. Check by name.
    assert not hasattr(presets_module, "datasets"), (
        "presets.py must not import the 'datasets' package — keep the "
        "soft-dependency boundary so the module loads without it."
    )
