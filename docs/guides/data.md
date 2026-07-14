# Data Pipelines Guide

This guide covers the streaming data layer for pretraining and the
**built-in dataset presets** that ship with the framework.

For everything related to *training loops* (optimizers, schedulers,
gradient accumulation) see the [training flow guide](../development/training-flow.md).
This page is about **what data you point your run at** and how the
framework's `DataConfig` knows where to look.

## The streaming pipeline in one diagram

```
                   ┌────────────────┐
                   │  DataConfig    │   ← pydantic model, one source of truth
                   │  data_source   │
                   │  dataset_name  │
                   │  max_seq_len   │
                   └───────┬────────┘
                           │
                build_text_source(config)
                           │
                           ▼
              ┌────────────────────────┐
              │     SOURCE_REGISTRY    │  ← plugin entry points
              │   (local | hf | …)     │
              └────────────┬───────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │  TextSource  │
                   │  iter_texts  │
                   └──────┬───────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │  StreamingTextDataModule   │   ← pretraining
             │  / other DataModules       │
             └────────────────────────────┘
```

`DataConfig` is the only thing the rest of the framework reads; the
source layer is pluggable so users can register new ones (S3, GCS,
custom archives) via the same `SOURCE_REGISTRY` plugin boundary.

## Built-in presets

The framework ships a handful of well-known datasets out of the box
so you don't have to look up the HF identifiers every time. Three
families are pre-wired:

| Preset | HF dataset | Config | Text column |
|---|---|---|---|
| `C4_PRESET` | `allenai/c4` | `en` | `text` |
| `THEPILE_PRESET` | `monology/pile-uncopyrighted` | _(none)_ | `text` |
| `REDPAJAMA_PRESETS["redpajama/arxiv"]` | `togethercomputer/RedPajama-Data-1T` | `arxiv` | `text` |
| `REDPAJAMA_PRESETS["redpajama/book"]` | `togethercomputer/RedPajama-Data-1T` | `book` | `text` |
| `REDPAJAMA_PRESETS["redpajama/common_crawl"]` | `togethercomputer/RedPajama-Data-1T` | `common_crawl` | `text` |
| `REDPAJAMA_PRESETS["redpajama/c4"]` | `togethercomputer/RedPajama-Data-1T` | `c4` | `text` |
| `REDPAJAMA_PRESETS["redpajama/github"]` | `togethercomputer/RedPajama-Data-1T` | `github` | `text` |
| `REDPAJAMA_PRESETS["redpajama/stackexchange"]` | `togethercomputer/RedPajama-Data-1T` | `stackexchange` | `text` |
| `REDPAJAMA_PRESETS["redpajama/wikipedia"]` | `togethercomputer/RedPajama-Data-1T` | `wikipedia` | `text` |

### Listing them programmatically

```python
from llm.data.presets import list_presets

for preset in list_presets():
    print(preset.preset_name, "→", preset.dataset_name)
```

### Resolving by name

```python
from llm.data.presets import resolve_preset

# All of these resolve to the same RedPajama Wikipedia preset:
resolve_preset("redpajama/wikipedia")
resolve_preset("redpajama:wikipedia")  # shorthand with ":"
resolve_preset("togethercomputer/RedPajama-Data-1T")  # full dataset id
```

Unknown names raise `KeyError` with the available list so the error
message alone is enough to self-correct.

## Applying a preset to a `DataConfig`

The :func:`llm.data.presets.apply_to_config` helper mutates a
`DataConfig` in place to bind it to a preset — it sets
`data_source="hf"` and fills the four HF fields
(`dataset_name`, `dataset_config`, `dataset_split`, `text_column`).
Unrelated fields (`max_seq_len`, `tokenizer_type`, validation paths)
are left untouched.

```python
from llm.training.core.config import DataConfig
from llm.data.presets import C4_PRESET, apply_to_config

cfg = DataConfig(max_seq_len=2048, tokenizer_type="hf", tokenizer_path="gpt2")
apply_to_config(cfg, C4_PRESET)

# Now cfg is:
#   data_source      = "hf"
#   dataset_name     = "allenai/c4"
#   dataset_config   = "en"
#   dataset_split    = "train"
#   text_column      = "text"
#   max_seq_len      = 2048         ← unchanged
#   tokenizer_type   = "hf"         ← unchanged
```

### YAML / CLI example

The same flow works when loading a config from YAML — apply the
preset *after* parsing, since presets only touch the four HF fields
and you'll typically want to keep your YAML focused on the training
harness (batch size, optimizer, etc.):

```yaml
# configs/c4-pretrain.yaml
data:
  data_source: hf            # overwritten by the preset below
  dataset_name: ""           # filled in by the preset
  max_seq_len: 2048
  tokenizer_type: hf
  tokenizer_path: gpt2

training:
  batch_size: 16
  # … (optimizer, scheduler, etc.)
```

```python
from llm.data.presets import C4_PRESET, apply_to_config
from llm.training.core.config import Config

cfg = Config.from_yaml("configs/c4-pretrain.yaml")
apply_to_config(cfg.data, C4_PRESET)  # in-place mutation
```

## Custom presets

If you want to bind the framework to a dataset that's not in the
built-in list (a private S3 dump, an in-house tokenized corpus,
etc.) just construct a :class:`DatasetPreset` directly:

```python
from llm.data.presets import DatasetPreset, apply_to_config

MY_CORPUS = DatasetPreset(
    dataset_name="my-org/private-corpus",
    dataset_config="v3",
    text_column="body",
    description="Internal R&D corpus, v3 snapshot.",
)
apply_to_config(cfg.data, MY_CORPUS)
```

The four HF fields are all you need for any HF-streaming-compatible
dataset.

## Source fingerprints and resume safety

`HFStreamTextSource` and `LocalLineTextSource` both expose a
`source_fingerprint()` method that returns a stable identity dict.
The streaming `DataModule` records this fingerprint in its
checkpoint state and validates it on resume, so accidentally
swapping `C4` for `The Pile` mid-run raises a clear error rather
than silently corrupting the data stream.

See `src/llm/data/sources.py:source_fingerprint_from_config` for
the implementation.

## Related

- [Evaluation guide](evaluation.md) — same preset pattern, applied to
  `lm-eval-harness` benchmarks instead of HF training datasets.
- [Training flow guide](../development/training-flow.md) — how the
  streaming `DataModule` plugs into the trainer loop.
- [Tier 3 ticket #8](../audits/2026-07-12-tickets/28-data-presets.md)
  — the audit follow-up that motivated this module.
