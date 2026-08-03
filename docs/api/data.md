# `llm.data` — Datasets, DataModules, and Sources

The `llm.data` package is split into three layers:

- **Sources** (`llm.data.sources`) — pluggable text iterators backed
  by local files or HuggingFace streaming. Plugins register into
  `SOURCE_REGISTRY`.
- **Datasets** (`llm.data.datasets`) — `IterableDataset` and `Map`
  wrappers that turn raw text into token chunks ready for the
  trainer.
- **DataModules** (`llm.data.modules`) — `Lightning`-style
  `setup` / `prepare_data` / `train_dataloader` /
  `val_dataloader` containers that combine the above with config
  validation and checkpoint resume.

See the [data guide](../guides/data.md) for end-to-end usage and the
[streaming guide](../guides/streaming.md) for detailed streaming
pipeline documentation.

## Streaming Data Modules

For large-scale pretraining, the streaming data module handles
memory-bounded data loading with checkpoint resume support:

::: llm.data.modules.streaming

## Streaming Dataset

::: llm.data.datasets.streaming

## Stream State

::: llm.data.stream_state

## Built-in Dataset Presets

The presets module ships well-known pretraining dataset
configurations so users don't have to hand-author the HF triples.

::: llm.data.presets

## Pluggable Text Sources

The `TextSource` abstraction + `SOURCE_REGISTRY` plugin entry
points. Most users won't need to read this — the built-in
`local` and `hf` sources cover the common cases — but custom
sources (S3, GCS, private archives) plug in here.

::: llm.data.sources

## DVC Integration

::: llm.data.dvc
