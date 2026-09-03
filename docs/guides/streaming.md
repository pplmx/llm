---
tags:
  - 指南
  - 数据
  - 训练
---

# Streaming Data Pipeline Guide

This guide covers streaming data ingestion for large-scale pretraining
using the `stream_lm` training task. It complements the
[Data Pipelines Guide](data.md) with workflow-specific details.

## When to Use Streaming

Streaming is the recommended approach when:

- Your dataset is too large to download/download into memory (>10GB)
- You want to start training immediately without a separate download step
- You need checkpoint resume with exact cursor tracking
- You're using HuggingFace datasets in streaming mode

For small datasets that fit in memory, use the standard `lm` task with
`data_source: local` and a map-style `TextDataset`.

## Streaming Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    StreamingTextDataModule                   │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │           StreamDataState (cursor tracker)            │  │
│  │  line_index: {rank_0: N, rank_1: M, ...}             │  │
│  │  token_buffer: [partial tokens at boundary]          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  TextSource ──→ StreamingTextDataset ──→ DataLoader         │
│     │                    │                                  │
│     │ HFStreamTextSource │                                  │
│     │ LocalLineTextSource│                                  │
│     │ DedupTextSource    │ (optional wrapper)              │
│     └────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Reference

### Data Configuration (`data:` section)

| Field               | Type | Default    | Description                                                  |
| ------------------- | ---- | ---------- | ------------------------------------------------------------ |
| `data_source`       | str  | `"local"`  | Source type: `local`, `hf`, `dedup_local`, `dedup_hf`        |
| `dataset_path`      | str  | `""`       | Local file path (required for `local`/`dedup_local`)         |
| `dataset_name`      | str  | `""`       | HF dataset identifier (required for `hf`/`dedup_hf`)         |
| `dataset_config`    | str  | `None`     | HF dataset config/subset                                     |
| `dataset_split`     | str  | `"train"`  | HF dataset split                                             |
| `text_column`       | str  | `"text"`   | Text field name in dataset rows                              |
| `max_seq_len`       | int  | 512        | Maximum sequence length                                      |
| `steps_per_epoch`   | int  | —          | Number of optimizer steps per epoch (required for streaming) |
| `seen_hashes_path`  | str  | `None`     | Path for dedup hash persistence                              |
| `write_seen_hashes` | bool | `false`    | Append new hashes to `seen_hashes_path`                      |
| `hash_algo`         | str  | `"sha256"` | Hash algorithm for deduplication                             |

### Tokenizer Configuration

```yaml
data:
  data_source: local
  tokenizer_type: simple  # or "hf"
  tokenizer_path: null    # For HF: "gpt2", "EleutherAI/gpt-neox-20b"
  dataset_path: data/demo.txt
  max_seq_len: 512
  steps_per_epoch: 10
```

## Built-in Dataset Presets

The framework ships with presets for common pretraining datasets. These
are defined in `src/llm/data/presets.py` and can be applied programmatically:

```python
from llm.data.presets import C4_PRESET, THEPILE_PRESET, REDPAJAMA_PRESETS, apply_to_config
from llm.training.core.config import DataConfig

# Apply C4 preset
cfg = DataConfig(data_source="hf", max_seq_len=512)
apply_to_config(cfg, C4_PRESET)
# cfg.dataset_name == "allenai/c4"
# cfg.dataset_config == "en"

# Apply RedPajama subset
cfg = DataConfig(data_source="hf", max_seq_len=512)
apply_to_config(cfg, REDPAJAMA_PRESETS["redpajama/arxiv"])
# cfg.dataset_name == "togethercomputer/RedPajama-Data-1T"
# cfg.dataset_config == "arxiv"
```

### Available Presets

| Preset Name               | Dataset                              | Config          | Text Column |
| ------------------------- | ------------------------------------ | --------------- | ----------- |
| `C4_PRESET`               | `allenai/c4`                         | `en`            | `text`      |
| `THEPILE_PRESET`          | `monology/pile-uncopyrighted`        | _(none)_        | `text`      |
| `redpajama/arxiv`         | `togethercomputer/RedPajama-Data-1T` | `arxiv`         | `text`      |
| `redpajama/book`          | `togethercomputer/RedPajama-Data-1T` | `book`          | `text`      |
| `redpajama/common_crawl`  | `togethercomputer/RedPajama-Data-1T` | `common_crawl`  | `text`      |
| `redpajama/c4`            | `togethercomputer/RedPajama-Data-1T` | `c4`            | `text`      |
| `redpajama/github`        | `togethercomputer/RedPajama-Data-1T` | `github`        | `text`      |
| `redpajama/stackexchange` | `togethercomputer/RedPajama-Data-1T` | `stackexchange` | `text`      |
| `redpajama/wikipedia`     | `togethercomputer/RedPajama-Data-1T` | `wikipedia`     | `text`      |

## Deduplication

The `DedupTextSource` wraps any text source to drop duplicate records
by content hash. This is essential for pretraining on web-crawled data
which contains substantial exact duplicates.

### How It Works

1. **Hashing**: Text is normalized (whitespace collapse + strip) and
   hashed using the specified algorithm (default: SHA-256).
2. **In-memory tracking**: Seen hashes are stored in a `set` during
   the run to deduplicate within a single streaming pass.
3. **Cross-run persistence**: Optionally load and append to a file
   so dedup state survives across runs.

### Configuration

```yaml
data:
  data_source: dedup_hf    # or "dedup_local"
  dataset_name: allenai/c4
  dataset_config: en
  seen_hashes_path: checkpoints/c4_seen_hashes.txt
  write_seen_hashes: true
  hash_algo: sha256        # sha256, sha1, md5, etc.
```

### Important: Resume Behavior

When using `DedupTextSource` in streaming mode:

- **In-memory dedup**: During a single run, duplicate records within
  the same data stream are dropped based on an in-memory hash set.
- **Cross-run persistence**: If `write_seen_hashes=True` with a
  `seen_hashes_path`, the hash set is persisted to disk and reloaded
  on restart, preventing re-processing of records from previous runs.
- **Resume cursor**: The `StreamDataState` tracks the line index cursor
  independently from dedup state. On resume, the cursor skips to the
  correct position, and dedup state (if persisted) is restored.

**Warning**: If `write_seen_hashes=false` (default), dedup state is
in-memory only. On checkpoint resume, the `DedupTextSource` is
recreated fresh and previously-dseen records will be re-processed.
This is by design — the streaming framework prioritizes correct cursor
resume over dedup consistency to avoid blocking training on slow
disk I/O for hash files.

## Checkpoint Resume

Streaming checkpoints store extra state beyond model weights:

```python
{
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "stream_data": {
        "line_index": {
            "0": 15420,      # rank 0 processed 15420 lines
            "1": 15418,      # rank 1 (slightly different due to padding)
        },
        "token_buffer": [...]  # partial tokens at boundary
    },
    "stream_source": {
        "type": "dedup",
        "inner": {
            "type": "hf",
            "dataset_name": "allenai/c4",
            "dataset_config": "en",
            ...
        },
        "hash_algo": "sha256",
    }
}
```

On resume:

1. **Source fingerprint validation**: The framework checks that the
   current `DataConfig` produces the same source fingerprint as the
   checkpoint. Changing datasets mid-run raises a clear error.

2. **Cursor restoration**: The `StreamDataState` is restored, and
   `iter_texts(skip=line_index)` resumes from the exact position.

3. **Deterministic tokenization**: The same tokenizer state ensures
   identical tokenization of resumed data.

## Performance Considerations

### Multiprocessing Limitation

The streaming pipeline intentionally uses `num_workers=0` for the
DataLoader. This is because:

1. **Cursor persistence**: The streaming cursor lives in the main
   process. DataLoader workers fork and lose state at checkpoint time.
2. **Resume correctness**: Checkpointing the cursor requires it to
   be in the same process as the checkpoint logic.
3. **Simplicity**: For most datasets, tokenization is fast enough
   to not be a bottleneck.

If your tokenizer is very slow, consider:

- Pre-tokenizing the dataset (store as token IDs)
- Using a faster tokenizer (e.g., HF tokenizers in Rust)
- Implementing a custom `StreamingTextDataset` with internal threading

### Memory Budget

Streaming keeps only `max_seq_len * batch_size` tokens in memory per
worker. The rest of the dataset is never fully loaded — records are
yielded one at a time from the source iterator.

## Custom Sources

You can register custom text sources via the `SOURCE_REGISTRY`:

```python
from llm.data.sources import TextSource, SOURCE_REGISTRY


class S3TextSource(TextSource):
    """Stream text records from an S3 bucket."""

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key

    def iter_texts(self, skip: int = 0):
        import boto3

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=self.bucket, Key=self.key)
        lines = obj["Body"].iter_lines()
        for i, line in enumerate(lines):
            if i < skip:
                continue
            text = line.decode("utf-8").strip()
            if text:
                yield text

    def source_fingerprint(self):
        return {"type": "s3", "bucket": self.bucket, "key": self.key}


# Register it
SOURCE_REGISTRY.register("s3", lambda cfg: S3TextSource(cfg.dataset_bucket, cfg.dataset_key))
```

Then use it in YAML:

```yaml
data:
  data_source: s3
  dataset_bucket: my-pretraining-bucket
  dataset_key: c4-en-2024-01.txt
```

## E2E Example: Full Streaming Workflow

```bash
# 1. Create a small demo corpus
mkdir -p data
python -c "print('\n'.join(f'sample line {i}' for i in range(500)))" > data/demo.txt

# 2. Train for 1 epoch (smoke test)
uv run llm-train --task stream_lm \
  --config-path configs/streaming_local_demo.yaml \
  --epochs 1 --steps-per-epoch 5

# 3. Resume training (simulates interruption + recovery)
#    checkpoint_dir / resume 走 YAML（无 CLI 参数）。编辑
#    configs/streaming_local_demo.yaml：
#      checkpoint:
#        checkpoint_dir: checkpoints/demo
#        resume_from_checkpoint: checkpoints/demo/latest
uv run llm-train --task stream_lm \
  --config-path configs/streaming_local_demo.yaml \
  --epochs 1 --steps-per-epoch 5

# 4. Scale to production with C4
#    max_steps 也是 YAML（training.max_steps），无 CLI 参数。
uv run llm-train --task stream_lm \
  --config-path configs/streaming_c4.yaml \
  --epochs 1
```

## Troubleshooting

### "stream source fingerprint mismatch"

The checkpoint's source fingerprint doesn't match the current config.
This is a **loud failure** by design — it prevents silently training
on different data after a resume.

**Fix**: Ensure your config matches the checkpoint's source, or start
fresh training.

### "StreamingTextDataset keeps its resume cursor in the main process"

This warning appears when `num_workers > 0`. It's informational — the
framework forces `num_workers=0` to maintain checkpoint correctness.

### "DedupTextSource is running with in-memory only state"

This warning appears when using `DedupTextSource` without
`write_seen_hashes=True`. Cross-run dedup state won't persist.

**Fix**: Set `seen_hashes_path` and `write_seen_hashes: true` in your
config if cross-run dedup is important.

### "dataset_path is required when data_source='local'"

**Fix**: Provide a `dataset_path` pointing to a text file with one
record per line.

### "dataset_name is required when data_source='hf'"

**Fix**: Provide a HuggingFace dataset identifier like
`allenai/c4` or use a preset:

```python
from llm.data.presets import C4_PRESET, apply_to_config

apply_to_config(cfg.data, C4_PRESET)
```
