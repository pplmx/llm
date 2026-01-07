# Scripts

This directory contains development and utility scripts for the LLM framework.

## Usage

All scripts can be run with `uv run`:

```bash
uv run scripts/<script_name>.py [options]
```

## Available Scripts

| Script | Description |
| ------ | ----------- |
| `e2e_pipeline.py` | End-to-end smoke test (Train → Evaluate → Inference) |
| `train_simple_decoder.py` | Train a decoder model on text files |
| `benchmark_inference.py` | Benchmark inference performance |

## Script Details

### e2e_pipeline.py

Automated end-to-end pipeline for validating the framework:

```bash
# Quick test with defaults
uv run scripts/e2e_pipeline.py

# Custom configuration
uv run scripts/e2e_pipeline.py --epochs 5 --hidden-size 128 --num-layers 4
```

Options:

- `--hidden-size`: Model dimension (default: 64)
- `--num-layers`: Number of transformer layers (default: 2)
- `--epochs`: Training epochs (default: 3)
- `--num-samples`: Synthetic samples to generate (default: 200)
- `--device`: Device to use: 'cpu', 'cuda', or 'auto' (default: auto)

### train_simple_decoder.py

Train a decoder model on your own text data:

```bash
uv run scripts/train_simple_decoder.py \
  --file-path data/train.txt \
  --val-file-path data/val.txt \
  --epochs 10 \
  --batch-size 32
```

### benchmark_inference.py

Benchmark inference throughput and latency:

```bash
uv run scripts/benchmark_inference.py
```
