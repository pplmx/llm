# llm

[![CI](https://github.com/pplmx/llm/workflows/CI/badge.svg)](https://github.com/pplmx/llm/actions)
[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Roadmap](#roadmap)
- [Changelog](#changelog)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview

`llm` is a modular and extensible PyTorch framework for training and experimenting with Large Language Models (LLMs). It provides a robust infrastructure for building, training, and evaluating LLMs, focusing on modularity, scalability, and ease of experimentation.

## Features

- **Flexible & Pluggable Architecture**:
    - **Component Registry**: Switch between MHA/MoE or future implementations (FlashAttn) via config.
    - Grouped Query Attention (GQA) & SwiGLU support.
    - Unified QKV projection & Flexible Norms (RMSNorm, LayerNorm).
- **Robust Training Framework**:
    - Distributed Data Parallel (DDP) & Automatic Mixed Precision (AMP).
    - Type-safe Configuration via Pydantic & Typer CLI.
    - `torch.compile` optimization integration.
- **Parameter-Efficient Fine-Tuning**:
    - **LoRA**: Low-rank adaptation with merge/unmerge support.
    - **QLoRA**: 4-bit NF4 quantized base + fp16 LoRA adapters (~4x memory reduction).
- **Inference & Serving**:
    - **OpenAI-Compatible API**: `/v1/chat/completions` endpoint works with OpenAI SDK.
    - **KVCache**: Pre-allocated cache for efficient autoregressive generation.
    - Streaming support (Server-Sent Events).
    - Prometheus metrics & API key authentication.
- **Data & Tokenization Abstraction**:
    - **HuggingFace Integration**: Direct support for HF Tokenizers (GPT-2, Llama, etc.).
    - Modular `DataModule` design for text datasets.
    - Legacy Character-level tokenizer for simple experiments.
- **CLI Tools**: `llm train`, `llm serve`, `llm quantize`, `llm prune`, `llm decompose`, `llm migrate-ckpt` commands for easy usage.

## Quick Start

To get a model training quickly, install the framework and run the CLI:

1. **Install the CLI tools (once)** — from the Git source (not yet on PyPI):

    ```bash
    uv tool install git+https://github.com/pplmx/llm.git
    ```

   This provides the `llm train`, `llm serve`, `llm quantize`, `llm prune`,
   `llm decompose`, and `llm migrate-ckpt` commands. (See
   [Installation](#installation) for the library form and the developer
   workflow.)

2. **Run a Training Example**: Execute the main training script with a language modeling task.

    ```bash
    llm train --task lm --epochs 1 --batch-size 32
    ```

   This command will train a language model for 1 epoch with a batch size of 32.

3. **Start Inference Server** (optional):

    ```bash
    llm serve
    ```

   The server will be available at `http://localhost:8000` with OpenAI-compatible API.

4. **Quantize a Model with GPTQ** (optional):

    ```bash
    llm quantize gptq \
        --model ckpt.pt \
        --output ckpt-int4.pt \
        --calib-data texts.txt \
        --tokenizer gpt2 \
        --bits 4
    ```

   Runs Hessian-aware 4-bit / 8-bit PTQ (Frantar 2022) with the same
   validation rules as the Python API. Mixed-precision per-layer
   dispatch via `LayerQuantPolicy` is supported through the Python
   API; the CLI exposes `--target-modules` for a coarse filter.

## Installation

### Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv): A fast Python package installer and resolver, written in Rust.
- [just](https://github.com/casey/just): A command runner (the `Makefile`'s successor here, http://just.systems). **Developer-only** — needed for the `just init` / `just sync` / `just check` developer workflow below; users installing via `uv add` / `uv tool install` do not need it. Install via the official script
  (`curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sh -s -- --to <dir>`), `cargo install
  just`, or a package manager (`brew install just` / `choco install just` / `apt install just`).

### For users: install and use `llm`

`llm` is not yet published to PyPI, so install it straight from the Git source.

Use `llm` as a **library** dependency of your own project:

```bash
uv add git+https://github.com/pplmx/llm.git
```

Or install it as a **command-line tool** — this provides the `llm train`,
`llm serve`, `llm quantize`, `llm prune`, `llm decompose`, and
`llm migrate-ckpt` commands:

```bash
uv tool install git+https://github.com/pplmx/llm.git
```

### For developers: work on the source in this repo

This project uses `uv` for dependency management and `Justfile` for common
development tasks. The steps below are for **editing the code in this
repository**, not for installing the package (see the user install above).

1. **Install `uv`**: If you don't have `uv` installed, follow the official
   instructions [here](https://github.com/astral-sh/uv#installation). `just` is
   also required (see Requirements above).
2. **Initialize the project**: From the repo root, run `just init` to create
   the virtual environment, install all dependencies, and install pre-commit
   hooks.

    ```bash
    just init
    ```

3. **Re-sync dependencies (if needed)**: Whenever `pyproject.toml` or
   `uv.lock` changes, re-synchronize:

    ```bash
    just sync
    ```

## Usage

For comprehensive documentation, including detailed usage examples, development guides, and troubleshooting, please refer to our dedicated documentation section:

- [Architecture Guide](docs/reference/architecture.md)
- [Fine-Tuning Guide (LoRA/QLoRA)](docs/guides/finetuning.md)
- [Inference Optimization Guide](docs/guides/inference.md)
- [Development Guide](docs/development/README.md)
- [Quick Start](docs/getting-started.md)
- [Project Troubleshooting](docs/troubleshooting.md)
- [Tutorials](docs/tutorials/01-pretraining.md)
- [Examples](examples/README.md)
- [Scripts](scripts/README.md)
- [Notebooks](notebooks/README.md)

## Development

For detailed information on setting up the development environment, running tests, maintaining code quality, and other development workflows, please refer to our comprehensive [Development Guide](docs/development/README.md).

## Troubleshooting

If you encounter any issues while using `llm`, please check our [Troubleshooting Guide](docs/troubleshooting.md) for common problems and their solutions. If you can't find a solution to your problem, please [open an issue](https://github.com/pplmx/llm/issues) on our GitHub repository.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/pplmx/llm/blob/main/CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Roadmap

For our detailed development roadmap and future plans, please see [ROADMAP.md](ROADMAP.md).

## Changelog

For a detailed history of changes to this project, please refer to our [CHANGELOG.md](CHANGELOG.md).

## Contact

For questions, suggestions, or support, please open an issue on our GitHub repository.

## Acknowledgements

We acknowledge all contributors, open-source projects, and resources that have inspired and supported the development of this project.
