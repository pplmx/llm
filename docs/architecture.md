# Architecture Documentation

This document provides a deep dive into the architecture of the `llm` project, explaining its core design principles, component structure, and key abstractions.

## Design Philosophy

The project follows a **Modular & Composable** design philosophy:

* **src-layout**: Code is isolated in `src/` to prevent import layout confusion.
* **Separation of Concerns**:
    * `core`: Pure, reusable PyTorch modules (Attention, MLP, Norm).
    * `models`: Logic to assemble core components into full architectures (Decoder).
    * `training`: Orchestration of training loops, DDP, logging.
    * `serving`: High-performance inference API (FastAPI).
* **Registry Pattern**: Core components are decoupled and selectable via configuration.
* **Configuration as Code**: Pydantic models define type-safe, validating configurations.

## System Overview

```mermaid
graph TD
    Config[Configuration (Pydantic)] --> Training[Training Engine]
    Config --> Serving[Serving Engine]

    subgraph "Core Layers (src/llm/core)"
        Reg[Registry]
        MHA[MultiHeadAttention]
        MLP[MLP / MoE]
        Norm[RMSNorm / LayerNorm]
    end

    subgraph "Data Abstraction (src/llm/data)"
        Tokenizer[BaseTokenizer / HFTokenizer]
        Dataset[TextDataset]
        DataModule[TextDataModule]
    end

    subgraph "Models (src/llm/models)"
        Decoder[DecoderModel]
    end

    Reg --> MHA
    Reg --> MLP

    Training --> DataModule
    Training --> Decoder

    DataModule --> Tokenizer
    Dataset --> Tokenizer

    Decoder --> MHA
    Decoder --> MLP
    Decoder --> Norm
```

## Core Components & Registry

To support rapid experimentation with different architectural variants (e.g., Flash Attention, SwiGLU, MoE), we employ a **Registry Pattern**.

### Component Registry

Located in `src/llm/core/registry.py`, registries allow dynamic selection of implementations:

* **`ATTENTION_REGISTRY`**: `mha` (Standard), `flash` (Planned)
* **`MLP_REGISTRY`**: `mlp` (Standard), `moe` (Mixture of Experts)
* **`NORM_REGISTRY`**: `layer_norm`, `rms_norm`

Components register themselves via decorators:

```python
@ATTENTION_REGISTRY.register("mha")
class MultiHeadAttention(nn.Module): ...
```

Configuration controls which implementation is used:

```yaml
model:
  attn_impl: "mha"
  mlp_impl: "moe"
```

## Data Abstraction

The project decouples data loading from tokenization logic to support both simple character-level experiments and production-grade HuggingFace tokenizers.

### Tokenizer Hierarchy

* **`BaseTokenizer` (Protocol)**: Defines the interface (`encode`, `decode`, `vocab_size`).
* **`SimpleCharacterTokenizer`**: A lightweight, dependency-free tokenizer for basic testing.
* **`HFTokenizer`**: A wrapper around `transformers.AutoTokenizer`, enabling access to the entire HuggingFace ecosystem.

### Data Module

`TextDataModule` uses `DataConfig` to determine which tokenizer to load and how to process the dataset.

## Configuration System

All configuration is managed via Pydantic models in `src/llm/training/core/config.py`, offering:

* **Type Safety**: Automatic type validation.
* **Environment Variables**: Override via `LLM_MODEL__HIDDEN_SIZE=1024`.
* **CLI Integration**: `Typer` automatically exposes these configs as command-line arguments.

### Config Structure

* **`ModelConfig`**: Architecture params (`hidden_size`, `num_layers`, `attn_impl`).
* **`DataConfig`**: Data params (`tokenizer_type`, `dataset_path`).
* **`TrainingConfig`**: loop params (`epochs`, `lr`).
* **`DistributedConfig`**: DDP params (`master_addr`, `world_size`).
* **`OptimizationConfig`**: performance (`use_compile`, `use_amp`).

## Attention Mechanism

The project supports multiple attention variants through the registry pattern:

```mermaid
graph LR
    subgraph "Attention Flow"
        Input[Hidden States] --> QKV[QKV Projection]
        QKV --> Q[Query]
        QKV --> K[Key]
        QKV --> V[Value]

        Q --> SDPA[Scaled Dot-Product Attention]
        K --> SDPA
        V --> SDPA

        SDPA --> Out[Output Projection]
    end

    subgraph "Variants"
        MHA[MHA: All heads independent]
        GQA[GQA: Grouped KV heads]
        MLA[MLA: Latent attention]
    end
```

### Supported Features

| Feature | Description |
| ------- | ----------- |
| **GQA** | Multiple query heads share KV heads (memory efficient) |
| **Sliding Window** | Limits attention scope for long sequences |
| **KV Cache** | Caches key/value for autoregressive generation |
| **RoPE** | Rotary position embeddings with scaling |
| **ALiBi** | Attention with linear biases |

## Training Pipeline

```mermaid
sequenceDiagram
    participant User
    participant CLI as llm-train
    participant Engine as TrainingEngine
    participant DDP as DistributedDataParallel
    participant Model as DecoderModel

    User->>CLI: llm-train --task lm
    CLI->>Engine: Initialize with Config
    Engine->>DDP: Wrap Model

    loop Each Epoch
        Engine->>Engine: Set epoch for sampler
        loop Each Batch
            Engine->>Model: Forward pass
            Model-->>Engine: Loss
            Engine->>Engine: Backward + Optimize
        end
        Engine->>Engine: Validation
        Engine->>Engine: Callbacks (logging, checkpointing)
    end

    Engine-->>User: Training complete
```
