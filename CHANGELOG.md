# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Flash Attention 2 integration
- Paged Attention for improved memory efficiency
- RLHF/DPO alignment techniques
- Multi-modal extensions (vision-language models)
- Model quantization (INT8/INT4)

## [0.0.3] - 2025-12-23

### Added

- **Inference Serving**:
  - Production-ready REST API with FastAPI
  - Streaming support via Server-Sent Events (SSE)
  - Advanced sampling strategies (nucleus sampling/top-p, repetition penalty)
  - Prometheus metrics endpoint for monitoring
  - API key authentication (`X-API-Key` header)
  - Structured logging with `python-json-logger`
  - Real PyTorch model weights loading from checkpoint files
  - Pickled tokenizer object loading support

- **Component Registry**:
  - Automatic component registration system (`ComponentRegistry`)
  - Core components (MHA, MLP, MoE) auto-registered via side-effect imports
  - Prevents "component not found" errors in simplified scripts

- **Data Abstraction**:
  - Formalized `BaseTokenizer` protocol
  - `BaseDataModule` abstraction for flexible data handling
  - Environment variable configuration support (e.g., `LLM_TRAINING__EPOCHS`)

- **Testing & CLI**:
  - `--num-samples` flag in `train.py` for rapid regression testing
  - Scheduler edge case tests (`test_scheduler_edge_cases.py`)
  - Validation logging tests (`test_engine_logging.py`)
  - Component registry tests (`test_init.py`)
  - Model loading verification tests
  - Auto-device detection in training scripts (prioritizes CUDA)

- **Documentation**:
  - Comprehensive usage guide (`docs/usage.md`)
  - Architecture documentation (`docs/architecture.md`)
  - Engineering documentation (ADRs, PR templates, FAQ)
  - VS Code configuration and extensions

### Changed

- **Architecture Modernization**:
  - Migrated to Pydantic v2 (`BaseSettings`, `BaseModel`) for configuration
  - Fully typed and validated configuration system
  - CLI migration from `argparse` to `typer` for better UX

- **Naming Standardization**:
  - Unified `ffn_hidden_size` → `intermediate_size` across codebase
  - Standardized input parameter `x` → `hidden_states` in forward methods
  - Applied to `MLP`, `LayerNorm`, `RMSNorm`, `DecoderModel`, `TransformerBlock`
  - Updated all 309 tests to reflect API changes

- **Code Quality**:
  - Standardized punctuation in documentation (full-width → half-width)
  - Improved type hints and documentation comments
  - Refactored `TransformerBlock.forward` for clarity

### Fixed

- **Core Bugs**:
  - `CosineAnnealingLR` `T_max` calculation when `epochs == warmup_epochs` (ZeroDivisionError)
  - `TrainingEngine` validation logging crash when `gradient_norms` is empty (IndexError)
  - PAD token generation issue in inference (logits masking)
  - `SyntheticDataModule` `prefetch_factor` handling with `num_workers=0`
  - `TransformerBlock` shared norm instance bug (independent `norm1`/`norm2`)
  - Scheduler/optimizer step order warnings in tests
  - PositionalEncoding support for `start_pos` in incremental generation
  - MLP SwiGLU operation order for numerical consistency
  - Prompt truncation respecting `max_seq_len` with new tokens
  - Auto AMP dtype resolution for CPU-only environments

- **Registry & Imports**:
  - Package auto-registration via `import llm`
  - Component not found errors in simplified execution

## [0.0.2] - 2025-12-21

### Added

- Comprehensive ROADMAP.md with 15 development stages and clear priorities
- Grouped Query Attention (GQA) support in MultiHeadAttention for balanced performance and memory efficiency
- SwiGLU activation function in MLP layers for enhanced performance
- Unified QKV projection optimization for improved memory layout and computational throughput
- BPETokenizer for production-ready subword tokenization
- RMSNorm support as alternative normalization layer
- KV Cache support in inference for efficient autoregressive generation
- Top-k and Top-p sampling strategies in inference
- Greedy search decoding
- Detailed training framework documentation (8 comprehensive guides)
- 262 comprehensive test cases covering all core functionality
- CPU-friendly LLM tutorial and development guide

### Changed

- Updated project structure for better modularity and maintainability
- Enhanced TrainingEngine with improved callback system and performance monitoring
- Improved mixed precision support with automatic BF16/FP16 detection
- Optimized DecoderModel architecture with configurable components
- Enhanced documentation with better cross-references and examples

### Fixed

- All test regressions resolved (262/262 tests passing)
- Type checking issues across the codebase
- Memory management in distributed training
- Edge cases in attention masking and positional encoding

## [0.0.1] - 2024

### Added

- Initial project setup with modern Python tooling (uv, hatchling)
- Basic Decoder-only Transformer architecture
- Multi-Head Attention (MHA) implementation
- Standard MLP with GELU activation
- SimpleCharacterTokenizer for basic experimentation
- Positional encoding (sinusoidal and learned)
- TrainingEngine with Distributed Data Parallel (DDP) support
- Automatic Mixed Precision (AMP) training
- Basic Mixture of Experts (MoE) implementation
- Core data loading and processing infrastructure
- BaseDataModule abstraction for flexible data handling
- pytest-based testing infrastructure
- CI/CD pipeline with GitHub Actions
- Code quality tools (ruff for linting/formatting, mypy for type checking)
- Pre-commit hooks for code quality enforcement
- Docker support for containerized development
- Comprehensive README and contributing guidelines
