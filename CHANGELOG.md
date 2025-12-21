# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Inference API with FastAPI and streaming support
- Flash Attention 2 integration
- Paged Attention for improved memory efficiency
- RLHF/DPO alignment techniques
- Multi-modal extensions (vision-language models)
- Model quantization (INT8/INT4)

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
