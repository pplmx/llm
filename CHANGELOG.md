# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Flash Attention 2 integration
- Paged Attention full model forward path (sidecar exists; see ADR-004)

### Added

- **Export registry parity** (Tier 3 #32): `EXPORT_REGISTRY` mirrors `BACKEND_REGISTRY`. Built-in `onnx` target plus the `llm.export_backends` setuptools entry-point group for third-party targets (`torch.compile`, `vLLM`, `TensorRT-LLM`, `torch.export`, `OpenVINO`, ...). `export_model("onnx", model, output_path, **kwargs)` is a drop-in for `export_to_onnx(model, output_path, **kwargs)`; the legacy ONNX API is preserved for backward compatibility.
- **TorchScript export target** (Tier 3 #33): first non-built-in backend to exercise the `EXPORT_REGISTRY` plug-in path. Registered through `pyproject.toml`'s `llm.export_backends` entry point as `torchscript = "llm.export.torchscript:build_torchscript_exporter"`. Trace method (`torch.jit.trace`) supported; script method (`torch.jit.script`) wired but `xfail`-tracked for `DecoderModel` (the model's `PositionalEncoding` uses dynamic attribute access that the TorchScript compiler rejects).
- **Docs sync for export registry** (Tier 3 #34): `docs/reference/architecture.md` now lists `export/`'s full layout (`registry.py`, `onnx.py`, `torchscript.py`, `_wrapper.py`) and adds the `llm.export_backends` / `EXPORT_REGISTRY` row to the plugin-kernel table. New `docs/adr/005-export-registry.md` records the architectural decision and cross-references the audit Finding BH plus Tier 3 #32/#33. ADR README now also lists `004-paged-attention-serving.md` (previously missing from the index). `ROADMAP.md` §阶段十四 checkboxes synced to current state — `safetensors 保存`, `TorchScript 导出`, `Export registry`, `HuggingFace Hub publish` all marked done; `checkpoint 格式统一` left unchecked with a one-line note explaining the gap.
- **OpenAI-compat `frequency_penalty`** (Tier 3 #35): the OpenAI-compatible chat / generate endpoints now actually apply the `frequency_penalty` parameter instead of silently dropping it. New `apply_frequency_penalty` sampling helper subtracts `frequency_penalty * count(token)` from each seen token's logit (matches OpenAI's documented semantics). `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat + generate routers plumb `request.frequency_penalty` to the service; `ChatCompletionRequest`, `GenerationRequest`, and `BatchGenerationRequest` schemas all accept it (range `[-2.0, 2.0]` per OpenAI). The `(not implemented)` hint in the schema drops.
- **OpenAI-compat `presence_penalty` with flat-per-token semantics** (Tier 3 #37): the OpenAI-compatible chat endpoint now applies `presence_penalty` correctly — penalising a token **once** if it has appeared at all in the generated text, regardless of how many times — instead of the legacy `repetition_penalty = 1.0 + presence_penalty` alias (which produced magnitude-scaled, count-independent behaviour opposite to OpenAI's spec). New `apply_presence_penalty` sampling helper subtracts a flat `presence_penalty` from every seen token's logit (count-independent — that's the key distinction from `frequency_penalty`, which scales by count). `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat router plumbs `request.presence_penalty` to the service as its own kwarg; `ChatCompletionRequest.presence_penalty` schema widens to OpenAI's `[-2.0, 2.0]` range and drops the `(mapped to repetition_penalty)` hint. The `repetition_penalty` alias hack at `routers/chat.py:81` is gone.
- **OpenAI-compat `logit_bias`** (Tier 3 #38): the OpenAI-compatible chat / generate endpoints now accept the `logit_bias` parameter (a JSON object mapping token ids to additive logit biases in `[-100, 100]`). New `apply_logit_bias` sampling helper adds the bias to each affected logit via `index_add_` — applied **after** the penalty helpers (repetition / frequency / presence) so the bias dominates any natural penalty, matching OpenAI's reference ordering. String keys (JSON's natural type for object keys) are coerced to `int` at the helper boundary; invalid keys (non-numeric or out-of-vocab) are silently dropped. `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat + generate + batch_generate routers plumb `request.logit_bias` to the service; `ChatCompletionRequest`, `GenerationRequest`, and `BatchGenerationRequest` schemas all accept it.
- **Data dedup `TextSource` wrapper** (Tier 3 #39, P0 pretraining productization): new `DedupTextSource` class in `src/llm/data/sources.py` wraps any inner `TextSource` and drops duplicate records by content hash. Default normalization is **case-sensitive** strip + collapse-internal-whitespace (conflating "Apple" vs "apple" would silently drop legitimate records, so case is preserved). `iter_texts(skip=N)` delegates `skip` to the inner source so the `line_index` resume semantics used by `StreamingTextDataset` stay consistent with non-dedup sources. `source_fingerprint()` exposes the inner source's fingerprint plus the dedup strategy so `validate_source_fingerprint` catches config drift on checkpoint resume. Two new `SOURCE_REGISTRY` entries — `dedup_local` and `dedup_hf` — compose the wrapper with the existing `LocalLineTextSource` / `HFStreamTextSource` builders, so `data_source="dedup_local"` / `data_source="dedup_hf"` work end-to-end with the streaming DataModule. `DataConfig` gains three optional fields (`seen_hashes_path`, `write_seen_hashes`, `hash_algo`) and its `data_source` regex widens to `^(local|hf|dedup_local|dedup_hf)$`. The `seen_hashes_path` file is loaded on construction (so dedup state survives across runs) and appended to in append-only mode when `write_seen_hashes=True` (so the seen-set grows monotonically with minimal I/O). Fuzzy / MinHash dedup is a deliberate follow-up — the interface intentionally allows `normalize` and `hash_algo` to be swapped so that can layer on top of this foundation.

### Changed

- **2026 Q2 architecture convergence** (Phases 1–4, Waves 1–3, P2 cleanup):
    - Unified `runtime.Registry` for all component registries; removed legacy `ComponentRegistry`
    - KV cache: single `KVCache` / `kv_caches` API; removed `past_key_value` tuple path
    - Norm: `norm_impl` config + `NORM_REGISTRY` wiring (`layer_norm`, `rms_norm`)
    - Eval: removed `evaluator.py` / `infer_task.py`; unified `EvaluationRunner`
    - Serving: removed `priority_scheduler.py`, `serving/prefix_cache.py`; `ContinuousBatchingEngine` gains `SlotPrefixCache`, `from_serving_config()`
    - Data: `TokenizedMapDataModule.setup_tokenized_file_dataset()`, `SamplerMapDataModule`, `TokenizerFactory` helpers
    - Tasks: `regression_mlp` via `llm.models` entry point; `RegressionTask` uses `ModelFactory`
    - Bootstrap: model registration via setuptools entry points only (`decoder`, `regression_mlp`)
    - MLA: registered as `@register_attention("mla")`; supports linear `KVCache` and paged `PagedKVCache` (Tier 3 #31). The current implementation is the placeholder variant (learnable latent queries + uniform-mean output); DeepSeek-V2-style latent-compressed K, V is a separate follow-up.

### Refactored

- **Code Organization**:
    - Extracted `make_factory_kwargs()` and `init_lora_weights()` utilities
    - Migrated all `__main__` demo code to test files
    - Added custom exception module with hierarchical exception types

- **Error Handling**:
    - Replaced broad `except Exception` with specific exception types
    - Improved API error logging and message handling

- **Type Annotations**:
    - Fixed `pad_token_id` duplicate definition
    - Fixed `normalized_shape` tuple type mismatches
    - Added missing type annotations in MoE module
    - Fixed None handling in config utilities

- **Code Quality**:
    - Removed ~600 lines of demo code from source modules
    - Preserved educational NumPy implementations for learning
    - Added comprehensive test coverage for demo functionality

## [0.0.5] - 2026-01-08

### Added

- **SFT (Supervised Fine-tuning)**:
    - `SFTDataset` for instruction tuning with input masking
    - `SFTDataModule` for data loading
    - `SFTTask` registered as `--task sft` in CLI
    - Tests for all SFT components

- **DPO (Direct Preference Optimization)**:
    - `DPODataset` handling chosen/rejected pairs
    - `DPODataModule` for preference data loading
    - `DPOTask` with reference model management and DPO loss
    - Registered as `--task dpo` in CLI
    - Tests for all DPO components

- **Continuous Batching Engine** (Serving):
    - `src/llm/serving/engine.py` with `ContinuousBatchingEngine` class
    - Iteration-level scheduling via `Scheduler` and `SlotAllocator`
    - Pre-allocated KV cache pool for efficient memory management
    - Supports mixed prefill/decode batching with automatic padding
    - Clean API: requires `model` and `tokenizer` instances upfront
    - `src/llm/serving/scheduler.py` with FCFS scheduling logic

- **LoRA (Low-Rank Adaptation)**:
    - `src/llm/core/lora.py` with `LoRALinear` class for parameter-efficient fine-tuning
    - `apply_lora()`, `merge_lora()`, `get_lora_parameters()` helper functions
    - Device/dtype handling for CUDA compatibility
    - 17 tests covering training and weight merging

- **QLoRA (Quantized LoRA)**:
    - `src/llm/core/qlora.py` with `QLoRALinear` class
    - NF4 4-bit quantization for base weights (~4x memory reduction)
    - LoRA adapters remain in fp16/bf16 for training stability
    - `apply_qlora()` and `get_qlora_parameters()` helpers

- **RoPE (Rotary Position Embedding)**:
    - `src/llm/core/rope.py` with `RotaryPositionEmbedding` class
    - Linear, dynamic, and NTK-aware scaling methods for extended context
    - `apply_rotary_pos_emb()`, `get_rope_scaling_factor()` utilities
    - 15 tests

- **ALiBi (Attention with Linear Biases)**:
    - `src/llm/core/alibi.py` with `ALiBiPositionBias` class
    - `get_alibi_slopes()`, `build_alibi_bias()` functions
    - Cached bias computation for efficiency
    - 13 tests

- **Sliding Window Attention**:
    - `window_size` parameter in `scaled_dot_product_attention`
    - Propagated through `MultiHeadAttention`, `TransformerBlock`, `DecoderModel`
    - Reduces memory for long sequences by limiting attention scope
    - 10 tests

- **KV Cache Optimization**:
    - `src/llm/core/kv_cache.py` with `KVCache` class for pre-allocated cache buffers
    - In-place updates during autoregressive generation (avoids O(n²) memory operations)
    - Integrated into `MHA`, `TransformerBlock`, `DecoderModel`
    - Factory method `KVCache.from_model_config()` for easy instantiation
    - Unified `kv_caches` API; legacy tuple format removed in Wave 3

- **E2E Testing Infrastructure**:
    - `tests/e2e/` directory with comprehensive pipeline tests
    - `test_training.py`, `test_sft.py`, `test_dpo.py`
    - `test_gradient_accumulation.py`, `test_resume_training.py`
    - Advanced inference and callback tests

- **Documentation**:
    - `notebooks/quick_start.ipynb` interactive tutorial
    - Covers model building, training, inference, and advanced features

### Changed

- **SDPA Refactoring**:
    - Consolidated `scaled_dot_product_attention` wrapper into `src/llm/core/attn/sdpa.py`
    - Refactored `MultiHeadAttention` and `MultiLatentAttention` to use common `sdpa` wrapper
    - Archived custom implementation to `_learning/03_lab/experiments/custom_sdpa.py`

- **Test Suite Refactoring**:
    - Organized test files into subdirectories (`tests/training/`, `tests/inference/`, etc.)
    - Converted to functional testing style (real components over mocks)
    - Added shared fixtures in `tests/conftest.py`
    - Test count: 385 → 432

- **TrainingEngine**:
    - Support for dictionary batches in training/validation loops
    - Gradient accumulation implementation

- **DPO Reference Model**:
    - Use model reconstruction instead of `deepcopy` for ref_model creation

- **Documentation**:
    - Added `docs/README.md` as documentation entry point
    - Added MkDocs Material configuration (`mkdocs.yml`) for documentation site
    - Added GitHub Actions workflow for automatic GitHub Pages deployment
    - Added `guide-finetuning.md` (LoRA/QLoRA) and `guide-inference.md` (KVCache/GQA/Continuous Batching)
    - Enhanced `architecture.md` with detailed component diagrams and data flow analysis
    - Updated ROADMAP Phase 10.2 (Continuous Batching complete)

## [0.0.4] - 2026-01-07

### Added

- **Gradient Checkpointing**:
    - Memory-efficient training via `gradient_checkpointing` parameter in `DecoderModel`
    - `enable_gradient_checkpointing()` / `disable_gradient_checkpointing()` methods
    - Automatic incompatibility check with `use_cache=True`

- **E2E Pipeline Automation**:
    - `scripts/e2e_pipeline.py` for automated Train → Evaluate → Inference workflow
    - `src/llm/utils/e2e.py` with reusable E2E core functions (`E2EConfig`, `E2EResult`, `run_e2e_pipeline`)
    - Rich progress UI and configurable CLI options

- **OpenAI-Compatible Chat API** (`/v1/chat/completions`):
    - Compatible with official OpenAI Python SDK
    - Streaming and non-streaming chat completions
    - Bearer token authentication support
    - Multi-turn conversation handling
    - 8 new test cases for compatibility layer

- **Batch Inference**:
    - `batch_generate` function in `inference.py` with left-padding and batched forward pass
    - `BatchGenerationRequest` / `BatchGenerationResponse` schemas
    - `/batch_generate` API endpoint
    - 3 tests for batch inference (basic, single, empty)

- **Request Queue and Concurrency Control**:
    - `max_concurrent_requests` and `request_timeout` in `ServingConfig`
    - `asyncio.Semaphore` for concurrency limiting
    - `asyncio.timeout` for request timeout handling (504 response)

- **CLI Entry Points**:
    - `llm-train` command for training models
    - `llm-serve` command for starting inference server

- **Testing Infrastructure**:
    - Pytest markers using decorators: `quick`, `slow`, `heavy`, `e2e`
    - MoE integration tests (6 tests for expert routing, gradient flow)
    - E2E pipeline tests (full workflow, streaming consistency)
    - Gradient checkpointing tests (8 tests)
    - Total test count: 296 → 337

- **Examples Directory**:
    - `inference_demo.py` for basic text generation
    - `openai_client_demo.py` for OpenAI SDK usage

- **Documentation**:
    - `scripts/README.md` documenting all available scripts
    - HFTokenizer example in `usage.md`
    - Updated root `README.md` with links to Examples and Scripts

### Changed

- **Makefile Reorganization**:
    - `make test` now runs all tests by default
    - `make test-fast` for daily development (excludes heavy/e2e)
    - `make test-quick` for rapid iteration (~6s)
    - `make test-cov` for CI with coverage and allure reports
    - Removed redundant `test-all` and `test-integration`

- **CLI Standardization**:
    - CLI parameters changed from snake_case to kebab-case (`--file-path`, `--batch-size`)
    - Replace `typer` with `typer-slim[standard]` for reduced dependencies

- **Code Quality Improvements**:
    - Translate Chinese docstrings to English in serving module
    - Remove ~75 lines of redundant comments
    - Simplify section comments while preserving algorithm clarity

- **Documentation Refactoring**:
    - Eliminated redundancy between README, usage.md, and development.md
    - Clear document responsibility separation
    - Updated all docs to use new CLI commands
    - Enhanced package metadata (keywords, classifiers)

- **Module Exports**:
    - Enhanced `llm/__init__.py` with public API exports (`DecoderModel`, `generate`, etc.)
    - Enhanced `llm.serving` module exports (`LLMEngine`, `ServingConfig`, OpenAI schemas)

### Fixed

- Removed obsolete TODO comment in `engine.py`
- Removed duplicate `num_kv_heads` field in `ModelConfig`
- Fixed MD051/link-fragments in `tutorial-cpu-llm.md` and `faq.md`
- Fixed `train.py` task registration for `lm` task

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

- **Modern Architecture Features**:
    - Grouped Query Attention (GQA) for balanced performance and memory efficiency
    - SwiGLU activation function in MLP layers
    - Unified QKV projection optimization for improved memory layout and throughput
    - RMSNorm support as alternative normalization layer

- **Tokenization & Training**:
    - BPETokenizer for production-ready subword tokenization
    - LanguageModelingTask for language model training
    - Automatic BF16/FP16 mixed precision detection and support
    - Robust NaN loss handling

- **Inference Capabilities**:
    - KV Cache support in MHA, TransformerBlock, and DecoderModel
    - Top-k and Top-p sampling strategies
    - Greedy search decoding (temperature=0)
    - Dynamic sequence length support
    - Simple autoregressive generation loop

- **Testing & Quality**:
    - 262 comprehensive unit test cases covering all core functionality
    - Functional tests for causal masking, KV cache consistency, architecture properties
    - Convergence tests for training validation
    - Mock-free test design using real components

- **Documentation**:
    - Comprehensive ROADMAP.md (405 lines) with 15 development stages
    - Priority levels (P1-P4), timelines, and success metrics
    - Detailed training framework documentation (8 comprehensive guides)
    - CPU-friendly LLM tutorial and development guide
    - FAQ document covering core topics
    - ADR (Architecture Decision Records) system with 4 initial records
    - PR template for standardized contributions

### Changed

- **Architecture Optimization**:
    - Refactored DecoderModel with configurable components
    - Optimized padding mask and KV cache handling
    - Improved GradScaler usage for bfloat16

- **Training Enhancements**:
    - Enhanced TrainingEngine with improved callback system
    - Performance monitoring and logging improvements
    - Auto AMP dtype resolution for CPU-only environments

- **Code Quality**:
    - Enhanced Ruff linting rules (SIM, RUF, PTH for pathlib)
    - PEP 561 compliance with py.typed marker
    - Standardized punctuation across documentation
    - Project structure improvements for modularity

- **Documentation**:
    - Updated Quick Start example from regression to lm task
    - Enhanced feature descriptions with technical highlights
    - Better cross-references and examples throughout

### Fixed

- **Core Issues**:
    - All 262 test regressions resolved
    - PositionalEncoding support for `start_pos` in incremental generation
    - MLP SwiGLU operation order for numerical consistency
    - Prompt truncation respecting `max_seq_len` with new tokens
    - Device mismatch in MLP when norm instance provided
    - Auto AMP dtype test failures on CUDA environments

- **Quality & Stability**:
    - Type checking issues across the codebase
    - Memory management in distributed training
    - Edge cases in attention masking and positional encoding
    - Device comparisons robustness (comparing device.type)
    - Failed runs on CPU-only environments

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
