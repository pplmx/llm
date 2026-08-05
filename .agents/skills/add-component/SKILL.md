---
name: add-component
description: >
  Workflow for adding components to this repo's plugin-kernel registries
  (Python LLM training framework): attention/MLP/norm variants in core/,
  generation backends, export backends, PEFT methods, and streaming text
  sources. Use when the user wants a new attention variant, normalization,
  MLP variant, inference backend, export target, PEFT method, or data source
  (新注意力 / 新算子 / 新 backend / 新导出格式 / 新数据源).
---

# Add a Registry Component

This repo is a plugin kernel: extension points are `Registry` instances
(`src/llm/runtime/registry.py`: `register` / `get` / `names` / `in`) plus
setuptools entry-point groups loaded by `llm.runtime.plugins`. Pick the right
registry, implement against its contract, register, test.

## Registry map

| Registry             | File                                                                          | Contract                                                                                                           | Entry-point group                     |
| -------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `ATTENTION_REGISTRY` | `src/llm/core/registry.py`                                                    | attention module class                                                                                             | — (in-code)                           |
| `MLP_REGISTRY`       | `src/llm/core/registry.py`                                                    | MLP module class                                                                                                   | — (in-code)                           |
| `NORM_REGISTRY`      | `src/llm/core/registry.py`                                                    | **factory callable** `(*args, **kwargs) -> nn.Module`, not a class                                                 | — (in-code)                           |
| `BACKEND_REGISTRY`   | `src/llm/generation/registry.py`                                              | `GenerationBackend` subclass (ABC in `generation/backends.py`) + factory `build_xxx_backend(engine=..., **kwargs)` | `llm.generation_backends`             |
| `EXPORT_REGISTRY`    | `src/llm/export/registry.py`                                                  | exporter factory `build_xxx_exporter(...)` (see `export/onnx.py`, `export/torchscript.py`, `export/gguf/`)         | `llm.export_backends`                 |
| `PEFT_REGISTRY`      | re-exported via `core/registry.py`; methods in `src/llm/core/peft/methods.py` | `PEFTMethod` dataclass record                                                                                      | `llm.peft_methods`                    |
| `SOURCE_REGISTRY`    | `src/llm/data/sources.py`                                                     | builder `(data_config) -> TextSource`                                                                              | `llm.data_sources`                    |
| `MODEL_REGISTRY`     | `src/llm/runtime/model_factory.py`                                            | builder callable                                                                                                   | `llm.models` (see `add-architecture`) |
| `TASK_REGISTRY`      | `src/llm/training/task_registry.py`                                           | task+DataModule pairing                                                                                            | see `add-task`                        |

## Workflow

1. **Implement** in the layer that owns the concern:
    - attention/MLP/norm → new module under `src/llm/core/` (e.g.
   `core/attn/<name>.py`). Keep the forward signature compatible with the
   existing implementations so `DecoderModel` can swap them via
   `attn_impl` / `mlp_impl` / `norm_impl`.
    - generation backend → subclass `GenerationBackend`, implement
   `generate(...)`; factory returns an instance.
    - export backend → follow `export/gguf/` layering (keep format core
   torch-free when possible); ADR-005 documents the pattern.
    - text source → subclass `TextSource` (`iter_texts(skip=0)`,
   `source_fingerprint()` for checkpoint-resume validation).
2. **Register.** Use `@register_attention("<name>")` /
   `@register_mlp("<name>")` decorators, or `REGISTRY.register(...)` inside an
   `ensure_*_registered()` guard (the pattern used by norms/backends/sources).
   Attention implementations MUST also call
   `set_attention_kv_cache_capability("<name>", supports)` at import time —
   `ModelConfig.check_consistency` raises on any `attn_impl` without a
   capability record.
3. **Expose via entry point** only for third-party-style extension surfaces
   (`llm.generation_backends`, `llm.export_backends`, `llm.data_sources`,
   `llm.peft_methods`, `llm.models`). Built-ins stay registered in code so
   they resolve even without entry-point machinery (minimal test envs).
   After editing `pyproject.toml` entry points: `uv sync`.
4. **Test** (see `write-test` skill): construct through the registry
   (`REGISTRY.get("<name>")`) to prove registration, then assert concrete
   forward-pass behavior/values. Run the owning test dir
   (`tests/core`, `tests/generation`, `tests/export`, `tests/data`).
5. **Docs**: update `docs/reference/architecture.md` plugin-kernel table when
   adding a new registry surface.

## Pitfalls

- `NORM_REGISTRY` stores factories, not classes — the block calls
  `factory(hidden_size, eps=..., **factory_kwargs)`.
- Do NOT create duplicate registries or restore deleted compat shims.
- Registry lookups raise on unknown names by design; do not silently
  fall back.
