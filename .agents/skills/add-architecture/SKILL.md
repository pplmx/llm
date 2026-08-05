---
name: add-architecture
description: >
  Step-by-step workflow for adding a new model family / model builder to this
  repo (Python LLM training framework). Use when the user wants to add a new
  model type, extend ModelFactory / MODEL_REGISTRY, register a builder under
  the `llm.models` entry-point group, or asks "how do I add a new model"
  (加模型 / 新架构 / 新 model family). Covers builder scaffold, ModelConfig
  validation, entry-point registration, and verification.
---

# Add a New Model Architecture

## Key insight: one `DecoderModel`, many builders

This repo does NOT have per-family model classes. There is a single
`DecoderModel` (`src/llm/models/decoder.py`) whose behavior is configured
through kwargs (`attn_impl`, `mlp_impl`, `norm_impl`, GQA via `num_kv_heads`,
SwiGLU via `use_glu`, MoE via `num_experts`/`top_k`). New "architectures" are
added as **model builders** registered in `MODEL_REGISTRY`.

| Layer       | File                                                   | Contract                                 |
| ----------- | ------------------------------------------------------ | ---------------------------------------- |
| Builders    | `src/llm/runtime/model_factory.py`                     | `build_xxx(**model_kwargs) -> nn.Module` |
| Registry    | `MODEL_REGISTRY` (same file)                           | stores callables, not classes            |
| Entry point | `pyproject.toml` `[project.entry-points."llm.models"]` | `name = "module:builder_fn"`             |
| Config      | `ModelConfig` in `src/llm/training/core/config.py`     | validation via `check_consistency`       |

Built-ins: `decoder = build_decoder`, `regression_mlp = build_regression_mlp`.

## Workflow

1. **Prefer config over code first.** If the new family only differs in
   attention/MLP/norm wiring, you likely need a new *component* (see the
   `add-component` skill), not a new builder.
2. **Write the builder.** Add `build_<family>(...) -> nn.Module` in
   `src/llm/runtime/model_factory.py` (or a new module imported there).
   Accept explicit architecture kwargs; keep construction deterministic from
   kwargs alone — configs and checkpoints reconstruct models through this
   path (see `compat/hf_loader.py`).
3. **Register.** In-code: `MODEL_REGISTRY.register("<family>", build_<family>)`.
   Third-party builders go through the `llm.models` entry-point group in
   `pyproject.toml` (loader: `llm.runtime.plugins.load_entry_point_registry`).
   After editing entry points run `uv sync` so installed package metadata is
   refreshed.
4. **Validate config.** If the family constrains options (e.g. only certain
   `attn_impl`), add checks to `ModelConfig.check_consistency` so bad
   combinations fail at config-load time, not mid-training.
5. **Verify.**
    - Unit tests in `tests/models/` — build the model, run a forward pass,
   assert output shapes/values (never bare `isinstance` checks).
    - Reuse `decoder_model_kwargs(**overrides)` from `tests/support/models.py`
   for `DecoderModel`-based builders.
    - `uv run pytest tests/models -q`, then the full loop (see `run-ci` skill).

## Architecture boundaries (need an ADR before crossing)

- Multimodal / 3D parallelism: no registry exists yet; do NOT patch
  `DecoderModel` — design `MultimodalDataModule` etc. first (ROADMAP 阶段十二).
- Do not restore deleted compat shims; new code goes through `runtime/`
  and factories.
- MLA (`attn_impl=mla`) KV-cache support is a placeholder implementation;
  see `docs/audits/2026-07-12-tickets/31-mla-kv-cache.md`.

## Reference docs

- `docs/reference/architecture.md` — layering, data flow, registry mechanism
- `docs/adr/` — locked decisions (GQA, KV cache, Paged Attention, QLoRA…)
