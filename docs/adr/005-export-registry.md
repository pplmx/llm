# 5. Export Registry Parity with BACKEND_REGISTRY

Date: 2026-07-15

## Status

Accepted

## Context

The 2026-07-12 technical due-diligence (Finding BH) flagged that
`src/llm/export/` was the only component surface in the project
without a registry + setuptools-entry-point pattern. Every other
extension point followed the same shape:

| Surface | Registry | Entry-point group |
|---------|----------|-------------------|
| Models | `MODEL_REGISTRY` | `llm.models` |
| Attention | `ATTENTION_REGISTRY` | (in-code) |
| MLP | `MLP_REGISTRY` | (in-code) |
| Norm | `NORM_REGISTRY` | (in-code) |
| Generation backends | `BACKEND_REGISTRY` | `llm.generation_backends` |
| Data sources | `SOURCE_REGISTRY` | `llm.data_sources` |
| Tasks | `TASK_REGISTRY` | `llm.tasks` |
| **Export backends** | **(none)** | **(none)** |

`src/llm/export/` shipped a single `export_to_onnx` function with no
pluggability story. Adding a new target (e.g. `torch.compile`,
`vLLM`, `TensorRT-LLM`, `torch.export`, `OpenVINO`) would have
required editing `export/__init__.py` directly.

## Decision

1. **Add `EXPORT_REGISTRY` to mirror `BACKEND_REGISTRY`.** Both
   registries use the same generic `runtime.Registry[T]`, the same
   `load_entry_point_registry("llm.<group>", REGISTRY)` bootstrap,
   and the same factory contract: ``(model, output_path, **kwargs)
   -> Path``.
2. **`onnx` stays as a built-in** registered in code. The audit
   tier-3 ticket for the registry (#32) explicitly carved this out:
   the built-in is the reference implementation and must resolve
   even when entry-point machinery is unavailable (e.g. minimal
   test environments). Plugins that try to claim `onnx` raise
   loudly, which is intentional.
3. **The first entry-point-registered target is `torchscript`**
   (Tier 3 #33). Registered via
   `pyproject.toml`'s `llm.export_backends` group as
   `torchscript = "llm.export.torchscript:build_torchscript_exporter"`.
   The entry point points at the **factory** itself, matching
   `BACKEND_REGISTRY`'s `eager` / `batched` convention (the loader
   calls `ep.load()` and registers the returned callable directly).
4. **`export_model(name, model, output_path, **kwargs)`** is the
   single dispatch entry point. It calls
   `ensure_exporters_registered()` (idempotent) then resolves the
   factory by name. Unknown names raise `ValueError` with the
   available names listed.
5. **Shared `ExportCacheWrapper`** in `src/llm/export/_wrapper.py`
   replaces the private `_ExportWrapper` in `onnx.py`. The wrapper
   forces `use_cache=False` and unwraps the `(logits, kv_cache)`
   tuple to just `logits` so trace-based exporters (ONNX, TorchScript)
   see a clean graph. Adding a future trace-based target requires
   no new wrapper code.
6. **Backward compatibility.** `export_to_onnx`, `verify_onnx`,
   and `get_onnx_info` remain in `llm.export.__init__` as the stable
   ONNX API. The registry is additive — no caller of the old API
   needs to change.

## Consequences

**Advantages**:

- Third-party export targets (e.g. an internal `tensorrt_llm`
  plugin) plug in via `pyproject.toml` without forking
  `llm/export/`. The pattern is identical to extending
  `BACKEND_REGISTRY`.
- The deployment story gains a `torch.jit` artifact path that
  doesn't need ONNX runtime. Verified end-to-end by the
  `tests/export/test_torchscript_export.py` roundtrip suite.
- `ExportCacheWrapper` is the single place that knows how to
  neutralise the cache contract for tracing. Future trace-based
  targets (e.g. `torch.export`) just compose it.
- mkdocstrings picks up `EXPORT_REGISTRY`, `export_model`,
  `ensure_exporters_registered`, `build_onnx_exporter`, and
  `build_torchscript_exporter` from their docstrings — no extra
  reference-page maintenance.

**Limitations**:

- The script method (`torch.jit.script`) is wired but `xfail`-
  tracked for `DecoderModel` (its `PositionalEncoding` uses
  dynamic attribute access that the TorchScript compiler rejects).
  Purely-tensor submodules would script cleanly; full-model
  scripting is a separate slice if a deployment path needs it.
- The registry contract is intentionally narrow — any richer
  surface (e.g. a class with `.export(model, path, **kwargs)`)
  is trivial to wrap with a thin factory function but not
  prescribed by the type alias. Future targets that need richer
  APIs can either widen the alias or ship their own factory
  shape with a custom dispatch.
- `EXPORT_REGISTRY` and `BACKEND_REGISTRY` are independent objects.
  A future "unified `COMPONENT_REGISTRY`" would deduplicate code
  but is not justified yet — the two surfaces have different
  lifecycles and different entry-point groups.

## References

- `docs/audits/2026-07-12-technical-due-diligence.md` §Finding BH.
- Tier 3 #32 (registry foundation) — commit `3a371ff`.
- Tier 3 #33 (TorchScript target) — commit `bc4fe1d`.
- `docs/reference/architecture.md` §Plugin Kernel — updated row.
