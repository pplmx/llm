# Export registry (Tier 3 #10)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding BH ("No
`EXPORT_REGISTRY` symmetry"), ROADMAP §阶段十四 ("Export registry
(与 `BACKEND_REGISTRY` 对称)").

## Description
`src/llm/export/` today ships a single function
(`export_to_onnx`) with no registry. `BACKEND_REGISTRY`,
`SOURCE_REGISTRY`, `MODEL_REGISTRY`, `ATTENTION_REGISTRY`,
`MLP_REGISTRY`, and `NORM_REGISTRY` all follow the same
registry + setuptools-entry-point pattern. The export tier is the
only component surface without one.

This slice adds the missing registry so future export targets
(`torch.compile`, `vLLM`, `TensorRT-LLM`, `torch.export`,
`OpenVINO`, ...) can plug in without forking `export/`. The
existing ONNX export becomes the first built-in target — every
caller that imported `export_to_onnx` continues to work.

## Design

Mirror `generation/registry.py`:
- `EXPORT_REGISTRY: Registry[ExportBackendFactory]` (callable
  ``(model, output_path, **kwargs) -> Path``).
- `build_onnx_exporter` factory — thin wrapper over the existing
  `export_to_onnx` so the registry contract is ``(model, output_path, **kwargs) -> Path``.
- `ensure_exporters_registered()` idempotent bootstrap; loads the
  `llm.export_backends` entry-point group on top of the built-in
  `onnx` registration.
- High-level `export_model(name, model, output_path, **kwargs)`
  resolves the factory by name and forwards kwargs.

Entry point group `llm.export_backends` in `pyproject.toml`. The
ONNX target stays in-tree so a vanilla install works without
plugin packages.

## Acceptance criteria
- [ ] `EXPORT_REGISTRY` exists and the `onnx` factory resolves.
- [ ] `export_model("onnx", model, output_path)` is a drop-in
      replacement for `export_to_onnx(model, output_path)`.
- [ ] `pyproject.toml` declares the `llm.export_backends` group
      with an empty list (so plugin authors have a target).
- [ ] Tests cover: built-in onnx registration, factory dispatch,
      unknown-target error, and the entry-point-bootstrap idempotency.
- [ ] `docs/audits/2026-07-12-tickets/README.md` marks this slice done.

## Out of scope (intentional)
- Real `torch.compile` / `vllm` / `TensorRT-LLM` export targets —
  each is its own slice with non-trivial dependency surface.
- Backward-compat for the public `export_to_onnx` symbol — we
  keep it as a thin re-export so existing callers don't break.

## Estimate
~1 day for the foundation slice (no new export target).

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `export`, `runtime`,
`registry`
