# TorchScript export target (Tier 3 #11)

## Source

ROADMAP.md §阶段十四 (item 3: "TorchScript 导出 (可扩 `export/`
模块)"). Builds directly on the `EXPORT_REGISTRY` foundation shipped
in T3 #32.

## Description

Add a TorchScript export target as the first non-built-in registrant
of `EXPORT_REGISTRY` (i.e. via the `llm.export_backends` setuptools
entry-point group) to prove the registry is genuinely pluggable.

The shipped factory lives in-tree (it's just another module), but
its entry in `pyproject.toml` is what wires it up — the loader
resolves the entry point with `ep.load()` and registers the returned
callable into `EXPORT_REGISTRY`. This is the same pattern used by
`BACKEND_REGISTRY`'s `eager` and `batched` entries.

TorchScript ships with PyTorch, so this slice adds no new runtime
dependencies. It is the smallest possible concrete export target
that demonstrates the registry's plug-in story and gives the
deployment story a `torch.jit` artifact path that doesn't need
ONNX runtime.

## Design

- `src/llm/export/_wrapper.py` (new, ~30 lines): extract
  `_ExportWrapper` from `onnx.py` into a shared
  `ExportCacheWrapper` so both backends can use it without
  duplication. The wrapper forces `use_cache=False` and unwraps
  the `(logits, kv_cache)` tuple to just logits.

- `src/llm/export/torchscript.py` (new):
    - `export_to_torchscript(model, output_path, *, method="trace",
      example_inputs=None, **kwargs) -> Path` — thin wrapper over
      `torch.jit.trace` (default) or `torch.jit.script`.
    - `build_torchscript_exporter(model, output_path, **kwargs) -> Path`
      — the factory shape expected by `EXPORT_REGISTRY`.

- `pyproject.toml`:
    - `[project.entry-points."llm.export_backends"]` now lists
      `torchscript = "llm.export.torchscript:build_torchscript_exporter"`
      — the entry point points at the factory itself, which is the
      contract `llm.runtime.plugins.load_entry_point_registry`
      expects.

- `src/llm/export/__init__.py`: re-export `export_to_torchscript`
  and `build_torchscript_exporter`.

- Tests (`tests/export/test_torchscript_export.py`):
    - `build_torchscript_exporter` writes a `.pt` file that loads via
      `torch.jit.load`.
    - The artifact runs forward and matches the eager-model output
      (numerical equivalence within FP tolerance).
    - `ensure_exporters_registered()` resolves both `onnx` and
      `torchscript` (proves the entry-point load path works).
    - `export_model("torchscript", model, output_path)` goes through
      the dispatch path.
    - `test_script_method_works` is `pytest.xfail`-marked: scripting
      a full `DecoderModel` currently fails on `PositionalEncoding`
      (TorchScript can't see `pos_embedding`). Documented as out of
      scope; the trace path is the supported one.

- Docs:
    - `docs/audits/2026-07-12-tickets/README.md`: add row #33, bump
      the Tier 3 counter.

## Acceptance criteria

- [x] `EXPORT_REGISTRY.get("torchscript")` resolves after
      `ensure_exporters_registered()`.
- [x] `export_model("torchscript", model, tmp_path / "m.pt")` writes
      a loadable TorchScript artifact.
- [x] The artifact's forward matches the eager model (numerical
      equivalence, `torch.allclose`).
- [x] Registration is via the `llm.export_backends` entry point in
      `pyproject.toml` (proves the plug-in path, not the in-code
      register call).
- [x] Tests cover: factory dispatch, entry-point bootstrap,
      unknown-target error, script vs trace method.
- [x] README marks slice done.

## Out of scope (intentional)

- `torch.compile` / `vLLM` / `TensorRT-LLM` targets — each is a
  separate slice with a real dependency surface.
- Optimizations (constant folding, freezing, quantization) — those
  are `**kwargs` the user can already pass through.
- A scripted-only mode for `DecoderModel` — the model's
  `PositionalEncoding` uses Python `isinstance` and dynamic
  attribute access that TorchScript's compiler rejects. Smaller,
  purely-tensor submodules would script; full-model scripting is
  a separate slice if a deployment path needs it.

## Estimate

~0.5 day. The wrapper is ~30 LOC and the tests are mechanical;
the entry-point glue is the only new pattern.

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `export`, `runtime`,
`registry`, `torchscript`
