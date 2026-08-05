---
name: write-test
description: >
  Write tests for this repo (Python LLM training framework) using its
  conventions: pytest + tests/support helpers, shared conftest fixtures,
  markers (quick/slow/heavy/e2e/gpu), real modules instead of mocks, strong
  assertions. Use when the user wants unit/integration/e2e tests, asks "how
  do I test this", "写测试 / 加测试 / 补回归测试".
---

# Write Tests

## Layout

`tests/` mirrors `src/llm/` (core/, data/, models/, training/, generation/,
serving/, export/, quantization/, evaluation/, compat/, …). Namespace
packages are enabled — no `__init__.py` needed; keep file names unique per
directory or rely on `consider_namespace_packages`.

## Shared helpers — never reinvent

| Helper                                                | Where                         | Use for                                                                                   |
| ----------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `StubTokenizer`, `LineTokenizer`                      | `tests/support/tokenizers.py` | all tokenizer needs; **inline `_Tok`/`_LineTokenizer` classes are a banned anti-pattern** |
| `decoder_model_kwargs(**overrides)`                   | `tests/support/models.py`     | canonical tiny `DecoderModel` kwargs                                                      |
| `SAMPLE_TEXT_CORPUS`, `DEFAULT_INFERENCE_CORPUS`      | `tests/support/corpus.py`     | fixed text fixtures                                                                       |
| `cuda_usable()`, `ALL_DEVICES`, `cuda_device_count()` | `tests/support/devices.py`    | device selection that treats visible-but-OOM GPUs as unusable                             |
| root `conftest.py` fixtures                           | `tests/conftest.py`           | `device`, `stub_tokenizer`, `line_tokenizer`, `model_and_tokenizer`                       |
| `model_kwargs` / `decoder_model` fixtures             | `tests/models/conftest.py`    | parametrize via `@pytest.mark.parametrize(..., indirect=True)`                            |
| `sample_text_tokenizer` / `line_tokenizer`            | `tests/data/conftest.py`      | data-layer tests                                                                          |

## Markers (pyproject.toml)

`quick` (<1s) · `slow` (1–30s) · `heavy` (>30s, maybe GPU) · `e2e` ·
`integration` (external deps e.g. HuggingFace) · `gpu` · `multi_gpu` ·
`full_cluster` (8 GPUs) · `need_gpu(n)`. GPU markers auto-skip when hardware
is missing; CI gates run `not heavy and not e2e` for daily work.

## Rules (AGENTS.md 测试约定)

1. **No mocks by default** — run the real module; stub only I/O boundaries
   (tokenizers, corpora) with the support helpers above.
2. **Strong assertions**: assert concrete behavior/values (shapes, numbers,
   byte-exact payloads, error messages). Pure `isinstance` / `is not None`
   assertions are forbidden.
3. **Test behavior, not implementation** — regression tests must fail if the
   bug returns, even after refactors.
4. Parametrize across devices via `ALL_DEVICES` rather than hardcoding CPU.
5. Tiny configs everywhere: small vocab/hidden/layers/seq via
   `decoder_model_kwargs` overrides; tests must stay in their marker budget.
6. Registry extensions: prove registration through the registry API
   (`REGISTRY.get(...)`) then exercise real forward behavior.
7. Streaming/checkpoint work: round-trip `get_checkpoint_state` →
   `load_checkpoint_state` and assert resumed iteration matches.

## Commands

```bash
uv run pytest tests/<dir> -q                    # targeted
uv run pytest tests/<dir> -m quick -q           # marker-filtered
uv run pytest -m "not heavy and not e2e" -q     # daily gate (make test-fast)
uv run pytest -m e2e -q                         # e2e only (make test-e2e)
uv run pytest tests/path/test_x.py::test_y -q   # single test
```

After adding tests: run them, then the full `run-ci` gate order
(`make test` → `make ruff` → `make test`).
