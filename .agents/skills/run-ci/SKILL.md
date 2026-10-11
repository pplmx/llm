---
name: run-ci
description: >
  Run verification gates for this repo (Python LLM training framework):
  pytest suites, ruff format/lint, ty type check, uv lock sync, doctests,
  pre-commit (prek) hooks, coverage floor. Use when the user says "run CI",
  "跑测试", "检查代码", "verify before commit/push", or wants to ensure code
  passes all quality gates.
---

# Run CI / Verify Code Quality

Package manager is **uv only** (never pip/requirements.txt). In sandboxed
environments where `/root/.cache` is read-only, prefix commands with
`UV_CACHE_DIR=/tmp/uv-cache`.

## Decision tree

```text
Start
  ├─ Quick sanity while iterating   → uv run pytest <target-dir> -q
  ├─ Daily development gate         → just test-fast   (excludes heavy+e2e)
  ├─ Full gate before claiming done → just test        (ALL tests must pass)
  ├─ Lint/format                    → just ruff        (format + lint --fix)
  ├─ Full static gate (matches CI)  → just check       (lock + ruff --check + ty + examples)
  ├─ Type check                     → just ty          (uvx ty check src/llm/)
  ├─ Dependency change              → uv lock && uv lock --check
  ├─ Docstring examples changed     → just test-doc    (doctests src/llm)
  └─ e2e workflows                  → just test-e2e
```

**AGENTS.md verification order after changes: `just test` → `just ruff` →
`just test`.** Ruff can rewrite files, so tests run again after it.

## Just targets (Justfile)

| Target                   | Effect                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `just test`              | `uv run --frozen pytest` — full suite (all markers)                                                  |
| `just test-fast`         | CPU-clean daily gate: excludes `heavy` + `e2e` + all GPU markers                                     |
| `just test-gpu`          | only GPU markers (`gpu`/`multi_gpu`/`need_gpu`/`full_cluster`)                                       |
| `just test-heavy`        | only `heavy` marker                                                                                  |
| `just test-e2e`          | only `e2e` marker                                                                                    |
| `just test-quick`        | only `quick` marker                                                                                  |
| `just test-cov`          | full suite coverage `--cov-fail-under=77` (scheduled deep gate)                                      |
| `just test-cov-fast`     | core CPU suite coverage (per-push CI gate), `--cov-fail-under=77`                                    |
| `just test-doc`          | `uv run --frozen pytest --doctest-modules src/llm/`                                                  |
| `just ruff`              | `uvx ruff format .` then `uvx ruff check . --fix`                                                    |
| `just check`             | `uv lock --check` + `ruff format --check` + `ruff check` + `ty` + examples compile (mirrors CI lint) |
| `just ty`                | `uvx ty check src/llm/` — matches the CI lint gate                                                   |
| `just sync` / `just dev` | `uv sync` default groups / all groups+extras                                                         |
| `just init`              | deps + `uvx prek install` hooks                                                                      |

## How tests handle GPU-less hosts

`tests/conftest.py::pytest_collection_modifyitems` auto-skips every GPU-requiring
marker when the host lacks enough *usable* GPUs (via `tests.support.devices`):
`gpu` (<1), `multi_gpu` (<2), `need_gpu(n)` (<n), `full_cluster` (<8). So a CPU
CI runner or laptop **skips** (never fails/hangs) GPU tests. `just test-fast` /
`just test-cov-fast` additionally exclude GPU markers so a CPU host never even
collects them. GPU tests run via `just test` / `just test-gpu` on a GPU host.

## What CI enforces (.github/workflows/ci.yml)

- **lint**: `uv lock --check`, `uvx ruff format --check .`,
  `uvx ruff check .`, `uvx ty check src/llm/` (hard gate since 2026-08-08,
  ticket 08-ty-in-ci), `py_compile` of every `examples/*.py`.
- **doctest**: `uv run pytest --doctest-modules src/llm/`.
- **build-and-test**: ubuntu/macos/windows × `uv sync --frozen` + `uv build`
    - `just test-cov-fast` (core CPU suite + coverage, per-push gate).
- **test-full**: ubuntu, `just test-cov` (full suite + coverage) — runs on the
  weekly schedule + manual dispatch, not per-push (heavy/e2e too slow).
- **security**: `pip-audit --strict` on the exported lock set +
  `bandit -r src/llm/ --severity-level high`.
- Docker build excludes test/docs/streaming dependency groups.

## Pre-commit hooks (prek, installed via `just init`)

end-of-file-fixer, trailing-whitespace, check-toml/yaml, merge-conflict,
mixed-line-ending (LF), commitizen (Conventional Commits), rumdl markdown
lint (CHANGELOG.md + docs/audits excluded), ruff check+format, **uv-lock**
(runs `uv lock --check` — fails a commit whose `uv.lock` is stale), **ty type
check**, **doctests**. Hooks run on every commit; `--no-verify` is forbidden.

## Commit workflow (AGENTS.md)

1. Write message to `commit_message.txt` (Conventional Commits; commitizen
   validates). Body describes the human author's intent; **no AI
   co-author lines**.
2. `git commit -F commit_message.txt`, then delete the file.
3. Commit only when the user explicitly asked (or the active workflow, e.g.
   graph-engineering loop, mandates it). Never force-push `main`.
4. RIL graph commits reference node ids, e.g.
   `fix(core): ... (RIL TASK-001, ISS-001)`.

## Failure triage

- Test failure → root-cause and fix; **never** delete/skip tests, lower
  assertions, or edit thresholds to force green.
- `uv lock --check` fails → `uv lock` then commit `uv.lock`.
- ruff rewrites → re-run the test gate.
- If a fix cannot be made reliable, revert the change (`git checkout` /
  `git revert`) rather than leaving a broken tree.
