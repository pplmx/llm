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
  ├─ Daily development gate         → make test-fast   (excludes heavy+e2e)
  ├─ Full gate before claiming done → make test        (ALL tests must pass)
  ├─ Lint/format                    → make ruff        (format + lint --fix)
  ├─ Type check                     → make ty          (uvx ty check src/llm/)
  ├─ Dependency change              → uv lock && uv lock --check
  ├─ Docstring examples changed     → make test-doc    (doctests src/llm)
  └─ e2e workflows                  → make test-e2e
```

**AGENTS.md verification order after changes: `make test` → `make ruff` →
`make test`.** Ruff can rewrite files, so tests run again after it.

## Make targets (Makefile)

| Target                   | Effect                                                       |
| ------------------------ | ------------------------------------------------------------ |
| `make test`              | `uv run pytest` — full suite (600+ tests, must be all green) |
| `make test-fast`         | excludes `heavy` + `e2e` markers                             |
| `make test-e2e`          | only `e2e` marker                                            |
| `make test-quick`        | only `quick` marker                                          |
| `make test-cov`          | coverage with `--cov-fail-under=77` (CI floor, ticket #11)   |
| `make test-doc`          | `pytest --doctest-modules src/llm/`                          |
| `make ruff`              | `uvx ruff format .` then `uvx ruff check . --fix`            |
| `make ty`                | `uvx ty check src/llm/` — matches the CI lint gate           |
| `make sync` / `make dev` | `uv sync` default groups / all groups+extras                 |
| `make init`              | deps + `uvx prek install` hooks                              |

## What CI enforces (.github/workflows/ci.yml)

- **lint**: `uv lock --check`, `uvx ruff format --check .`,
  `uvx ruff check .`, `uvx ty check src/llm/` (hard gate since 2026-08-08,
  ticket 08-ty-in-ci), `py_compile` of every `examples/*.py`.
- **doctest**: `uv run pytest --doctest-modules src/llm/`.
- **build-and-test**: ubuntu/macos/windows × `uv sync --frozen` + `uv build`
    - `make test-cov`.
- **security**: `pip-audit --strict` on the exported lock set +
  `bandit -r src/llm/ --severity-level high`.
- Docker build excludes test/docs/streaming dependency groups.

## Pre-commit hooks (prek, installed via `make init`)

end-of-file-fixer, trailing-whitespace, check-toml/yaml, merge-conflict,
mixed-line-ending (LF), commitizen (Conventional Commits), rumdl markdown
lint (CHANGELOG.md + docs/audits excluded), ruff check+format, **ty type
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
