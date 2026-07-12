# Technical Due Diligence — `llm`

**Reviewer perspective**: Principal Engineer / Software Architect / prospective long-term maintainer
**Date**: 2026-07-12
**Repo state at review**: branch `main`, v0.0.5, 115 src files (~12.4k LOC) + 98 test files (~10.6k LOC, 509 tests)

> **Status**: snapshot review. Tier 1 items have been converted to GitHub issues with the
> `audit-2026-07` label and the `v0.0.6-audit-followup` milestone. Tier 2/3 items are reflected
> in ROADMAP.md. This document is intentionally archival — do not edit after-the-fact;
> write a new audit instead.

---

## 1. Executive Summary

`llm` is a **single-author Beta-stage PyTorch framework** for training and serving decoder-only
LLMs, layered around a clean **plugin/registry kernel** (`runtime/`) and setuptools entry
points. The architecture is unusually disciplined for its size — strict `src/` layout,
ADR-driven decisions, a registry pattern unifying attention/MLP/norm/models/data-sources/
generation backends, a `TaskRegistry` decoupling tasks from data modules, and a
`CheckpointContributor` protocol for extra-state — and it is backed by ~509 tests.

However, the project is **thin in three respects**: (1) only one human maintainer
(bus-factor 1) with no contributor guide beyond `CONTRIBUTING.md`; (2) several advanced
subsystems are **wired but unverified at scale** (FSDP, Paged Attention full path,
MLA+KV cache, multimodal); (3) **operational maturity is light** — no observability
beyond Prometheus, no real auth, no benchmark suite, no model registry/HF Hub publish
pipeline. The biggest structural risks are concentrated in the training engine and
serving layer, where DDP plumbing, callback ordering, and
`ContinuousBatchingEngine.step` correctness are non-trivial and lightly commented.

The most pragmatic path forward is: stabilize what exists (test coverage + docs),
harden the serving tier (real auth, rate limiting, request IDs, structured errors),
gate the missing subsystems behind explicit "experimental" markers, and only then
invest in scaling features (FSDP e2e, multimodal, Hub publish).

**Overall grade**: **B+** (architecture) / **B-** (operations) / **C+** (community/process).

---

## 2. Project Positioning

### Current state
- README and ROADMAP position `llm` as a **modular PyTorch framework for training and
  experimenting with LLMs** — i.e., an educational/research framework, not a production
  serving platform.
- Target audience: developers, researchers. PyPI version 0.0.5, classifiers include
  `Development Status :: 4 - Beta`.
- Dual license (MIT / Apache-2.0) — appropriate for a PyTorch ecosystem project.

### Strengths
- Clear positioning ("modular and extensible", "for experimentation").
- Layered architecture makes it easy to onboard contributors.

### Weaknesses
- README "Features" lists OpenAI-compatible serving, KV cache, Prometheus metrics, etc. —
  implying a **production-capable serving tier** that the code does not yet substantiate
  (no auth beyond a single shared API key, no rate-limit middleware, no request-ID
  propagation, FastAPI runs with `reload=True` in `main()` — `src/llm/serving/api.py:405`).
- ROADMAP labels P0 = "pretraining productization" yet serving security/operational
  concerns are scattered and unmentioned.
- No stated **non-goals**; readers cannot tell what `llm` deliberately doesn't do.

### Severity / priority
**MEDIUM**. Wrong positioning leads to misuse and support burden.

### Recommendations
1. Add a short "Project status & non-goals" section to README: explicitly mark serving
   tier as "Beta — not hardened for public internet exposure".
2. Drop `reload=True` from `main()` (`src/llm/serving/api.py:405`) before any release tag.
3. Re-classify serving in PyPI classifiers
   (`Topic :: Internet :: WWW/HTTP :: WSGI :: Application` is missing).

---

## 3. Architecture

### 3.1 Layering and module boundaries

The project follows a clean six-layer separation:

| Layer | Path | Owner concern |
|---|---|---|
| Core | `src/llm/core/` | Attention, MLP, Norm, RoPE, ALiBi, KV cache, LoRA, QLoRA, MoE |
| Models | `src/llm/models/` | `DecoderModel` assembly |
| Data | `src/llm/data/` | Dataset / DataModule / TextSource |
| Training | `src/llm/training/` | Engine, Task, DDP/FSDP, callbacks |
| Generation | `src/llm/generation/` | Sampling, eager + batched backends |
| Runtime | `src/llm/runtime/` | Registry kernel, entry-point discovery, ModelFactory, CheckpointContributor |
| Serving | `src/llm/serving/` | FastAPI + ContinuousBatchingEngine |
| Evaluation | `src/llm/evaluation/` | Offline eval runner + per-task evaluators |
| Compat | `src/llm/compat/` | HF weight loading |
| Export | `src/llm/export/` | ONNX export |

This is **legitimately good architecture** for a small framework — boundaries are real,
not aspirational, and the registry pattern (`runtime/registry.py`, 47 lines) is the
smallest thing that could possibly work.

### 3.2 Plugin kernel (the centerpiece)

- `runtime/registry.py` defines a generic `Registry[T]` with `register/get/names/__contains__`.
  It is generic over the entry type, not the registration function — correct.
- `runtime/plugins.py` discovers entry points via `importlib.metadata`. The dual-API
  handling (`hasattr(eps, "select")`) is correct for Python 3.10+ vs older.
- `runtime/model_factory.py` exposes `ModelFactory.from_config()` and
  `ModelFactory.build()`; **built-in models register only via setuptools entry points**,
  which is the right call — it eliminates the "two ways to register" footgun.
- `runtime/bootstrap.py` is a 19-line idempotent loader. The `_builtins_registered`
  guard is module-level (correct for a single-process CLI; would need a class-level
  wrapper if multi-process spawn is used).

### 3.3 Architectural issues

#### Finding A — `TaskRegistry.register` factory resolution is fragile (MEDIUM)
`src/llm/training/task_registry.py:48-54`: when only `data_module_cls` is given, the
`resolved_factory` closure captures `cls` via default-arg trick — this works but is
non-obvious; future contributors frequently break it.

- **Impact**: subtle bugs when subclasses override behavior.
- **Solution**: convert to a small `_factory_from_cls(cls)` helper. (Complexity: trivial.)

#### Finding B — Two parallel registration paths for attention/MLP vs. norms (LOW)
`src/llm/core/registry.py:18-23` requires `ensure_norms_registered()` to be called
explicitly before use; `DecoderModel` correctly invokes it (`src/llm/models/decoder.py:14-17`).
But it must be called **again per `from_config`** because the import-time registration
of `ATTENTION_REGISTRY`/`MLP_REGISTRY` is done via decorator in `core/attn/mha.py`/`mlp.py`.
So the **norm registry must be lazily registered**, while the others are **eagerly registered**
— this asymmetry is undocumented and the only reason it works is the explicit
`ensure_norms_registered()` call. Future maintainers will add a new attention impl and
forget to import it.

- **Fix**: collapse both into a single `ensure_components_registered()` (called once at
  bootstrap). (Complexity: trivial; risk: low.)

#### Finding C — `_resolve_norm_type` returns a `type` (HIGH)
`src/llm/models/decoder.py:13-17` resolves the **class** rather than an instance. This
is fine for `nn.LayerNorm` and `RMSNorm`, but it will bite the next person who tries
to register `rms_norm_torch` (which requires shape inference at construction). The
model then does `isinstance(resolved_norm_type, type)` to branch
(`src/llm/models/decoder.py:116-119`) — a code smell that should be eliminated.

- **Solution**: change `NORM_REGISTRY` to store **factory functions**
  `(**kwargs) -> nn.Module`, mirroring `MODEL_REGISTRY` builders. (Complexity: 30 min;
  risk: medium — public contract change.)

#### Finding D — Paged Attention is partial and the contract is unclear (HIGH)
`docs/adr/004-paged-attention-serving.md` and `ROADMAP.md` both state that Paged
Attention is **partial**: prefix cache works, model forward still uses `list[KVCache]`.
The `ContinuousBatchingEngine` accepts both `self.kv_caches` and `self.paged_kv_cache`
(`src/llm/serving/batch_engine.py:124-151`) but only ever forwards the standard
`kv_caches` to the model. Users who enable `use_paged_attention=True` get **no benefit**
and may not realize it.

- **Solution**: either (i) raise `NotImplementedError` at `ContinuousBatchingEngine`
  construction when `use_paged_attention=True` and `from_serving_config()` is used
  (force the user to know), or (ii) document explicitly that it's sidecar-only.
  Option (i) is the honest fix. (Complexity: 5 min.)

#### Finding E — MLA with `use_cache=True` raises deep in `forward` (MEDIUM)
`AGENTS.md` notes "`attn_impl=mla` does not support KV cache" but the failure is
deferred until the attention layer tries to write into the cache — not at construction.
Better: validate `attn_impl` × `use_cache` compatibility in
`ModelConfig.check_consistency` (`src/llm/training/core/config.py:29-45`).

- **Fix**: add a validator. (Complexity: 5 min; benefit: fail-fast config errors.)

#### Finding F — Two engine modes (standard loop + custom loop) is a hidden contract (MEDIUM)
`src/llm/training/tasks/base_task.py:17-50` shows that `TrainingTask` is the only base,
but `PPOTask.run_training` overrides the loop entirely
(`src/llm/training/tasks/ppo_task.py`). The engine has two completely different code
paths in `run()` (`src/llm/training/core/engine.py:316-332` vs `334-402`). This is
documented (`src/llm/training/tasks/base_task.py:17-26`) but it's a real source of
bugs: standard-loop callbacks (`MetricsLogger`, `TensorBoardLogger`) do not fire inside
the PPO loop unless the custom task re-emits them.

- **Solution**: make custom-loop tasks **opt-in** to a unified callback bridge —
  `task.run_training(engine, callbacks=engine.callbacks)`. (Complexity: 2 hours;
  benefit: eliminates an entire class of silent-metric-loss bugs.)

### 3.4 Strong architectural decisions worth keeping

- **`src/` layout** — non-negotiable for testability; honored.
- **Pydantic config with env-var nesting** (`LLM_*`, `LLM_MODEL__HIDDEN_SIZE`) —
  clean, type-safe.
- **`CheckpointContributor` Protocol** — extensible without inheritance.
- **Single `Registry[T]`** backing 6 component registries — small surface area.
- **`load_entry_point_hooks` + `load_entry_point_registry`** — third-party plugins
  without forking.
- **ADRs in `docs/adr/`** — explicit architectural decisions, versioned.
- **Idempotent `ensure_*_registered()`** — correct pattern for setuptools entry points.

---

## 4. Code Organization

### Current state
- File sizes: largest source files ~400–500 LOC (`engine.py` 402, `api.py` 409,
  `decoder.py` 208). Healthy.
- 12.4k LOC source for a feature surface this wide is **tight** — some files could
  be split (`serving/batch_engine.py` at 350 LOC mixes slot allocator, prefix cache,
  continuous batching — borderline).
- Tests are equally split (10.6k LOC, 509 functions across 98 files) — a 0.85
  test:LOC ratio, which is healthy.

### Strengths
- One concept per file (e.g., `core/registry.py` is 23 LOC — exemplary).
- `tests/support/` helper layer (`tokenizers.py`, `models.py`, `data.py`) keeps test
  fixtures DRY.
- `conftest.py` hierarchy: root → `tests/core/attn/conftest.py` — good pytest discipline.

### Weaknesses

#### Finding G — `training/core/utils.py` is a 299-line junk drawer (MEDIUM)
Contains `PerformanceMonitor`, `Logger`, `DistributedManager`, `CheckpointManager`.
Each is conceptually independent and the file has no leading module docstring
explaining why they live together.

- **Fix**: split into `monitor.py`, `logger.py`, `distributed.py`, `checkpoint.py`.
  (Complexity: 1 hour; benefit: testability and discoverability.)

#### Finding H — `serving/api.py` mixes logger config + lifespan + endpoints + main() (MEDIUM)
At 409 LOC, it's the single biggest API surface. Hard-coded
`logger.setLevel(config.log_level)` happens **twice** (lines 39 and 42) — minor smell
but indicative of file scope creep.

- **Fix**: extract `observability.py` (logging + Prometheus), `_endpoints/` package
  (chat, generate, batch), keep `api.py` for `app` + lifespan. (Complexity: 3 hours.)

#### Finding I — Comment density is uneven (LOW)
`src/llm/core/kv_cache.py:120-127` and `src/llm/models/decoder.py:147-148` have
thorough comments. `src/llm/serving/api.py:178-182` and
`src/llm/training/core/engine.py:222-234` have explanatory comments that should be
docstrings on the method they live in. The repo would benefit from running `pydocstyle`
or `interrogate` in CI.

#### Finding J — `_learning/` directory exists (LOW)
Unclear what's in it; should either be promoted to `docs/learning/` or removed.

---

## 5. API Design

### 5.1 Public surface

```python
from llm import DecoderModel, SimpleCharacterTokenizer, generate, stream_generate, batch_generate
```

Plus CLI: `llm-train --task <name>`, `llm-serve`. Five primary CLI tasks
(regression, lm, stream_lm, sft, dpo, reward, ppo).

### Strengths
- One entry point per user intent (`llm-train`, `llm-serve`).
- Pydantic-typed configuration exposed via Typer — `llm-train --task lm --help` lists
  validated options.
- OpenAI-compatible `/v1/chat/completions` endpoint.
- Error envelopes follow standard FastAPI patterns (HTTPException with detail string).

### Weaknesses

#### Finding K — No structured error envelope (MEDIUM)
`src/llm/serving/api.py:171-182, 235-247, 314-326` — every error path returns
`{detail: str}`. There's no machine-readable error code, no request ID correlation,
no `errors[]` list. Compare to OpenAI's
`{"error": {"message": str, "type": str, "code": str}}`.

- **Solution**: introduce
  `class APIError(BaseModel): code: str; message: str; request_id: str`.
  (Complexity: 2 hours.)

#### Finding L — `_messages_to_prompt` uses a hard-coded chat template (MEDIUM)
`src/llm/serving/api.py:253-264`:
`"[System]: ...\n[User]: ...\n[Assistant]: ..."`. This produces one fixed template;
users cannot override. Real production systems need at least a `chat_template`
parameter (HF tokenizer's `apply_chat_template` is the obvious standard).

- **Fix**: take `chat_template: str | None` from `ServingConfig`; default to the
  current string. (Complexity: 30 min.)

#### Finding M — `_sync_generate`/`_sync_batch_generate` allocate a thread per request (HIGH)
`src/llm/serving/api.py:122-134, 156-164, 222-229` — `run_in_threadpool` offloads
every request to a worker thread, blocking the event loop for the duration of
`service.generate(prompt)`. With `max_concurrent_requests=4` and a 60-second timeout,
this spawns up to 4 blocking threads. The continuous batching engine already supports
concurrency, but the threadpool bypasses it.

- **Root cause**: `service.generate` is sync because the engine's `step()` is sync and
  `service.stream` is a sync generator. There is no proper async wrapper around the engine.
- **Fix**: either (i) make the engine async (proper fix, 1–2 weeks), or (ii) at minimum
  document the limitation and reduce `max_concurrent_requests` default to
  `2 * num_threads` so users don't deadlock. Option (i) is the long-term answer.
- **Severity**: HIGH because (a) it caps throughput, (b) it can deadlock under high
  concurrency, (c) it conflicts with FastAPI's concurrency model.

#### Finding N — API key check returns 403 even for missing config (LOW)
`src/llm/serving/api.py:67-79`: when `config.api_key` is None, the function returns
`None`, which `Security()` accepts as "no auth required". When the key is set but the
client didn't send it, you get 403. The semantics are correct but the **default of
no auth in a public-Facing-0.0.0.0 server** is dangerous — see `_extract_bearer_token`
and the FastAPI `host` default in `ServingConfig`.

- **Fix**: refuse to start when `host != "127.0.0.1"` and `api_key is None`.
  (Complexity: 5 min.)

#### Finding O — `host = "127.0.0.1"` is correct default but `main()` ignores it (LOW)
`src/llm/serving/api.py:405`:
`uvicorn.run("llm.serving.api:app", host=config.host, port=8000, reload=True)`. Good —
respects config. But `reload=True` is wrong for any non-toy deployment (it triggers
the watchdog importer which conflicts with `from llm.serving.api import app` in production).

---

## 6. Dependency Management

### Current state (`pyproject.toml`)
- Build backend: `uv_build>=0.11.0,<0.12.0` — modern PEP 517.
- Runtime deps: torch, transformers, fastapi, lm_eval, openai,
  prometheus-fastapi-instrumentator, python-json-logger, rouge-score, sacrebleu,
  scikit-learn, seaborn, tensorboard.
- Dependency groups: `dev`, `test`, `streaming`, `docs`.
- Entry points: `llm.models`, `llm.generation_backends`, `llm.data_sources`.
- Scripts: `llm-train`, `llm-serve`.
- Python `>=3.14` (bleeding edge — limits contributor pool).
- Lockfile: `uv.lock` (484KB).

### Strengths
- **Single source of truth** (`pyproject.toml`); no `requirements.txt`. AGENTS.md enforces this.
- Groups are well-defined: streaming for HF `datasets`, docs for mkdocs.
- Entry-point-based plugin system — third-party packages can register without forking.
- `uv lock --check` in CI — correct.

### Weaknesses

#### Finding P — Heavy dependencies pulled in for minor features (MEDIUM)
- `lm_eval` (≥70MB transitive) is in runtime deps but only used in `evaluation/harness/`.
- `scikit-learn`, `seaborn`, `rouge-score`, `sacrebleu` — useful for eval but should be
  optional (`evaluation` group).
- `tensorboard` — large dep; should be in `logging` group.

- **Solution**: introduce `[eval]`, `[logging]`, `[harness]` groups. Core install
  (`pip install llm`) should be < 200MB. (Complexity: 1 hour.)

#### Finding Q — `transformers>=5.10.2` — bleeding edge (HIGH)
`transformers` is a fast-moving target; pinning `>=5.10.2` with no upper bound is
risky. Combined with Python 3.14+, the **contributor pool is small**.

- **Fix**: cap with `<6.0.0` until you can test against new majors. (Complexity: trivial;
  benefit: CI stability.)

#### Finding R — `torch>=2.12.0` with no upper bound (MEDIUM)
Same risk as transformers.

- **Fix**: add `<3.0.0`.

#### Finding S — Streaming group split is wrong (LOW)
`datasets>=3.0.0` is in `streaming` group, but `HFStreamTextSource` is in core
(`src/llm/data/sources.py:60`). Result: import-time `from datasets import load_dataset`
(line 86) is wrapped in try/except — good — but the user discovers this only at runtime.

- **Fix**: lazy-import **only** the function body, surface a clearer error in
  `LLMError` with a remediation link. (Already mostly done — see `DataError` in
  `src/llm/exceptions.py:47-50`.)

---

## 7. Build System

### Current state
- `uv_build>=0.11.0,<0.12.0` backend.
- `Makefile` with `init`, `sync`, `dev`, `build`, `test*`, `ruff`, `ty`, `image`,
  `compose-up`, `clean`, `help`.

### Strengths
- `make help` auto-generates from comments — clever.
- Default target is `help`.
- Docker uses multi-stage.

### Weaknesses

#### Finding T — No `pyproject.toml` `[project.optional-dependencies]` for `dev` (LOW)
`uv` dependency groups aren't a PEP-standard, so users of plain `pip` get nothing. If
you want to support both `pip install llm[dev]` and `uv sync --group dev`, you need
both `optional-dependencies` and `dependency-groups`.

- **Fix**: add `optional-dependencies = {dev = [...], test = [...], streaming = [...]}`
  to `[project]`. (Complexity: 5 min.)

#### Finding U — Docker image missing for serving (MEDIUM)
`Dockerfile` exists, but verify whether it serves `llm-serve` or only runs tests.
`compose.yml` is 218 bytes — likely minimal.

- **Fix**: ensure `Dockerfile` produces a runtime image (not test deps) with
  `llm-serve` as entrypoint.

---

## 8. CI/CD

### Current state (`.github/workflows/`)
- `ci.yml` — lint + build-and-test matrix on ubuntu/macos/windows.
- `cd.yml`, `docker.yml`, `release.yml`, `docs.yml`, `labeler.yml`.

### Strengths
- **Multi-OS matrix** with `fail-fast: false` — correct.
- `uv lock --check` enforced.
- Coverage report uploaded as artifact.
- Renovate is configured (`.github/renovate.json`) — dependency updates automated.

### Weaknesses

#### Finding V — No GPU runner in CI (HIGH)
None of the jobs use a GPU. `e2e/` tests with markers `gpu`, `multi_gpu`,
`full_cluster` cannot run in this CI — meaning the most important correctness tests
(DDP, FSDP, real KV-cache batched inference) are **not in CI**.

- **Fix**: add a self-hosted GPU runner job (GitHub Actions has hosted GPU runners at
  premium tier), or document explicitly that GPU tests are local-only.
  (Complexity: integration work; benefit: confidence in the path users actually care about.)

#### Finding W — No type check in CI (MEDIUM)
`Makefile` has `make ty` (line 58) but `.github/workflows/ci.yml` does not run it.

- **Fix**: add `uvx ty check` as a step in `lint` job. (Complexity: 5 min.)

#### Finding X — No security scanning (MEDIUM)
No `bandit`, `pip-audit`, `safety`, `trivy`. For an LLM-serving project, this is
non-trivial.

- **Fix**: add `pip-audit` step on weekly cron + `bandit -r src/` on PR.
  (Complexity: 30 min.)

#### Finding Y — `release.yml` and `cd.yml` likely overlap (LOW)
Both exist; need to verify they don't conflict.

#### Finding Z — No test selection by marker in CI (LOW)
CI runs `make test-cov` (full suite) on all three OSes, even though many tests require
GPU and skip on CPU-only runners. This wastes CI minutes.

- **Fix**: gate GPU tests behind `needs: gpu` and have CPU-only run `make test-fast`.
  (Complexity: 30 min.)

---

## 9. Testing Strategy

### Current state
- **509 tests** across **98 files**.
- Markers: `quick`, `slow`, `heavy`, `e2e`, `integration`, `gpu`, `multi_gpu`,
  `full_cluster`, `need_gpu(n)`.
- Pytest config: `pythonpath = ["src", "."]` (correct for src-layout).
- Coverage: branch coverage enabled, `parallel = true`, with path mapping for `src/llm`.
- `tests/support/` — shared fixtures.
- `conftest.py` at root, `tests/models/`, `tests/data/`, `tests/core/attn/` (deeper).

### Strengths
- Excellent marker taxonomy.
- `pytest_collection_modifyitems` auto-skips GPU tests on CPU-only runners
  (`tests/conftest.py:23-38`).
- Stub tokenizer, line tokenizer, support corpus, decoder kwargs helper — real
  attention to fixture ergonomics.
- Most unit tests are CPU-runnable.

### Weaknesses

#### Finding AA — No coverage gate in CI (MEDIUM)
`ci.yml` runs `make test-cov` but does not fail if coverage drops. A target of 80%+
is implicit; not enforced.

- **Fix**: add `coverage report --fail-under=80` (or current value) to the CI step.
  (Complexity: 2 min.)

#### Finding AB — Tests touch GPU-only code paths without GPU (MEDIUM)
Some tests in `tests/core/test_paged_attention.py` and
`tests/serving/test_paged_integration.py` reference `.cuda()` calls. Without a GPU,
they're skipped silently — but skipping is correct. The risk is that **developers see
green locally and miss broken CUDA paths**.

- **Fix**: gate such tests on `need_gpu(1)` marker; document the policy in
  `tests/README.md`.

#### Finding AC — `tests/conftest.py` seeds `manual_seed(42)` autouse for every test (LOW)
This is conservative and good. But it makes tests using `torch.randn` order-dependent
across runs if anyone disables the fixture. Acceptable.

#### Finding AD — Missing: property-based tests, fuzz tests, mutation tests (MEDIUM)
- For an inference engine, **invariant tests** are critical: e.g., "greedy decoding is
  deterministic", "KV cache hit on prefix produces identical output", "batch=1 matches
  single-request output".
- `hypothesis` is not in dev deps.
- `mutmut` / `cosmic-ray` would be nice-to-have but expensive.

- **Fix**: add `hypothesis` to test deps and seed 3–5 invariant tests
  (`tests/inference/test_invariants.py`). (Complexity: 1 day.)

#### Finding AE — Missing: regression tests for known bugs (LOW)
Without a `tests/regression/` directory or numbered test naming convention, future
regressions will not be co-located with their fix. Consider
`tests/regression/test_<issue#>_<slug>.py`.

---

## 10. Documentation

### Current state
- `README.md` (175 lines) — feature list, quick start, install, usage pointer to docs.
- `AGENTS.md` — decision hub for AI agents (Chinese; bilingual project).
- `ROADMAP.md` (440 lines) — extremely thorough.
- `CHANGELOG.md` — keep-a-changelog format.
- `docs/reference/architecture.md` (355 lines) — mermaid diagrams, registry pattern,
  training pipeline.
- `docs/development/` (8 files): components, deep-dive-ddp, flow-callbacks,
  guide-configuration, guide-extending, training-flow, troubleshooting.
- `docs/guides/`: checkpoints, distributed, finetuning, inference.
- `docs/tutorials/`: 01-pretraining, 02-finetuning, 03-inference.
- `docs/adr/`: 5 ADRs (GQA, SwiGLU, KV cache, Paged Attention, QLoRA) + template + README.

### Strengths
- **ADRs** — rare in a project this size; gold for maintainers.
- Mermaid diagrams in `architecture.md` — excellent.
- `getting-started.md`, `faq.md`, `troubleshooting.md` — onboarding.
- ROADMAP is unusually thorough for a small project.

### Weaknesses

#### Finding AF — `AGENTS.md` is bilingual and structured around AI agents (LOW)
Helpful for the maintainer's own Claude workflow; less helpful for human contributors.
Consider separating AI-agent guidance into a hidden/internal file.

#### Finding AG — `docs/superpowers/` — unclear (LOW)
Should be `docs/_superpowers/` (underscore prefix) or merged into `AGENTS.md`.

#### Finding AH — `CHANGELOG.md` is comprehensive but not linked to GitHub releases (LOW)
Each entry doesn't reference PR/issue numbers — making historical context hard to recover.

#### Finding AI — `mkdocs.yml` exists but no evidence of docs workflow (MEDIUM)
`docs.yml` workflow exists. But the README links use raw GitHub URLs, not
mkdocs-rendered URLs — defeating the purpose of having a docs site.

- **Fix**: deploy to GitHub Pages from `docs.yml`; update README to link to
  `https://pplmx.github.io/llm/`. (Complexity: 2 hours.)

#### Finding AJ — API reference is missing (HIGH)
Users land on the repo and see no auto-generated API reference. For a framework with
~30 public classes, this is the **#1 documentation gap**.

- **Fix**: enable `mkdocstrings` for `llm.*` and have it build on every push.
  (Complexity: 4 hours.)

#### Finding AK — `docs/development/components.md` duplicates `docs/reference/architecture.md` (LOW)
The Chinese version (`components.md`) and English version (`architecture.md`) overlap
heavily. Pick one to be canonical, link the other as a translation.

---

## 11. Performance

### Current state
- `torch.compile` integration via `mode="reduce-overhead"` in
  `src/llm/training/core/engine.py:70`.
- AMP auto-detects bf16 vs fp16 (`src/llm/training/core/engine.py:101-117`).
- Gradient checkpointing (`src/llm/models/decoder.py:147-171`).
- KV cache with pre-allocated buffers (`src/llm/core/kv_cache.py:52-53`).
- Continuous batching with slot allocator (`src/llm/serving/batch_engine.py:43-66`).
- Prefix cache (`src/llm/serving/batch_engine.py:16-41`).
- Paged Attention sidecar (`src/llm/core/paged_attention/`).

### Strengths
- Solid set of standard optimizations present.
- `torch.set_float32_matmul_precision("high")` set globally in train
  (`src/llm/training/train.py:110`).
- `cudnn.benchmark = True` (`src/llm/training/train.py:109`).

### Weaknesses

#### Finding AL — `torch.compile(mode="reduce-overhead")` in production is wrong (MEDIUM)
`reduce-overhead` mode uses CUDA graphs, which conflict with `use_cache=True` /
variable-length sequences. Worse, the code silently falls back on any error
(`src/llm/training/core/engine.py:71-72`).

- **Fix**: switch to `mode="default"` for variable-length; document.
  (Complexity: 5 min.)

#### Finding AM — `ContinuousBatchingEngine._copy_kv_between_slots` is naive (MEDIUM)
`src/llm/serving/batch_engine.py:169-172`: per-step KV copy on prefix cache hit,
fully serialized in Python. For sequences with 8k+ context, this serializes
8k × head_dim × num_kv_heads × num_layers × 2 (K+V) floats per copy.

- **Fix**: do the copy **inside** the model forward as a fused gather. Or: pre-allocate
  slots per request and use the existing `update_at_indices` machinery (already in
  `src/llm/core/kv_cache.py:105-166`). (Complexity: 1 day.)

#### Finding AN — No throughput/latency benchmarks in CI (HIGH)
`scripts/benchmark_inference.py` exists but isn't in CI. Without benchmarks,
refactoring the serving layer is dangerous.

- **Fix**: add a `make bench` target + GitHub Action that runs the benchmark and posts
  results as a PR comment. (Complexity: 1 day.)

#### Finding AO — No Flash Attention 2 integration (MEDIUM — explicitly in roadmap)
ROADMAP §Unreleased lists this. SDPA already uses
`torch.nn.functional.scaled_dot_product_attention` (in `src/llm/core/attn/mha.py`) so
PyTorch's SDPA backend should auto-select Flash 2 on Hopper/Ampere. But **explicit
Flash-Attn integration** would unlock sliding-window patterns and longer sequences.

- **Fix**: optional `attn_impl="flash_attn"` registration via the ATTENTION_REGISTRY.
  (Complexity: 2 days.)

#### Finding AP — AMP scaler used unconditionally for bf16 (`src/llm/training/core/engine.py:115-118`)
`use_scaler = self.config.optimization.use_amp and self.device.type == "cuda" and self.resolved_amp_dtype == "float16"`
— correct. But `GradScaler(enabled=use_scaler)` will still call `.scale(loss)` and
`.step(optimizer)` paths, which on bf16 with `enabled=False` are no-ops. This is fine,
but the verbose branching is hard to read.

- **Fix**: extract `AmpScaler` wrapper class. (Complexity: 1 hour; benefit: clarity.)

---

## 12. Security

### Current state
- `SECURITY.md` — placeholder (versions table says "5.1.x" which is wrong; project is 0.0.5).
- API key check via header (`X-API-Key` or `Bearer`).
- `huggingface_hub.snapshot_download` in `_load_from_hub` — does NOT pass `token=` —
  relies on env vars.
- No CSRF, no rate limiting, no audit logging.

### Critical findings

#### Finding AQ — `SECURITY.md` is fictional (HIGH)
Lines 12–15 list versions "5.1.x", "5.0.x", "4.0.x" — **none of these match the
project**. This is a copy-paste artifact.

- **Fix**: rewrite with actual version 0.0.5 + report-vulnerability instructions.
  (Complexity: 10 min.)

#### Finding AR — `from_pretrained` accepts any `model_path` without validation (HIGH)
`src/llm/compat/hf_loader.py:48-54`: passes user-controlled path to `_load_from_local`
or `_load_from_hub`. The HF Hub path downloads to a temp directory with
`allow_patterns=["*.json", "*.safetensors", "*.bin"]` — but `.bin` files are untrusted
pickle. `torch.load(weights_only=True)` is used (`src/llm/compat/hf_loader.py:200`) —
good — but other projects that consume these `.bin` files may not.

- **Fix**: drop `*.bin` from `allow_patterns`; safetensors is sufficient. (Complexity:
  5 min; benefit: prevents arbitrary code execution via pickle.)

#### Finding AS — API key compared with `==` (HIGH — timing attack)
`src/llm/serving/api.py:71, 76`: `if api_key_header == config.api_key`. Use
`hmac.compare_digest`.

- **Fix**: replace with `hmac.compare_digest`. (Complexity: 1 min.)

#### Finding AT — `ContinuousBatchingEngine.step` is not thread-safe but lives in async context (HIGH)
Multiple FastAPI workers calling `service.generate` concurrently will hit
`_copy_kv_between_slots` and KV cache mutations without locking. PyTorch CUDA
operations may also serialize, but Python-level mutations to `self._seq_len`,
`self.free_slots`, etc. are racy.

- **Fix**: wrap step in a `threading.Lock`. (Complexity: 30 min.)

#### Finding AU — No structured logging of auth failures (MEDIUM)
`src/llm/serving/api.py:79` raises HTTPException directly; the failed request is
invisible to operators.

- **Fix**: log on failure with masked key prefix (e.g., `key[:4] + "***"`).
  (Complexity: 10 min.)

#### Finding AV — HF `trust_remote_code=False` is good — but no explanation (LOW)
`src/llm/compat/hf_loader.py:30`: defaults to `False`. Correct, but should be
highlighted in docs.

#### Finding AW — No dependency audit (MEDIUM)
See Finding X.

---

## 13. Observability

### Current state
- Prometheus via `prometheus_fastapi_instrumentator` (`src/llm/serving/api.py:110`).
- `python-json-logger` for structured logs.
- `tensorboard` for training.
- `PerformanceMonitor` in `src/llm/training/core/utils.py`.

### Strengths
- HTTP metrics are out-of-the-box.
- Structured logs are JSON.

### Weaknesses

#### Finding AX — No custom Prometheus metrics for inference (MEDIUM)
The default instrumentator gives request count/latency. But not:
- Active generation requests
- Tokens/sec (prefill + decode split)
- KV cache utilization
- Slot allocator pressure

- **Fix**: add a `serving/metrics.py` module exposing `Counter`/`Gauge`/`Histogram`
  for these. (Complexity: 1 day.)

#### Finding AY — No OpenTelemetry tracing (MEDIUM)
For a multi-stage pipeline (decode → KV cache update → sampling → response), tracing
would help debug latency spikes. Not critical for v0.1, but plan for it.

#### Finding AZ — No log of model version or config on startup (HIGH)
Operators cannot tell what model is serving. `lifespan` should log: model class,
param count, dtype, device, max_seq_len, paged_attention_enabled, prefix_cache_enabled.

- **Fix**: log on startup. (Complexity: 10 min.)

---

## 14. Release Process

### Current state
- `release.yml` — exists; not fully reviewed.
- `cliff.toml` — `git-cliff` for changelog generation. Good.
- PyPI version 0.0.5; tag/release process not visible.

### Strengths
- `git-cliff` ensures CHANGELOG is auto-generated from conventional commits.

### Weaknesses

#### Finding BA — No release notes template (LOW)
No `RELEASE_TEMPLATE.md` for GitHub release notes.

#### Finding BB — Versioning policy unclear (MEDIUM)
"0.0.5" with classifiers `Development Status :: 4 - Beta` — but is each 0.0.x a
feature or a fix? The CHANGELOG mixes them.

- **Fix**: align with semver: 0.x.y where x = minor features, y = fixes.

---

## 15. Community Health

### Current state
- Single maintainer: Mystvio.
- No `CODEOWNERS` file.
- Issue templates: bug_report, feature_request.
- PR template exists.

### Critical findings

#### Finding BC — Bus factor = 1 (HIGH — by far the largest project risk)
A single human maintainer is the #1 existential risk. If Mystvio is unavailable for
2 weeks, the project stalls.

- **Mitigation**:
  1. Document **decision-making boundaries** in `AGENTS.md` so contributors know what
     they can decide without asking.
  2. Add `CODEOWNERS` so at least 2 humans can review.
  3. Move **opinionated choices** into ADRs so they survive a maintainer transition.
  4. Promote at least one reviewer from the community within 6 months.

#### Finding BD — `CODE_OF_CONDUCT.md` is 135 bytes (LOW)
Probably the Contributor Covenant. Verify it's actually the Covenant or rewrite.

#### Finding BE — No contributor ladder (LOW)
New contributors don't know how to become maintainers. Add
`docs/maintainers/ladder.md` or similar.

#### Finding BF — Issue templates are minimal (LOW)
`bug_report.md` and `feature_request.md` exist; verify they request **repro steps**,
**versions**, **environment**.

#### Finding BG — No Discord / Slack / Discourse (LOW)
Acceptable for a small project; document the GitHub Discussions workflow if used.

---

## 16. Extensibility

### Current state — well-designed

The registry + entry-point system genuinely works. To add:
- New attention: `@register_attention("name") class Foo(...): ...`
- New MLP: `@register_mlp("name") class Bar(...): ...`
- New model: add entry point in your own package's `pyproject.toml` → `llm.models`
- New data source: same, `llm.data_sources` group
- New task: `TASK_REGISTRY.register("name", TaskClass, DataModuleClass)` (or via entry point)
- New metric: subclass `BaseMetric`

This is **best-in-class** for a framework of this size.

### Weakness

#### Finding BH — No `EXPORT_REGISTRY` symmetry (LOW)
`BACKEND_REGISTRY`, `SOURCE_REGISTRY`, `MODEL_REGISTRY` exist, but `export/` doesn't
have one. `pyproject.toml` only lists 3 entry-point groups.

- **Fix**: add `llm.export_backends` group when introducing torch.compile export /
  vLLM / TensorRT-LLM targets. (Complexity: trivial.)

---

## 17. Roadmap (Engineering ROI)

### Tier 1 — Immediate improvements (1–2 weeks)

| # | Item | Effort | Impact |
|---|---|---|---|
| 1 | Fix `SECURITY.md` (Finding AQ) | 10 min | Trust |
| 2 | `hmac.compare_digest` for API key (Finding AS) | 1 min | Security |
| 3 | Drop `*.bin` from HF allow_patterns (Finding AR) | 5 min | Security |
| 4 | Raise on `use_paged_attention=True` until full path implemented (Finding D) | 5 min | UX |
| 5 | Add `attn_impl` × `use_cache` validator (Finding E) | 5 min | UX |
| 6 | Remove `reload=True` from `main()` (Finding O) | 1 min | Production safety |
| 7 | Refuse to start if `host != 127.0.0.1` and `api_key is None` (Finding N) | 5 min | Security |
| 8 | Add `make ty` to CI lint job (Finding W) | 5 min | Quality |
| 9 | Add `pip-audit` + `bandit` to CI (Finding X) | 30 min | Security |
| 10 | Cap `transformers<6` and `torch<3` (Findings Q, R) | 5 min | Stability |
| 11 | Add `coverage --fail-under=80` to CI (Finding AA) | 2 min | Quality |
| 12 | Log model version + config on startup (Finding AZ) | 10 min | Operability |
| 13 | `eval`, `logging` dependency groups (Finding P) | 1 hour | Install size |
| 14 | `threading.Lock` around `ContinuousBatchingEngine.step` (Finding AT) | 30 min | Correctness |

### Tier 2 — Short-term evolution (1–2 months)

| # | Item | Effort | Impact |
|---|---|---|---|
| 1 | Make `ContinuousBatchingEngine.step` truly async (Finding M) | 1–2 weeks | Throughput |
| 2 | Structured `APIError` envelope + request IDs (Finding K) | 2 days | API quality |
| 3 | `chat_template` parameter from `ServingConfig` (Finding L) | 30 min | Compatibility |
| 4 | Collapse NORM_REGISTRY to factory functions (Finding C) | 1 hour | Maintainability |
| 5 | Split `training/core/utils.py` into 4 files (Finding G) | 1 hour | Maintainability |
| 6 | Split `serving/api.py` (Finding H) | 3 hours | Maintainability |
| 7 | Custom-loop task callback bridge (Finding F) | 2 hours | RLHF correctness |
| 8 | Hypothesis invariant tests (Finding AD) | 1 day | Correctness |
| 9 | `mkdocstrings` API reference (Finding AJ) | 4 hours | Onboarding |
| 10 | Deploy docs to GitHub Pages (Finding AI) | 2 hours | Discoverability |
| 11 | Make `torch.compile` mode configurable (Finding AL) | 1 hour | Flexibility |
| 12 | Custom Prometheus metrics (Finding AX) | 1 day | Operability |
| 13 | Optimize `KVCache.update_at_indices` for non-1 seq_len (Finding AM) | 1 day | Throughput |
| 14 | Add `CODEOWNERS`, contributor ladder, decision-making boundaries (Findings BC, BE) | 2 hours | Bus factor |

### Tier 3 — Mid-term architecture evolution (3–6 months)

| # | Item | Effort | Impact |
|---|---|---|---|
| 1 | Self-hosted GPU runner for `e2e`/`gpu` tests (Finding V) | 2 weeks | CI confidence |
| 2 | FSDP end-to-end + docs | 2 weeks | Scaling |
| 3 | Paged Attention **full forward path** (replace `list[KVCache]`) | 4 weeks | Memory |
| 4 | Flash Attention 2 integration via registry (Finding AO) | 2 weeks | Throughput |
| 5 | MLA + KV cache (full implementation) | 4 weeks | Architecture parity |
| 6 | Eval pipeline integration with `lm_eval` (it's a dep, but not deeply wired) | 2 weeks | Eval rigor |
| 7 | HF Hub publish pipeline | 1 week | Distribution |
| 8 | Eval preset files (C4, Pile, RedPajama — per ROADMAP P0) | 2 weeks | UX |
| 9 | Speculative decoding in continuous batching | 3 weeks | Latency |

### Tier 4 — Long-term vision (6–24 months)

| # | Item | Effort | Impact |
|---|---|---|---|
| 1 | Multimodal registry spike (Vision/Speech/...) | 3 months | New market |
| 2 | Pipeline Parallelism (per ROADMAP P3) | 6 months | Multi-node |
| 3 | DeepSpeed ZeRO-3 integration | 3 months | Memory |
| 4 | RLHF at scale (full PPO + GRPO + DPO + online DPO) | 6 months | Production RLHF |
| 5 | Hugging Face TRL interop | 1 month | Ecosystem |
| 6 | vLLM-style PagedAttention with prefix cache | 4 months | Throughput |
| 7 | Web-based training dashboard | 6 months | UX |
| 8 | 3rd-party plugin discovery + showcase | 3 months | Community |

---

## 18. Comparison with Similar Projects

| Capability | `llm` | `nanoGPT` | `LitGPT` | `transformers` | `trl` |
|---|---|---|---|---|---|
| Decoder architecture | ✅ full | minimal | ✅ full | ✅ | n/a |
| Registry plugin kernel | ✅ | ❌ | ❌ | partial | ❌ |
| Continuous batching | ✅ | ❌ | ❌ | via `transformers` | ❌ |
| OpenAI-compatible serve | ✅ | ❌ | ❌ | ✅ | ❌ |
| RLHF (PPO/DPO) | ✅ | ❌ | ❌ | n/a | ✅ |
| HF Hub publish | ❌ | ❌ | ❌ | ✅ | ❌ |
| Test count | 509 | <50 | >1000 | 1000+ | 1000+ |
| ADRs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Single maintainer | ✅ | ✅ | ❌ (Lightning) | ❌ | ❌ |

**Unique strengths of `llm`**: registry kernel + ADRs + bilingual docs + entry-point
plugin system + OpenAI-compat in core.
**Notable gaps vs. peers**: no Hub publish, no vLLM-style throughput, no TRL interop,
no Lightning Fabric integration.

---

## 19. Critical Thinking & Hidden Risks

1. **The single maintainer is the #1 risk** (Finding BC). Everything else is
   recoverable; bus factor is not.
2. **`ContinuousBatchingEngine` is a 350-LOC untested-at-scale inference kernel** —
   it's the most likely source of production incidents. Gate it behind heavy e2e
   tests before declaring it production-ready.
3. **The `_learning/` directory is mysterious** — investigate before it leaks into
   a release.
4. **`examples/kv_cache_demo.py` and `examples/openai_client_demo.py`** may have
   drifted from the actual API; verify they run on a fresh clone.
5. **ROADMAP mentions RLHF (PPO/DPO)** — but `PPOTask.run_training` overrides the
   entire loop. If this loop is wrong, no callback-based observability catches it.
   Add `tests/training/rlhf/test_ppo_invariants.py` with synthetic preference data.
6. **The `compat/` layer is shrinking** (`compat/hf_loader.py` still has code;
   `compat/weight_mapping.py` exists) but its boundary with `runtime/` is fuzzy.
   Long-term, decide: keep `compat/` for one-version-back compat, or fold into
   `runtime/`.

---

## 20. Verdict

`llm` is **architecturally well-built for its size**, with a registry kernel, ADRs,
type-safe config, and disciplined layering that punches above its weight. It is,
however, **operationally immature for anything more than experiments** — security,
observability, throughput, and contributor scalability all need attention.

The single best ROI move is to **land Tier 1 (1–2 days of work) and ship 0.0.6**,
then use Tier 2 as the basis for the next minor version with one well-scoped
maintainer recruit. The Tier 3/4 vision is achievable but only if the maintainer
pipeline is fixed first.

**Recommended grade on becoming long-term maintainer**: **B** — the code is
high-quality and self-documenting; the operations and community surface need
investment to make the project survive the maintainer transition.

---

## Appendix A — Severity Summary

| ID | Severity | Title |
|---|---|---|
| AQ | HIGH | `SECURITY.md` is fictional |
| AR | HIGH | `from_pretrained` allows unsafe `.bin` files |
| AS | HIGH | API key compared with `==` (timing attack) |
| AT | HIGH | `ContinuousBatchingEngine.step` is not thread-safe |
| AN | HIGH | No throughput/latency benchmarks in CI |
| M | HIGH | Sync `_sync_generate` blocks event loop |
| V | HIGH | No GPU tests in CI |
| D | HIGH | Paged Attention sidecar with no benefit + no warning |
| C | HIGH | `_resolve_norm_type` returns a class, will break custom norms |
| BC | HIGH | Bus factor = 1 |
| AJ | HIGH | No API reference docs |
| Q | HIGH | `transformers>=5.10.2` no upper bound |
| F | MEDIUM | Custom-loop task callback bridge missing |
| X | MEDIUM | No security scanning in CI |
| P | MEDIUM | Heavy runtime deps |
| AD | MEDIUM | No invariant/regression tests |
| AM | MEDIUM | Naive `_copy_kv_between_slots` |
| AO | MEDIUM | No explicit Flash Attention 2 |
| AZ | MEDIUM | No startup config logging |
| W | MEDIUM | `ty` not in CI |
| AA | MEDIUM | No coverage gate |
| R | MEDIUM | `torch>=2.12.0` no upper bound |
| AL | MEDIUM | `torch.compile` mode hardcoded |
| K | MEDIUM | No structured error envelope |
| L | MEDIUM | Hard-coded chat template |
| N | MEDIUM | No-auth 0.0.0.0 server allowed |
| H | MEDIUM | `serving/api.py` is a junk drawer |
| G | MEDIUM | `training/core/utils.py` is a junk drawer |
| E | MEDIUM | MLA + use_cache failure deferred to forward |
| AU | MEDIUM | No structured logging of auth failures |
| AX | MEDIUM | No custom Prometheus metrics |
| AI | MEDIUM | `mkdocs.yml` exists but unused |
| AB | MEDIUM | GPU-only tests skipped silently |
| AY | MEDIUM | No OpenTelemetry tracing |
| BB | MEDIUM | Versioning policy unclear |
| U | MEDIUM | Docker image not verified for runtime |
| O | LOW | `reload=True` in `main()` |
| J | LOW | `_learning/` directory mystery |
| I | LOW | Comment density uneven |
| B | LOW | Two parallel registration paths |
| AH | LOW | CHANGELOG not linked to PRs |
| AG | LOW | `docs/superpowers/` unclear |
| AF | LOW | Bilingual docs split |
| AK | LOW | `components.md` / `architecture.md` overlap |
| T | LOW | `optional-dependencies` missing |
| BH | LOW | No `EXPORT_REGISTRY` |
| S | LOW | Streaming group split wrong |
| AV | LOW | `trust_remote_code=False` not highlighted |
| BE | LOW | No contributor ladder |
| BA | LOW | No release notes template |
| BD | LOW | `CODE_OF_CONDUCT.md` minimal |
| BF | LOW | Issue templates minimal |
| BG | LOW | No community channel |
| Z | LOW | CI doesn't gate by marker |
| Y | LOW | `release.yml` / `cd.yml` overlap suspected |
| A | LOW | `TaskRegistry.register` factory closure fragile |

---

## Appendix B — How to use this document

- **For each Tier 1 item**, open a GitHub issue using the audit-template below.
- **For each Tier 2/3 item**, cross-reference with `ROADMAP.md` P0–P3 sections.
- **For architectural decisions** (Tier 3/4 items affecting registry contract),
  write a new ADR in `docs/adr/`.

### Issue template

```markdown
## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding <ID>

## Description
<2–3 sentences from the review>

## Acceptance criteria
- [ ] <observable, testable outcome>
- [ ] <regression test or CI check>

## Estimate
<T-shirt size or hours>

## Labels
audit-2026-07, v0.0.6-audit-followup
```
