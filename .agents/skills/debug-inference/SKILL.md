---
name: debug-inference
description: >
  Systematic debugging workflow for inference/serving issues in this repo
  (Python LLM training framework): generation backends, KV cache, paged
  attention, continuous batching, OpenAI-compatible API. Use when the user
  reports incorrect model output, slow generation, request hangs, KV-cache
  errors, serving 500s, or asks to "debug / diagnose / trace" an inference
  problem (推理调试 / 生成结果不对 / 服务报错).
---

# Debug Inference Issues

Follow the loop: **reproduce → isolate to a layer → minimal repro → fix →
regression test**. Never fix on a hypothesis alone; produce a failing test
first when possible.

## 1. Reproduce at the lowest sufficient layer

Work bottom-up; each layer strips one variable:

| Layer              | How to run                                                                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sampling math only | unit-run helpers in `src/llm/generation/sampling.py` (`apply_frequency_penalty`, `apply_presence_penalty`, `apply_logit_bias`, top-k/top-p)                                 |
| Backend, no HTTP   | instantiate a backend via `create_backend(name, engine=...)` (`src/llm/generation/registry.py`: `eager` / `batched` / `speculative`) and call `generate(...)` from a script |
| Model + KV cache   | `DecoderModel.forward(..., use_cache=True, kv_cache=...)` or `PagedKVCache` directly (see `tests/models/`, `tests/core/`)                                                   |
| Full serving       | `uv run llm-serve` (FastAPI, OpenAI-compatible); exercise via `curl` / `tests/serving/` clients                                                                             |

Logging: Python `logging` (the server logs model version + config on startup,
audit ticket #12). Raise level before starting:

```bash
uv run python -X dev -c "import logging; logging.basicConfig(level=logging.DEBUG); ..."
# or for serving:
PYTHONUNBUFFERED=1 uv run llm-serve  # check startup config log lines first
```

## 2. Symptom → component map

| Symptom                                      | First suspects                                                                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Wrong/degenerate tokens                      | `generation/sampling.py` (temperature/top-k/top-p ordering), penalty & logit_bias helpers, next-token shift off-by-one               |
| Output differs between backends              | backend-specific sampling path vs eager reference; speculative decode acceptance logic (`generation/speculative.py`)                 |
| Garbage after long context / position errors | KV cache indexing (`core/kv_cache.py`, `update_at_indices`), `start_pos` accounting, RoPE `max_seq_len`                              |
| `use_cache` / attn_impl validation error     | `ATTENTION_KV_CACHE_CAPABILITY` in `core/registry.py` + `ModelConfig.check_consistency` (ticket #05)                                 |
| Paged-attention wrong results                | ADR-004 path: `PagedKVCache` block tables, prefix caching; see ticket #30                                                            |
| Serving hang / stuck request                 | `serving/batch_engine.py` continuous-batching loop + its threading lock (ticket #14); check for un-awaited futures / missing wakeups |
| HTTP 4xx/5xx shape issues                    | structured API error envelope (ticket #15), request schema validators in `serving/` routers                                          |
| Quantized model misbehaves                   | `llm-quantize` output + `quantization/` loaders; GPTQ packing scale fix history (commit d20fe87), checkpoint key formats (494ee10)   |

## 3. Isolate

- Bisect between a known-good commit and HEAD (`git log --oneline --
  src/llm/generation src/llm/serving src/llm/core`) — inference regressions
  are usually recent and localized.
- Pin deterministic inputs: fixed prompt token ids, `temperature=0` / greedy
  to remove sampling noise; compare logits, not strings, when outputs drift.
- For cache bugs, compare step-by-step incremental decode against a single
  full-sequence forward (the reference equality many tests use).

## 4. Fix + regression test

- Fix the root cause, not the symptom; keep the fix inside the owning layer
  (see AGENTS.md 源码分层 — do not patch `DecoderModel` for serving concerns).
- Add a regression test that fails without the fix (behavior, not
  implementation details — see `write-test` skill).
- Verify: targeted dir first (`uv run pytest tests/generation tests/serving
  -q`), then the full loop (`run-ci` skill).
- Record findings in the RIL graph (evidence nodes cite test names /
  file:line) via the `graph-engineering` skill.
