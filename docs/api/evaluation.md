# `llm.evaluation` — Metrics, Tasks, and Harness Adapters

The evaluation subpackage is split into two slices:

- **Metrics + offline tasks** (`llm.evaluation.metrics`, `llm.evaluation.eval_tasks`)
  — pure-Python accuracy/F1/perplexity helpers and the offline task
  protocol. See below for the full API reference.
- **lm-evaluation-harness adapter** (`llm.evaluation.harness`) — the
  thin shim that lets our `DecoderModel` plug into the upstream
  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
  benchmark suite.

The metrics and task bases have no optional dependencies; the harness
slice is gated behind the ``lm_eval`` **optional** dependency —
importing the harness modules below never crashes on a host that doesn't
have it installed, only instantiation raises.

## Metrics

Abstract base class and concrete scoring implementations for evaluation.

::: llm.evaluation.metrics.base
::: llm.evaluation.metrics.accuracy
::: llm.evaluation.metrics.generation
::: llm.evaluation.metrics.perplexity

## Evaluation Tasks

Abstract base class and concrete task implementations for offline evaluation.

::: llm.evaluation.eval_tasks.base
::: llm.evaluation.eval_tasks.lm_task

## Benchmark Presets

`EvalPreset` bundles a benchmark name with the kwargs that
`lm_eval.evaluator.evaluate` understands. Three built-in presets ship
out of the box; users can construct their own by passing the same
fields.

::: llm.evaluation.harness.presets

## `LlamaLmEvalLM` — `DecoderModel` adapter for lm_eval

Minimal `lm_eval.api.model.LM` implementation that wraps a
`DecoderModel` + tokenizer. Implements the three protocol methods
(`loglikelihood`, `loglikelihood_rolling`, `generate_until`) without
pulling in `HFLM`'s HF-only kwargs (`prefix_token`, `backend`).

::: llm.evaluation.harness.lm_eval_lm

## `LmEvalAdapter` — top-level driver

Preset lookup, kwarg merging, and structured result flattening on top
of `lm_eval.evaluator`.

::: llm.evaluation.harness.adapter

## End-to-end usage

See the [Evaluation guide](../guides/evaluation.md) for a worked
example (preset selection, result flattening, soft-dependency
contract).
