# `llm.evaluation` — Metrics, Tasks, and Harness Adapters

The evaluation subpackage is split into two slices:

- **Metrics + offline tasks** (`llm.evaluation.metrics`, `llm.evaluation.eval_tasks`)
  — pure-Python accuracy/F1/perplexity helpers and the offline task
  protocol. See [`metrics.base`](../reference/architecture.md) for the
  task/metric contract.
- **lm-evaluation-harness adapter** (`llm.evaluation.harness`) — the
  thin shim that lets our `DecoderModel` plug into the upstream
  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
  benchmark suite.

This page documents the harness slice. `lm_eval` is an **optional**
dependency; importing the modules below never crashes on a host that
doesn't have it installed — only instantiation raises.

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
