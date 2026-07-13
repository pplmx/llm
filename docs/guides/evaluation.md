# Evaluation Guide

This guide covers how to run benchmarks on a trained model using the
`lm-evaluation-harness` integration that ships with this project.

For everything related to *training-time* metrics (perplexity tracked
during a training run, callback-based logging, etc.) see the
[training flow guide](../development/training-flow.md). This page is
about **standalone evaluation on standard benchmarks** (MMLU, ARC,
WikiText, …).

## Installation

The lm-eval integration is an optional dependency — it is **not**
required for the core training and serving install. Pull it in with:

```bash
# uv
uv sync --group eval
# or pip
pip install 'llm[eval]'
```

The modules involved (`llm.evaluation.harness.*`) all import cleanly
without `lm_eval` installed; the `ImportError` fires only when you
actually try to construct an `LlamaLmEvalLM` or `LmEvalAdapter`, and
the message tells you exactly how to fix it.

## The three building blocks

```
┌────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│  EvalPreset        │   │  LlamaLmEvalLM       │   │  LmEvalAdapter     │
│  (presets.py)      │   │  (lm_eval_lm.py)     │   │  (adapter.py)      │
│                    │   │                      │   │                    │
│  task + kwargs +   │   │  wraps DecoderModel  │   │  task manager +    │
│  batch_size + …    │   │  to lm_eval's LM     │   │  preset lookup +   │
│                    │   │  protocol            │   │  result flattening │
└─────────┬──────────┘   └──────────┬───────────┘   └─────────┬──────────┘
          │                         │                         │
          └─────────────────────────┴─────────────────────────┘
                                    │
                          run_preset(preset, model)
                                    │
                                    ▼
                       lm_eval.evaluator.evaluate(...)
```

### 1. `EvalPreset` — *what* to run

A preset bundles a benchmark name with the kwargs that
`lm_eval.evaluator.evaluate` understands. Three ship out of the box
(defined in `src/llm/evaluation/harness/presets.py`):

| Preset          | Task         | Few-shot | Default batch | Purpose                         |
|-----------------|--------------|---------:|--------------:|---------------------------------|
| `MMLU_PRESET`   | `mmlu`       |        5 |             8 | Massive Multitask Language Understanding |
| `ARCEASY_PRESET`| `arc_easy`   |        0 |             8 | AI2 Reasoning Challenge (easy)  |
| `WIKITEXT_PRESET`| `wikitext`  |        0 |             4 | WikiText-103 perplexity         |

Construct your own when you need a custom configuration:

```python
from llm.evaluation.harness.presets import EvalPreset

MMLU_ALGEBRA = EvalPreset(
    task="mmlu",
    num_fewshot=5,
    batch_size=16,
    limit=200,                                  # smoke-test slice
    task_kwargs={"subject": "abstract_algebra"},
    description="MMLU abstract_algebra, 200-sample smoke test",
)
```

Look up a built-in by name:

```python
from llm.evaluation.harness.presets import get_preset
preset = get_preset("mmlu")
```

Unknown names raise `KeyError` with the list of available built-ins.

### 2. `LlamaLmEvalLM` — *how* to talk to your model

`lm-evaluation-harness` expects model wrappers that implement the
`lm_eval.api.model.LM` protocol (`loglikelihood`,
`loglikelihood_rolling`, `generate_until`). Our
[`LlamaLmEvalLM`](../api/evaluation.md#llm.evaluation.harness.lm_eval_lm.LlamaLmEvalLM)
is a minimal adapter that wraps any `DecoderModel` + tokenizer pair:

```python
import torch
from llm.models import DecoderModel
from llm.tokenization import load_tokenizer
from llm.evaluation.harness.lm_eval_lm import LlamaLmEvalLM

model = DecoderModel.from_pretrained("checkpoints/my-model")
tokenizer = load_tokenizer("checkpoints/my-model/tokenizer.json")

lm = LlamaLmEvalLM(
    model,
    tokenizer,
    batch_size=8,
    max_length=2048,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

Internally `LlamaLmEvalLM` batches requests the way lm_eval expects:

- tokenizes each `(context, continuation)` pair,
- runs the model forward pass in batches of `batch_size`,
- extracts per-token log-probs and compares to greedy argmax for the
  `is_greedy_match` flag,
- for `generate_until`, decodes greedily and stops at any of the
  `until` token sequences.

The wrapper is intentionally small (~250 lines) so the contract is
obvious — `HFLM` would pull in HF-only kwargs (`prefix_token`,
`backend`) that don't apply here.

### 3. `LmEvalAdapter` — *how* to drive the harness

[`LmEvalAdapter`](../api/evaluation.md#llm.evaluation.harness.adapter.LmEvalAdapter)
adds the missing structure on top of `lm_eval.evaluator`:

- preset lookup,
- structured result flattening via `summarize(...)`,
- a soft-dependency boundary that yields a clear `ImportError` on
  hosts without `lm_eval`.

```python
from llm.evaluation.harness.adapter import LmEvalAdapter

adapter = LmEvalAdapter()
adapter.list_tasks()  # every task lm_eval knows about
```

## End-to-end: running a benchmark

```python
from llm.evaluation.harness.adapter import LmEvalAdapter
from llm.evaluation.harness.lm_eval_lm import LlamaLmEvalLM

model, tokenizer = load_my_model_and_tokenizer()
lm = LlamaLmEvalLM(model, tokenizer, batch_size=8)

adapter = LmEvalAdapter()
raw = adapter.run_preset("mmlu", lm)

# raw is the nested lm_eval dict (results / groups / configs).
flat = LmEvalAdapter.summarize(raw)
# -> {"mmlu": {"acc": 0.42, "acc_norm": 0.45}}
```

`run_preset(preset_name, lm, **kwargs)` accepts either a preset
**name** (`"mmlu"`) or an `EvalPreset` instance; extra `kwargs`
override the preset's defaults.

### Smoke-testing with `limit`

The whole point of presets is reproducibility — but the whole point of
*not* hard-coding kwargs is iteration. The two interact well:

```python
from llm.evaluation.harness.presets import EvalPreset
from llm.evaluation.harness.adapter import LmEvalAdapter

smoke = EvalPreset(task="mmlu", num_fewshot=5, limit=20)
LmEvalAdapter().run_preset(smoke, lm)  # 20-sample sanity check
```

## Result handling

`lm-evaluation-harness` returns a deeply nested result tree:

```python
{
    "results": {
        "mmlu": {
            "acc,none":       0.42,
            "acc_norm,none":  0.45,
            "alias,none":     "mmlu",  # string metadata
            ...
        },
        ...
    },
    "groups":  {...},
    "configs": {...},
}
```

`LmEvalAdapter.summarize(results)` flattens this into a stable shape:

```python
{
    "mmlu": {"acc": 0.42, "acc_norm": 0.45},
    ...
}
```

Rules it follows:

- Only the `results` block is flattened; `groups` and `configs` are
  ignored (callers that care about them should look at the raw dict).
- Keys like `"acc,none"` collapse to `"acc"` (the suffix is the
  lm_eval aggregation tag, dropped on purpose — it's almost always
  `"none"` for our tasks).
- Non-numeric values (string aliases) are dropped.
- Numeric values are coerced to `float` for a stable type.

## Common patterns

### Comparing two checkpoints

```python
ckpts = ["checkpoints/baseline", "checkpoints/finetuned-v2"]
results = {}
for ckpt in ckpts:
    model, tok = load(ckpt)
    lm = LlamaLmEvalLM(model, tok, batch_size=8)
    raw = LmEvalAdapter().run_preset("arc_easy", lm)
    results[ckpt] = LmEvalAdapter.summarize(raw)
```

### Single-subject MMLU slice

```python
from llm.evaluation.harness.presets import EvalPreset
from llm.evaluation.harness.adapter import LmEvalAdapter

MMLU_ANATOMY = EvalPreset(
    task="mmlu",
    num_fewshot=5,
    task_kwargs={"subject": "anatomy"},
)
LmEvalAdapter().run_preset(MMLU_ANATOMY, lm)
```

### Running on CPU-only hosts

`LlamaLmEvalLM` honors the `device` argument; pass `"cpu"` explicitly
when no GPU is available:

```python
lm = LlamaLmEvalLM(model, tokenizer, batch_size=2, device="cpu")
```

(WikiText perplexity is the most useful CPU-friendly task — see
[`WIKITEXT_PRESET`][wt-preset].)

[wt-preset]: ../api/evaluation.md#llm.evaluation.harness.presets

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: lm-eval integration requires …` | `lm_eval` is not installed | `uv sync --group eval` or `pip install 'llm[eval]'` |
| `KeyError: unknown preset 'mmlu_xyz'` | Preset name typo | Use one of `mmlu`, `arc_easy`, `wikitext`, or construct an `EvalPreset` directly |
| Result dict is missing `acc_norm` | Some tasks don't define `acc_norm` (e.g. WikiText perplexity) | This is correct — check the raw `results` block to see which metrics lm_eval reported |
| `RuntimeError: … out of memory` | Batch size too large for the GPU | Lower `batch_size` in the preset or the `LlamaLmEvalLM` constructor |

## Related

- [Inference optimization guide](inference.md) — for the
  `KVCache`-backed generation pipeline used by `LlamaLmEvalLM`.
- [Training flow guide](../development/training-flow.md) — for
  in-training perplexity tracking and metrics callbacks (different
  from this offline benchmark harness).
- [Tier 3 ticket #6](../audits/2026-07-12-tickets/27-lm-eval-pipeline.md)
  — the audit follow-up that motivated this pipeline.
