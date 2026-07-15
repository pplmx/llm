# presence_penalty semantic separation (Tier 3 #37)

## Source

`src/llm/serving/routers/chat.py:81`:
```python
repetition_penalty = 1.0 + request.presence_penalty
```

This conflates two OpenAI-distinct semantics:

- **`presence_penalty`** (OpenAI): penalises a token **once** if it
  has appeared at all in the generated text, regardless of how
  many times. Range `[-2.0, 2.0]`. `0.0` is no penalty.
- **`repetition_penalty`** (HuggingFace-style): divides a positive
  seen-token logit by N or multiplies a negative one by N. Range
  typically `[1.0, 2.0]`. `1.0` is no penalty.

The hack `repetition_penalty = 1.0 + presence_penalty` produces
the **wrong behaviour** for three of the four cases the user can
hit:

| `presence_penalty` | Intended effect | `1.0 + presence_penalty` actually does |
|---|---|---|
| `0.0` (off) | No change. | `repetition_penalty = 1.0` → no change. ✅ |
| `+0.5` (penalise) | Subtract `0.5` from each seen token once. | `repetition_penalty = 1.5` → divide positive logits by 1.5, multiply negatives by 1.5. The penalty **scales with the logit's magnitude**, not flat. ❌ |
| `−0.5` (boost) | Add `0.5` to each seen token once. | `repetition_penalty = 0.5` → divide positive logits by 0.5 (= boost) and multiply negatives by 0.5 (= reduce penalty). Net effect is biased, not flat. ❌ |
| `+1.0` (max) | Subtract `1.0` from each seen token once. | `repetition_penalty = 2.0` → divides positives by 2, multiplies negatives by 2. Bounded by logit magnitudes. ❌ |

This slice implements `presence_penalty` with the OpenAI-correct
semantics end-to-end and removes the alias.

## Description

Add `apply_presence_penalty` to the sampling helpers and plumb a
new `presence_penalty` field through the same path that #35
(`frequency_penalty`) carved out. The OpenAI-compatible chat
endpoint then sends `request.presence_penalty` to the service
**as itself**, not folded into `repetition_penalty`.

`frequency_penalty` (penalty × count) and `presence_penalty`
(penalty × 1) are intentionally separate functions because their
mathematics differ: a single helper with a `count`-flag would
obscure the per-call cost shape and make the schema range harder
to reason about (presence can be negative and *boost* tokens,
which would interact non-trivially with a `count` flag).

## Design

- `src/llm/generation/sampling.py`:
    - New `apply_presence_penalty(logits, token_ids, presence_penalty)`.
      Subtraction is flat (no count scaling): any token that has
      appeared at least once in `token_ids` is penalised by exactly
      `presence_penalty`. Implements OpenAI's documented semantics
      directly. Clones logits — input is never mutated.
    - No-op when `presence_penalty == 0.0` or `token_ids` is empty.

- `src/llm/generation/backends.py`:
    - `GenerationConfig` gains `presence_penalty: float = 0.0`.
    - All three backends forward the field (default `0.0` is a
      no-op so existing tests pass unchanged).

- `src/llm/generation/eager.py`:
    - `stream_generate` / `generate` / `batch_generate` accept
      `presence_penalty`. `stream_generate` applies it per-token
      after `frequency_penalty`. `batch_generate` applies it
      per-row.

- `src/llm/generation/speculative.py`:
    - `speculative_generate` accepts and applies `presence_penalty`
      to the draft logits.

- `src/llm/serving/generation_service.py`:
    - `generate` / `stream` / `batch_generate` /
      `_generation_config` all forward `presence_penalty`.

- `src/llm/serving/routers/chat.py`:
    - **Replace** `repetition_penalty = 1.0 + request.presence_penalty`
      with `repetition_penalty = 1.0` (the OpenAI chat endpoint
      does **not** expose `repetition_penalty`, so the default
      `1.0` is correct).
    - Pass `presence_penalty=request.presence_penalty` to
      `_sync_generate` and the streaming generator.
    - `_chat_stream_generator` signature gains the parameter.

- `src/llm/serving/routers/generate.py`:
    - `GenerationRequest` does not expose `presence_penalty`
      (it's an internal endpoint), but the router still forwards
      `request.repetition_penalty` as today. No change needed
      beyond what `GenerationConfig` already accepts.

- `src/llm/serving/schemas.py`:
    - `ChatCompletionRequest.presence_penalty`: range `[-2.0, 2.0]`
      (matches OpenAI), description updated to drop the
      "(mapped to repetition_penalty)" hint.

## Acceptance criteria

- [x] `apply_presence_penalty` is exposed and tested: zero
      penalty is a no-op; non-zero penalty subtracts a flat amount
      from every seen token regardless of count.
- [x] `GenerationConfig` includes `presence_penalty`. Round-trip
      works.
- [x] All three backends forward `presence_penalty`. Existing
      tests still pass with the new parameter at its default of
      `0.0`.
- [x] `serving/routers/chat.py` no longer computes
      `repetition_penalty = 1.0 + presence_penalty`. The
      `presence_penalty` reaches `service.generate` as its own
      kwarg.
- [x] `ChatCompletionRequest.presence_penalty` range is
      `[-2.0, 2.0]` and the description no longer says
      "(mapped to repetition_penalty)".
- [x] End-to-end: sending `presence_penalty=0.6` to
      `/v1/chat/completions` reaches the service's
      `generate(prompt=..., presence_penalty=0.6)`.

## Out of scope (intentional)

- **`stop` sequences** — already noted as out-of-scope in #35.
  Requires StreamingResponse / `finish_reason` protocol changes.
- **`repetition_penalty` schema range widening** — currently
  `ge=1.0` because the helper bails on `< 1.0`. Widening it to
  `[-2.0, 2.0]` for symmetry with the OpenAI params is a
  separate behaviour change (it would change the divide/multiply
  math). Not part of this slice.
- **Adding `presence_penalty` to `GenerationRequest` /
  `BatchGenerationRequest`** — those are the internal endpoints,
  not the OpenAI-compat one. Adding the field would be a small
  mechanical change but is not motivated by a documented user
  need today.

## Estimate

~0.5 day. The helper is ~15 LOC, the plumbing is mechanical
across 6 modules (same shape as #35), and the test surface is
small (one unit test for the helper, one router plumbing test
to confirm the hack is gone).

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `serving`, `generation`,
`openai-compat`
