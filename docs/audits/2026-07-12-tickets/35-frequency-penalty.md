# frequency_penalty in OpenAI-compat serving (Tier 3 #35)

## Source

`serving/schemas.py:110` declares `frequency_penalty: float = Field(0.0, description="Frequency penalty (not implemented).")`. The "not implemented" hint is the explicit signal that this OpenAI-compatible parameter is documented but the backend silently ignores it. `tests/serving/test_openai_compat.py:166` (``test_chat_completions_with_penalties``) sends the parameter and asserts only that the response is 200 — it does not verify any behavioural effect, because none exists today.

## Description

Wire `frequency_penalty` end-to-end so the OpenAI-compatible chat
endpoint actually penalises tokens in proportion to their count in
the generated text. This mirrors OpenAI's semantics:

> "Positive values penalize new tokens based on their existing
> frequency in the text so far, decreasing the model's likelihood
> to repeat the same line verbatim."

The implementation subtracts `frequency_penalty * count(token)` from
the logits before sampling. This is the standard formulation used
in OpenAI's reference implementation and in HuggingFace's
`LogitsProcessor` (`FrequencyPenaltyLogitsProcessor`).

The slice touches one new sampling helper, the shared
`GenerationConfig`, every generation backend, and the two serving
routers. No streaming-protocol changes are required (unlike a
`stop`-sequences slice), which keeps the scope contained.

## Design

- `src/llm/generation/sampling.py`:
    - Add `apply_frequency_penalty(logits, token_ids, frequency_penalty) -> logits`.
      Mirrors the existing `apply_repetition_penalty` shape: clone
      the logits, build a sparse penalty tensor, subtract. Returns
      the adjusted 1D logits (no in-place mutation).
    - No-op when `frequency_penalty == 0.0` or `token_ids` is empty.

- `src/llm/generation/backends.py`:
    - Extend `GenerationConfig` with `frequency_penalty: float = 0.0`.
    - `EagerGenerationBackend`, `BatchedGenerationBackend`,
      `SpeculativeDecodingBackend`: forward `frequency_penalty` to
      the underlying generate function.

- `src/llm/generation/eager.py`:
    - `stream_generate`, `generate`, `batch_generate`: accept
      `frequency_penalty`, pass it through.
    - `stream_generate`: in the per-token loop, apply the penalty
      after `apply_repetition_penalty` when non-zero.

- `src/llm/generation/speculative.py`:
    - `speculative_generate`: forward `frequency_penalty`. The
      penalty is applied on each verification pass.

- `src/llm/serving/generation_service.py`:
    - `generate`, `stream`, `batch_generate`: accept
      `frequency_penalty`, plumb into `GenerationConfig`.

- `src/llm/serving/routers/chat.py`:
    - Pass `request.frequency_penalty` through to `_sync_generate`
      (and the streaming generator). Stop sequences are explicitly
      out of scope for this slice.

- `src/llm/serving/routers/generate.py`:
    - `GenerationRequest` gains `frequency_penalty`. The handler
      forwards it to `service.generate`.

- `src/llm/serving/schemas.py`:
    - Drop the "not implemented" hint on `frequency_penalty`.

## Acceptance criteria

- [ ] `apply_frequency_penalty` is exposed and tested: zero
      penalty is a no-op; non-zero penalty decreases the logit of
      any token that appeared in `token_ids` in proportion to its
      count.
- [ ] `GenerationConfig` includes `frequency_penalty`. Round-trip
      via the dataclass works.
- [ ] Every backend (`Eager`, `Batched`, `Speculative`) forwards
      `frequency_penalty` correctly. The existing tests still
      pass when the new parameter is left at its default of `0.0`.
- [ ] Sending `frequency_penalty > 0` to `/v1/chat/completions`
      changes the generated text compared to `frequency_penalty = 0`
      (verified by a smoke test that runs both calls and compares).
- [ ] `schemas.ChatCompletionRequest.frequency_penalty` no longer
      says "(not implemented)" in its docstring.
- [ ] `tests/serving/test_openai_compat.py::test_chat_completions_with_penalties`
      is updated to actually assert that the parameter is plumbed
      (e.g. by mocking the sampling layer and verifying the call).

## Out of scope (intentional)

- **Stop sequences (`stop`)** — requires a streaming-protocol
  change to surface `finish_reason="stop"` (vs the current
  hard-coded `"stop"`). That's a separate slice because it
  touches `StreamingResponse`, `GenerationBackend.stream`, and
  every consumer of `stream_request`.
- **`presence_penalty`** — currently aliased to
  `repetition_penalty = 1.0 + presence_penalty`. The proper
  OpenAI semantics (`presence_penalty` penalises any token that
  appeared at all, irrespective of count) differ from
  `repetition_penalty`'s "divide or multiply by N" formulation.
  Splitting these correctly is a separate slice.
- **Per-batch `frequency_penalty` array** — OpenAI accepts the
  same scalar across all prompts in one batch. The current
  `batch_generate` API already matches that.

## Estimate

~1 day. The new sampling helper is ~10 LOC, the plumbing is
mechanical across 6 modules, and the test surface is small (one
unit test for the helper, one integration test for the router).

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `serving`, `generation`,
`openai-compat`
