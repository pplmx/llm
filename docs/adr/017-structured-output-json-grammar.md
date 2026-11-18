# 017. Structured output: a JSON grammar now, a hard logit mask later

Date: 2026-11-18

## Status

Accepted (grammar + post-generation check); Proposed (wiring it as a logit mask)

## Context

The Responses API exposes structured output as `text.format`:

```json
{"text": {"format": {"type": "json_schema", "name": "Person", "schema": {...}}}}
```

OpenAI's contract for it is strong: the returned text parses as JSON **and**
validates against the schema. Applications rely on that — they hand the reply
straight to a typed parser.

Returning a 200 whose text happens to be prose is not a degraded version of that
contract, it is a different one. So the question was never "should we accept the
field" but "how much of the guarantee can this server actually make".

## Decision

Implement the guarantee in two halves, and be explicit about which half runs.

### 1. A JSON grammar (`serving/json_grammar.py`) — DONE

A hand-written character automaton tracks the JSON document's prefix and reports
which characters may legally follow. It is cross-validated against `json.loads`:
`json.dumps` generates the valid side and targeted mutations generate the invalid
side, and a disagreement in **either** direction fails the test. 800 generated
documents were accepted and 30/30 mutated inputs agreed with the standard library.

The automaton is character-based, not token-based, because the tokenizer may be
character-level (`SimpleCharacterTokenizer`) or subword (HF BPE). A character
state machine composes with both: the vocabulary scan decides which *tokens* are
admissible given their text.

### 2. A post-generation schema check — DONE

`validate_against_schema` checks the constraints a caller most often relies on
(`type`, `required`, `properties` recursion, `enum`, `items`,
`additionalProperties: false`) and reports problems with a JSON path. Deliberately
dependency-free: `jsonschema` is not a project dependency, and adding one for this
would be a heavier decision than the feature warrants. Unknown keywords are
ignored rather than guessed at.

A violation is **logged and reported**, not silently returned as if it conformed.

### 3. Hard logit masking — NOT DONE, and the gap is named

`build_logit_mask(grammar, id_to_token)` exists and is tested: given the current
grammar state and the vocabulary, it returns the set of admissible token ids. It
is **not wired into the generation loop**.

Why not: `stream_generate` (`generation/eager.py`) has no per-step hook. Adding one
means threading a new parameter through `GenerationConfig`, `backends.py` (5 call
sites), `generation_service.py` and the eager loop — a plumbing change across four
files whose only effect is to make masked decoding possible. That is a reasonable
next step, and this ADR records it as the remaining gap rather than pretending the
guarantee is complete.

Until then, the honest statement is: **the server checks, it does not prevent.**

## Consequences

### Good

- The protocol surface is complete: a client can send `text.format` and the SDK
  parses the request.
- The JSON grammar is independently verified against the standard library, so it
  does not rest on hand-picked examples.
- The half that is missing is a *plumbing* gap with a tested component behind it,
  not an unbuilt design.

### Bad

- A caller that trusts `text.format` to be enforced can still receive prose. This
  is the current limitation and the reason the CHANGELOG entry says so plainly
  instead of claiming parity with OpenAI.
- The schema rendering costs prompt tokens. On the demo config (`max_seq_len=64`)
  a realistic schema exceeds the whole window — measured: a 367-token prompt
  against a 64-token model. Structured output needs a model with a real context
  window, which is a property of the deployment, not a bug here.

## Alternatives considered

**Prompt-only, no grammar.** Rejected: a markdown fence or a trailing sentence is
then invisible to the server, so it could not even report the failure.

**Hard mask first, protocol second.** Rejected as an ordering: the grammar has to
be correct before masking it means anything, and the grammar is the part that is
independently testable against `json.loads`.

**Depend on `jsonschema` for validation.** Deferred: one dependency for a
partially-implemented checker is a poor trade. If callers need full JSON Schema
semantics, that decision should be made on its own merits.

## References

- `src/llm/serving/json_grammar.py`
- `src/llm/serving/routers/responses.py`
- `tests/serving/test_structured_output.py`
- OpenAI structured outputs: <https://platform.openai.com/docs/guides/structured-outputs>
