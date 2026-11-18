# 017. Structured output: a JSON grammar now, a hard logit mask later

Date: 2026-11-18

## Status

Accepted (grammar + post-generation check + logit mask, with a measured gap)

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

### 3. Hard logit masking — DONE, with a measured gap

`GrammarConstrainedProcessor` is wired into the generation loop: `stream_generate`
takes an optional `logits_processor`, threaded from `GenerationConfig` through
`backends.py` and `generation_service.py` to the `/v1/responses` route. The
processor masks every token id whose text the grammar refuses, so a masked step
cannot emit text that makes the reply unparseable.

Measured on the demo model, 20 seeded runs, `max_new_tokens=60`:

| setting        | valid JSON  |
| -------------- | ----------- |
| unconstrained  | **0 / 20**  |
| grammar-masked | **17 / 20** |

Not 100%, and the residue is understood rather than waved away:

- **Truncation.** A run that hits the token budget mid-string produces a legal
  *prefix* (the processor is behaving correctly) but not a complete document.
  OpenAI reports this as `status="incomplete"`; here the post-generation check
  catches it and the response is flagged.
- **Token-boundary edge cases.** The automaton is character-based while the
  vocabulary is token-based, so a token that *starts* legally but is only partly
  consumed can leave slack the model exploits. Production constrained-decoders
  (outlines, llguidance) solve this with token-level automata plus rollback; that
  is a substantially larger piece of work and is not attempted here.

So the contract is now: **the server prevents most malformed replies and checks
every one of them.** That is stronger than the previous "we only check", and
weaker than OpenAI's guarantee. Both halves are stated in the CHANGELOG.

Two defects found while wiring it, both caught by measurement rather than review:

1. `accept()` had a **side effect** — it called `_value_completed()` to decide
   whether a top-level number was finished, which advanced the automaton. A
   constrained run emitted `9`, the next `accept()` flipped the state to `done`,
   every further token became illegal, the processor gave up, and an unconstrained
   tail followed. `accept()` is now pure and a test pins that it does not mutate.
2. Leading whitespace had to be **refused** at position 0. JSON permits it, so the
   grammar allowed it, and a model free to emit spaces emits nothing else —
   measured: 40 constrained tokens of pure blank.

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
