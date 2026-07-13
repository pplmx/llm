# Speculative decoding in continuous batching (Tier 3 #9)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #9
("Speculative decoding in continuous batching")

## Description
Speculative decoding (Leviathan et al., 2023) uses a small **draft**
model to speculate ``gamma`` candidate tokens ahead of the larger
**target** model. The target scores all ``gamma + 1`` positions in a
single forward pass; the algorithm either accepts each candidate
(probabilistically, preserving the target distribution) or samples a
correction token. Net effect: most generated tokens cost one draft
forward, not one target forward — 2–3× throughput on long-context
decode is typical.

This ticket establishes the **foundation slice**: a self-contained
``speculative_generate`` generator that operates on two
``DecoderModel`` instances and emits decoded chunks via the standard
generator protocol, plus a ``SpeculativeDecodingBackend`` that
integrates with the existing ``BACKEND_REGISTRY``.

Wiring speculative decoding into
:class:`llm.serving.batch_engine.ContinuousBatchingEngine` (so the
existing ``/generate`` endpoint can opt in via configuration) is a
follow-up ticket — this slice ships the algorithm + backend contract.

## Acceptance criteria
- [ ] ``src/llm/generation/speculative.py`` exposes
      ``speculative_generate(target, draft, tokenizer, prompt,
      max_new_tokens, *, gamma=5, ...)`` as a streaming generator.
- [ ] The acceptance criterion is correct: ``P(accept x_i) =
      min(1, q_target(x_i) / q_draft(x_i))``; on rejection, sample
      from ``(q_target - q_draft)+ / sum((q_target - q_draft)+)``;
      on full acceptance, sample the bonus from ``q_target``.
- [ ] Algorithm preserves the target distribution: a property-based
      test asserts ``KL(target_marginal || speculative_marginal)``
      is bounded by the sampling noise across a range of prompts.
- [ ] ``SpeculativeDecodingBackend`` registered in
      ``BACKEND_REGISTRY`` (entry point: ``llm.generation_backends``
      -> ``speculative``).
- [ ] Sampling parameters (``temperature``, ``top_k``, ``top_p``,
      ``repetition_penalty``) are honored and produce the same
      distribution as the eager backend given identical inputs.
- [ ] Stops on EOS or ``max_new_tokens``.
- [ ] New ``tests/generation/test_speculative.py``:
      - End-to-end: ``speculative_generate`` produces the same
        tokens as the eager backend for ``temperature=0`` (greedy).
      - Distribution preservation (hypothesis): sampled tokens
        from ``speculative_generate`` match the target model's
        greedy distribution within sampling noise.
      - Backend lookup: ``get_generation_backend("speculative",
        target_model=..., draft_model=..., tokenizer=...)``.
      - ``gamma=0`` raises ``ValueError``.
- [ ] Doc: ``docs/guides/inference.md`` section explaining the
      algorithm, when it helps (long-context decode, large target
      models), and the draft-model selection trade-off.

## Estimate
~3 weeks

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `perf`, `generation`,
`correctness`
