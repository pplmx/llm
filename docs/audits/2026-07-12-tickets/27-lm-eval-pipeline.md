# lm-eval pipeline integration (Tier 3 #6)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #6
("Eval pipeline integration with lm_eval"), Tier 3 #6

## Description
The project ships a thin ``LmEvalAdapter`` wrapper at
``src/llm/evaluation/harness/adapter.py`` that delegates to
``lm_eval.evaluator.evaluate``. The contract is leaky:

* No adapter from :class:`llm.models.DecoderModel` to lm_eval's
  ``HFLM``-style interface (must hand-roll a wrapper each time).
* Result objects are passed through opaquely — no structured
  extraction of ``acc`` / ``acc_norm`` / ``perplexity`` etc. into
  the project's reporting format.
* No preset system — running ``mmlu`` requires ``["mmlu"]`` plus
  remembering to pass ``num_fewshot`` and other kwargs.

This ticket closes the gap:

1. ``LlamaLmEvalLM`` — a minimal ``lm_eval.api.model.LM`` subclass
   that wraps a :class:`DecoderModel` + tokenizer. Batched loglikelihood
   + greedy-until generation, matching what lm_eval expects from
   ``HFLM``.
2. ``EvalPreset`` — a frozen dataclass carrying ``task``,
   ``num_fewshot``, ``batch_size``, ``limit`` and other benchmark
   kwargs. Ship three presets out of the box (``mmlu``,
   ``arc_easy``, ``wikitext``) and let users extend.
3. Strengthen ``LmEvalAdapter``: structured result extraction
   (single-task vs multi-task shapes, ``results`` vs ``groups``),
   CLI-friendly ``summarize(results)`` that emits a Markdown table.

## Acceptance criteria
- [ ] ``src/llm/evaluation/harness/lm_eval_lm.py`` exposes
      ``LlamaLmEvalLM(model, tokenizer, batch_size=8, max_length=None)``
      implementing the lm_eval ``LM`` protocol (``loglikelihood``,
      ``loglikelihood_rolling``, ``generate_until``).
- [ ] ``src/llm/evaluation/harness/presets.py`` exposes
      ``EvalPreset`` dataclass and three built-in presets:
      ``MMLU_PRESET``, ``ARCEASY_PRESET``, ``WIKITEXT_PRESET``.
- [ ] ``src/llm/evaluation/harness/adapter.py`` extended with
      ``run_preset(preset, model, tokenizer)`` and
      ``summarize(results)`` helpers.
- [ ] ``lm_eval`` remains an optional dependency: importing the
      new modules never raises when ``lm_eval`` is missing;
      instantiating ``LlamaLmEvalLM`` / calling
      ``adapter.evaluate()`` raises a clear ``ImportError`` with
      the install hint.
- [ ] New ``tests/evaluation/test_lm_eval_pipeline.py``:
      - ``EvalPreset`` construction + ``to_lm_eval_kwargs()``.
      - ``summarize()`` flattens lm_eval's nested result shape into
        a flat dict (using a fixture result, not a real eval).
      - Soft-dependency contract: ``LlamaLmEvalLM`` raises when
        ``lm_eval`` is missing.
- [ ] Doc: ``docs/guides/evaluation.md`` section on running
      benchmarks via the lm-eval pipeline.

## Estimate
~2 weeks

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `eval`, `correctness`,
`optional-dep`
