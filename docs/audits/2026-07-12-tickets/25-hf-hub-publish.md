# HF Hub publish pipeline (Tier 3 #7)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #7
("HF Hub publish pipeline"), Tier 3 #7

## Description
Models trained with this project cannot currently be published to
HuggingFace Hub, so users who want to share weights must hand-write
the conversion. The reverse of `compat/hf_loader.from_pretrained` is
missing: there is no `save_pretrained` for the
:class:`llm.models.DecoderModel`, and no `push_to_hub` helper.

This ticket closes the gap by adding a **roundtrip-publishable**
serializer:

1. ``save_pretrained(save_dir)`` writes a HuggingFace-shaped
   ``config.json`` + ``model.safetensors`` (Llama convention — the
   reverse of what ``from_pretrained`` already loads).
2. ``push_to_hub(repo_id, ...)`` calls ``save_pretrained`` to a
   staging directory, then uploads via ``huggingface_hub.upload_folder``.
3. Both functions are soft-dependency-friendly: ``safetensors`` is
   required for save, ``huggingface_hub`` is required for push —
   each raises a clear ``ImportError`` with the install hint when
   missing.

The roundtrip guarantee: ``from_pretrained`` (existing code) must be
able to load a directory produced by ``save_pretrained``. This pins
the contract for the reverse weight mapping.

## Acceptance criteria
- [ ] ``src/llm/compat/hf_publisher.py`` exposes
      ``save_pretrained(model, save_dir)`` and
      ``push_to_hub(model, repo_id, ...)``.
- [ ] ``save_pretrained`` writes ``config.json`` (Llama-style) and
      ``model.safetensors``. Output is loadable by the existing
      ``from_pretrained``.
- [ ] Reverse weight mapping added to ``compat/weight_mapping.py``
      (``our → HF``) plus a roundtrip invariant test
      (``convert_hf_weights(convert_our_weights(sd)) == sd``).
- [ ] ``push_to_hub`` uploads via ``huggingface_hub.upload_folder``
      (or ``HfApi.upload_folder``); auth via the standard HF token
      env (``HF_TOKEN`` / ``huggingface-cli login``).
- [ ] ``huggingface_hub`` and ``safetensors`` added to ``[compat]``
      optional dependency group (mirrored in
      ``[project.optional-dependencies]``); install via
      ``pip install 'llm[compat]'``.
- [ ] New ``tests/compat/test_hf_publisher.py``:
      - Roundtrip: small model → ``save_pretrained`` → ``from_pretrained``
        produces a model whose forward pass matches within tolerance.
      - ``push_to_hub`` smoke test using a mocked ``upload_folder``.
      - Reverse mapping unit test (named-tensor equality).
      - ``ImportError`` raised on missing ``huggingface_hub``.
- [ ] Doc section in ``docs/guides/inference.md`` describing the
      publish flow, including auth setup and a CLI example.

## Estimate
~1 week

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `compat`, `hf`,
`distribution`, `correctness`
