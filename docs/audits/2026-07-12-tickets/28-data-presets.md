# Data preset files for common pretraining datasets (Tier 3 #8)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #8
("Eval preset files (C4, Pile, RedPajama — per ROADMAP P0)"),
ROADMAP §1 预训练完善 (P0): "C4 / The Pile / RedPajama 预设配置
(CLI/YAML 模板, 复用 SOURCE_REGISTRY)"

## Description
The project ships a streaming data pipeline
(`StreamingTextDataModule` + `HFStreamTextSource`) but users have to
hand-author the `DataConfig` triples for every well-known dataset
(`dataset_name`, `dataset_config`, `text_column`). For C4 alone that's
`("allenai/c4", "en", "text")`; for RedPajama it's seven different
config names (`"arxiv"`, `"book"`, `"commoncrawl"`, `"c4"`, `"github"`,
`"stackexchange"`, `"wikipedia"`).

This ticket closes the gap:

1. ``DatasetPreset`` — frozen dataclass carrying
   ``dataset_name``, ``dataset_config``, ``dataset_split``,
   ``text_column``, ``description``.
2. ``apply_to_config(config, preset)`` — mutates a :class:`DataConfig`
   in place to bind it to the preset. Existing unrelated fields
   (``max_seq_len``, ``tokenizer_type``, …) are preserved.
3. Built-in presets for the three families:
   - ``C4_PRESET`` (``allenai/c4`` / ``en`` / ``text``)
   - ``THEPILE_PRESET`` (``monology/pile-uncopyrighted`` / ``None`` / ``text``)
   - ``REDPAJAMA_PRESETS`` (a dict keyed by subset:
     ``arxiv``, ``book``, ``common_crawl``, ``c4``, ``github``,
     ``stackexchange``, ``wikipedia``) — each entry is itself a
     :class:`DatasetPreset`.
4. ``resolve_preset(name)`` lookup helper + ``list_presets()`` for
   discoverability.

The presets are intentionally **decoupled from the ``datasets``
package** — this module imports nothing from ``llm.data.datasets`` or
``llm.data.modules``, only :class:`DataConfig`. That keeps the import
cheap on hosts that don't have ``datasets`` installed.

## Acceptance criteria
- [ ] ``src/llm/data/presets.py`` exposes :class:`DatasetPreset`
      with the four fields above plus ``description``.
- [ ] Built-in presets ship out of the box: ``C4_PRESET``,
      ``THEPILE_PRESET``, ``REDPAJAMA_PRESETS`` (dict with 7 subsets).
- [ ] ``apply_to_config(config, preset)`` mutates ``data_source`` to
      ``"hf"`` and sets the four HF fields without touching
      ``max_seq_len``, ``tokenizer_*``, or validation fields.
- [ ] ``resolve_preset(name)`` looks up by dataset id or preset name;
      unknown names raise ``KeyError`` with the available list.
- [ ] New ``tests/data/test_presets.py`` covering:
      - preset construction + immutability,
      - ``apply_to_config`` mutates only the four HF fields,
      - ``resolve_preset`` happy path + unknown name error,
      - the ``list_presets()`` enumerator.
- [ ] Doc: ``docs/guides/data.md`` section listing the built-in
      presets and showing a YAML/CLI snippet.

## Estimate
~2 weeks

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `data`, `ux`, `p0-roadmap`
