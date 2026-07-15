# Add a dedup wrapper to the TextSource pipeline (T3 #39)

## Source

`ROADMAP.md` §下一步探索方向 → "数据质量过滤和去重工具" (P0 pretraining
productization). Builds on `T3 #28` (data presets: C4 / Pile / RedPajama) —
picking a dataset by name is half the pretraining story, the other half is
cleaning it. Also referenced in the
[technical due-diligence](../2026-07-12-technical-due-diligence.md) as the
remaining unchecked P0 item.

## Description

The streaming data pipeline ships two `TextSource` implementations
(`LocalLineTextSource`, `HFStreamTextSource`) plus a `SOURCE_REGISTRY` for
pluggable extension. The `llm.data.presets` module makes C4 / The Pile /
RedPajama one config line away. **What it does not ship is any form of
content-level filtering**: every record the source yields is forwarded to
the tokenizer, including duplicates that are extremely common in
web-crawl-derived corpora (C4 alone has been measured at >7% exact
duplicates; The Pile at ~30% near-duplicates).

This ticket carves out a foundation slice: an exact-dedup wrapper that
plugs into the existing `TextSource` protocol. The wrapper:

1. wraps any inner `TextSource`,
2. hashes the normalized text and drops records whose hash has already
   been yielded this run,
3. optionally loads a pre-populated "seen hashes" file on construction so
   dedup state is shared across runs / shards,
4. optionally appends new hashes to that file so dedup state grows
   monotonically,
5. exposes a stable `source_fingerprint()` that includes the inner
   source's fingerprint plus the dedup strategy — so checkpoint-resume
   validation (`validate_source_fingerprint`) still catches configuration
   drift.

The slice is exact-dedup only (SHA-256 of normalized text). Fuzzy
deduplication via MinHash / LSH is a follow-up that can layer on top of
the same wrapper — the interface intentionally allows `normalize` and
`hash_algo` to be swapped.

## Acceptance criteria

- [ ] New `DedupTextSource` class in `src/llm/data/sources.py`, subclass
      of `TextSource`, with the documented constructor signature
      (`inner`, `normalize`, `seen_hashes_path`, `write_seen_hashes`,
      `hash_algo`).
- [ ] Default `normalize` strips and collapses internal whitespace runs;
      **does not lowercase** (case-sensitive by default — conflating
      "Apple" the company with "apple" the fruit is worse than missing
      some duplicates).
- [ ] `iter_texts(skip=N)` delegates `skip` to the inner source so the
      `line_index` resume semantics used by `StreamingTextDataset` stay
      consistent with non-dedup sources.
- [ ] `source_fingerprint()` returns `{"type": "dedup", "inner": <inner fp>,
      "hash_algo": <algo>, "normalize": <repr>, "seen_hashes_path": <abs path> | <absent>}`
      so `validate_source_fingerprint` can spot config drift.
- [ ] Two new `SOURCE_REGISTRY` entries: `dedup_local` and `dedup_hf`,
      each wrapping the corresponding inner builder and reading
      `seen_hashes_path` / `write_seen_hashes` / `hash_algo` off
      `DataConfig` via `getattr` defaults.
- [ ] `DataConfig.data_source` regex widened to
      `^(local|hf|dedup_local|dedup_hf)$` and three new optional fields
      added: `seen_hashes_path: str | None`,
      `write_seen_hashes: bool = False`,
      `hash_algo: str = "sha256"` (regex-validated).
- [ ] New test file `tests/data/test_dedup_source.py` covers: exact-dup
      drop, whitespace-normalized dup drop, case-sensitivity (preserves
      case-only dups), pre-loaded seen-hashes file drops matching
      records, `write_seen_hashes=True` appends new hashes, source
      fingerprint reflects inner + dedup config, `build_text_source`
      resolves `dedup_local` / `dedup_hf` with the right inner source,
      missing-config error paths for both new entries.
- [ ] `CHANGELOG.md` `[Unreleased] ### Added` gets an entry cross-
      referencing T3 #39.
- [ ] `ROADMAP.md` §1 P0 box "数据质量过滤和去重工具" is checked.
- [ ] `docs/audits/2026-07-12-tickets/README.md` status snapshot adds
      `#39` to the shipped list.

## Non-goals (deliberately deferred)

- Fuzzy / near-duplicate dedup (MinHash / LSH) — separate follow-up
  ticket once exact-dedup is shipped and exercised.
- Bloom-filter-style probabilistic seen-set — current set is
  deterministic and disk-backed; Bloom is an optimisation that needs
  measured benefit to justify.
- MinHash integration into `SOURCE_REGISTRY` as a sibling of
  `dedup_local` / `dedup_hf` — same reason; defer until the
  exact-dedup API is validated in real pretraining.

## Estimate

~1 focused iteration (~1 day). Mostly mechanical: one new class, two
registry entries, one config widening, one test file, doc sync. The
interface design is the only piece that benefits from a second pass.

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `data`, `p0-pretraining`
