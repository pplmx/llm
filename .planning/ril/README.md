# RIL Data Store

`graph.json` in this directory is the **single source of truth** for the
repository-intelligence-layer (RIL) typed engineering graph used by the
autonomous engineering loop (see `.agents/skills/graph-engineering/SKILL.md`).

- **Never edit `graph.json` by hand.** All reads and writes go through the
  CLI: `.agents/skills/graph-engineering/scripts/ril.py` (schema, edge typing,
  optimistic locking, lifecycle and consistency rules are enforced there).
- The file is committed and auditable; node/edge history is preserved, never
  rewritten (decisions are superseded, not deleted; stale nodes are marked,
  not removed).
- Commit messages reference node ids for traceability, e.g.
  `fix(core): ... (RIL TASK-001, ISS-001)`.
- Schema and CLI reference:
  `.agents/skills/graph-engineering/references/ril-schema.md`.
