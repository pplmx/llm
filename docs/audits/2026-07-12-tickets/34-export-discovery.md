# Export discovery: docs sync (Tier 3 #34)

## Source

This is a **doc-sync slice** carved out after Tier 3 #32
(EXPORT_REGISTRY) and Tier 3 #33 (TorchScript target) shipped.
The two implementation tickets were committed but the architecture
reference, the ADR index, and the ROADMAP checkboxes were not
updated to reflect them — so a user landing on the docs page
would still see the pre-#32 picture.

## Description

Sync three doc surfaces to the current state of the codebase:

1. `docs/reference/architecture.md`:
   - Expand the `export/` tree entry from a single
     `onnx.py` line to four lines
     (`registry.py`, `onnx.py`, `torchscript.py`, `_wrapper.py`).
   - Add `llm.export_backends` / `EXPORT_REGISTRY` to the
     plugin-kernel table.

2. `docs/adr/005-export-registry.md`:
   - New ADR recording the architectural decision to add
     `EXPORT_REGISTRY` (parity with `BACKEND_REGISTRY`),
     referencing Finding BH in the 2026-07-12 audit.
   - Follows the same Context / Decision / Consequences structure
     as the other four ADRs.

3. `docs/adr/README.md`:
   - Add the new ADR-005 link.
   - Add the missing `004-paged-attention-serving.md` link (the
     file existed in the directory but was not listed in the
     index — pre-existing drift).

4. `ROADMAP.md` §阶段十四:
   - Mark the four already-shipped items as done with a
     short reason: `safetensors 保存` (shipped in #25),
     `TorchScript 导出` (#33), `Export registry` (#32),
     `HuggingFace Hub publish` (#25).
   - Keep `checkpoint 格式统一` unchecked with a one-line note
     explaining the gap (training checkpoint still uses
     `torch.save`; a follow-up slice can split model weights →
     safetensors + metadata → pickle/json).

5. `docs/audits/2026-07-12-tickets/README.md`:
   - Add a row for ticket #34.
   - Bump the Tier 3 counter (now 10/9 audit foundation slices
     + 2 post-audit slices).

## Design

Pure doc work — no source-code changes. Each edit is mechanical
and follows the existing format of the surrounding content.

## Acceptance criteria

- [x] `architecture.md` §Plugin Kernel table includes
      `llm.export_backends` / `EXPORT_REGISTRY`.
- [x] `architecture.md` §项目结构 tree lists all four files under
      `export/`.
- [x] `docs/adr/005-export-registry.md` exists with
      Status / Context / Decision / Consequences.
- [x] `docs/adr/README.md` lists ADR-005 and the previously
      missing `004-paged-attention-serving.md`.
- [x] `ROADMAP.md` §阶段十四 checkboxes reflect the shipped
      state, with notes on the still-open `checkpoint 格式统一`.
- [x] `mkdocs build --strict` succeeds without warnings on the
      modified pages (where the docs build is wired up).

## Out of scope (intentional)

- A real "unify checkpoint format" slice — that requires deciding
  whether to split model weights into a separate safetensors file,
  how to handle backward compatibility with `.pt` checkpoints, and
  what happens to optimizer / scheduler / scaler / extra_state
  payload. That's a non-trivial design discussion, not a doc sync.
- Real DeepSeek-V2-style MLA — separate, larger architectural
  slice (referenced as out-of-scope in #31's docstring).
- Renumbering the two `004-*` ADRs into a clean sequence — that's
  a wider renaming task; this slice only fixes the index entry.

## Estimate

~30 min. The implementation is in this single commit; the only
verification beyond git diff is `mkdocs build --strict` (or the
project's equivalent) on the modified pages.

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `export`, `docs`,
`registry`
