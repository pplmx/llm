# 006. Checkpoint format unification (model.safetensors + meta.json + extra_state.pt)

Date: 2026-07-22

## Status

Accepted

## Context

Training checkpoints are currently a single `torch.save` blob
(`latest.pt`, `best.pt`, `epoch_<N>.pt`) containing the model state
dict, optimizer state, scheduler state, AMP scaler state, training
metadata (epoch / loss / best_loss / model_config), and an optional
`extra_state` dict for task-specific bookkeeping (e.g. the streaming
data cursor from `StreamingTextDataModule`).

This single-file format is convenient to write but has accumulated
three concrete problems:

1. **No partial loading.** Every resume reads the entire blob,
   including the optimizer state (typically the largest component,
   ~3× the model size for Adam). When you only need the model weights
   — e.g. to start a fresh fine-tuning run from a pretrained
   checkpoint, or to push a model to HuggingFace Hub without the
   training cruft — there's no way to skip the optimizer / scheduler
   / scaler overhead. Reading is O(full checkpoint size) instead of
   O(weights only).
2. **Single point of failure.** A corrupt pickle blob kills the
   whole checkpoint. The optimizer state and the model weights are
   stored in the same bytes, so a single bit flip takes both down.
3. **No human-readable metadata.** Inspecting what epoch a checkpoint
   is from, what the best loss was, or what `model_config` was
   committed requires running `python -c "import torch; ..."` to
   unpickle the whole 5+ GB blob. Listing several checkpoints to
   compare losses means loading each one in full.

The HuggingFace compat layer already splits weights (safetensors) +
config (JSON) — see `src/llm/compat/hf_publisher.py:118-162`. The
training checkpoint should follow the same shape so a single
`load_state_dict`-compatible helper works for both code paths.

## Decision

Split the single-file checkpoint into three sidecar files, named with
a shared prefix (`<name>` = `latest` / `best` / `epoch_<N>`):

- **`<name>.safetensors`** — model weights (`state_dict()` only).
  Safetensors is the standard format across the HF / Llama /
  Mistral ecosystems, supports zero-copy mmap loads, is safe against
  pickle deserialization attacks, and is already a transitive dep
  via the `compat` group.
- **`<name>.meta.json`** — training metadata as JSON: `epoch`,
  `loss`, `best_loss`, `model_config` (a JSON-serializable subset of
  the training config — arch fields + PEFT method + optimizer class
  name + any future fields), and a top-level `format_version` field
  for forward-compat.
- **`<name>.extra_state.pt`** — the optimizer / scheduler / scaler
  state dicts + the optional `extra_state` dict. These are arbitrary
  Python objects (Adam moment estimates, lr-scheduler step counts,
  `StreamDataState` instances) that don't fit cleanly into
  safetensors or JSON; keeping them as a single `torch.save` blob is
  fine because they're an order of magnitude smaller than the model
  weights and are only needed on resume.

The `CheckpointManager` writes all three files together, and
`load_checkpoint` accepts both the new three-file layout AND the
legacy single-file layout — existing checkpoints from v0.0.5 and
earlier keep loading without manual migration. The legacy path is
auto-detected by checking whether `<ckp_path>.safetensors` exists;
if not, fall back to `torch.load(<ckp_path>)`.

When the legacy path is loaded, the manager emits a one-line
`DeprecationWarning` recommending `llm-migrate-ckpt <ckp_path>` (a
future CLI helper, deliberately out of scope for this slice — the
in-place conversion can be a follow-up).

The implementation keeps the same `CheckpointManager` public API
(`save_checkpoint` / `load_checkpoint`); the only observable change
is the file layout on disk. No new config fields, no new CLI flags,
no breaking changes to existing training pipelines.

## Consequences

### Positive

- **Partial loads.** `safetensors.torch.load_file("latest.safetensors")`
  skips the optimizer state entirely — useful for Hub publishing
  (already does this via `hf_publisher`), for starting a fresh
  fine-tune from a pretrained base, and for evaluation-only
  inference. The speedup is roughly `state_dict_size / full_ckpt_size`
  on the load.
- **Fault isolation.** A corrupted `.pt` blob for the optimizer state
  no longer prevents you from extracting the model weights. You can
  start a fresh optimizer + scheduler from a clean slate and keep
  training.
- **Human-readable metadata.** `cat latest.meta.json | jq` works.
  Listing checkpoint directories with `ls -la` shows the file sizes
  and let you eyeball which checkpoint is best (combined with the
  `.meta.json` content).
- **Forward compatibility.** `format_version` is a top-level field;
  future schema migrations bump it and add a migration function
  rather than triggering a manual upgrade.
- **Symmetry with HF pub.** The same `safetensors.torch.save_file` /
  `load_file` pair works for both the training checkpoint and the HF
  publisher path. A future slice can share the underlying helper.

### Negative

- **3× more inodes per checkpoint.** `latest.safetensors` +
  `latest.meta.json` + `latest.extra_state.pt` instead of one
  `latest.pt`. Negligible on every modern filesystem.
- **Migration surface.** Users with hundreds of existing v0.0.5
  checkpoints need a way to convert (the future `llm-migrate-ckpt`
  CLI). Until that lands, the legacy loader still works, but new
  checkpoints are written in the split layout — so the two layouts
  coexist in the same directory across an upgrade window.
- **savetensors requires contiguous tensors.** Already handled in
  `hf_publisher.py:156-157` (`.detach().contiguous().clone()`) and
  we copy the pattern. Adds a small per-save memory spike (~2×
  state dict size at the clone step). For a 1.5B model that's a
  ~6 GB peak; well within typical training-host RAM.
- **Optimizer / scheduler state isn't portable.** The `extra_state.pt`
  blob still uses `torch.save`, so you can't move it across major
  torch versions without risk. Same status as today — not
  regressing, but also not fixing.

### Neutral

- **File extension conventions.** `.safetensors` is the
  ecosystem-standard extension (HF, Mistral, Llama); `.meta.json`
  mirrors the `config.json` naming from HF; `.extra_state.pt` is
  unique to us. Anyone scripting around these names needs to know
  about the third one — documented in the ADR + a docstring on the
  helper module.
- **Atomic write semantics.** Each file is still written via
  temp + rename (the existing pattern at `checkpoint.py:60-63`);
  no change to the durability guarantees.

## References

- `src/llm/compat/hf_publisher.py:118-162` — the safetensors +
  config.json split that this ADR mirrors for training checkpoints.
- `src/llm/training/core/checkpoint.py` — the current single-file
  implementation being refactored.
- ROADMAP §阶段十四 "checkpoint 格式统一" — the long-standing
  roadmap bullet this slice closes.
- HuggingFace safetensors docs — the format choice rationale
  (zero-copy mmap, no pickle, ecosystem alignment).
