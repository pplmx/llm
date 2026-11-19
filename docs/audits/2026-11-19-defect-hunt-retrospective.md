# Defect Hunt Retrospective — 2026-11-19 iteration series

**Date**: 2026-11-19
**Repo state at review**: branch `main`, v0.0.5, uncommitted work pushed to 163 commits past `6099d5b`
**Gate baseline at end**: `test-fast` 4823 passed / 19 skipped / 161 deselected / 1 xfailed, coverage ~86% (threshold 77%); `test-heavy` 86 passed / 3 skipped; `test-e2e` 58 passed / 1 skipped; `test-gpu` 32 passed / 4 skipped; `test-doc` 4 passed / 2 skipped; `check`, `docs`, `build` green.

> **Status**: snapshot retrospective. This document records *what was found and how it was found*,
> not current state. Do not edit after-the-fact; write a new audit instead.

---

## 1. Why this document exists

A long iteration series ran ~140 rounds of deliberate defect hunting over this repo. Most
rounds found nothing — that is the expected shape of mature-code auditing, but it makes the
signal hard to find later: 163 commits interleave high-severity fixes with one-line doc
corrections.

This retrospective separates the durable value from the noise:

- **§2** — the defects that mattered, with root cause and how each was detected.
- **§3** — which detection techniques actually worked, and which produced false positives.
- **§4** — a checklist for a future reviewer.
- **§5** — what was audited and found *clean*, so it need not be redone first.

---

## 2. The defects that mattered

Ordered by blast radius. Each entry is a real, user-visible failure — not a hypothetical.

### 2.1 GGUF export produced files llama.cpp could not read

`GGUFTensorInfo.offset` was written as an **absolute** offset into the file, and the reader
subtracted the same base, so our own round trip passed. The reference reader (`gguf-py`, and
therefore llama.cpp) computes `data_start + offset` itself, so **every** exported file was
rejected by the tool users actually load.

- **Detection**: cross-validation against an **external oracle** (`gguf.GGUFReader`). Our own
  reader shared the misconception, so no self-consistent test could have failed.
- **Fix**: offsets are relative to the data section.
- **Regression guard**: `tests/export/test_gguf_roundtrip.py` now reads the F32 export through
  `gguf.GGUFReader` and asserts element equality, plus a test pinning the
  dimensions-innermost-first convention so a future "fix" cannot transpose the output.

### 2.2 Every prefill was non-causal on the `batch_indices` attention path

`has_past` was hard-coded `True` on that branch, so the causal mask was skipped for any
prefill routed through it. Attention attended to future tokens. No error, no warning — the
model simply produced different numbers (measured divergence 0.144).

- **Detection**: cross-mode property comparison (the same input through the two paths).
- **Fix**: derive `has_past` from `start_pos` instead of assuming.
- **Follow-on**: the identical bug existed in `flash_attn.py`. That backend is
  CUDA-optional and untestable here, so it is guarded by a **source-level** test
  (`tests/core/attn/test_causal_flag_derivation.py`) asserting neither backend contains
  `has_past = True`, alongside backend-independent derivation cases.

### 2.3 HF configs from transformers 5.x loaded with a 100× wrong RoPE frequency

`get_config_mapping` read `rope_theta` from the top-level key. transformers 5.x moved the
rotary parameters into `rope_parameters`; `Qwen2Config` no longer has a `rope_theta`
attribute at all. A Qwen2.5 checkpoint (`rope_theta=1e6`) therefore loaded as `1e4` — a 100×
wrong rotary frequency, silently.

- **Detection**: mapping audited against **real upstream config objects**
  (`transformers.LlamaConfig` etc.), not hand-written dicts.
- **Fix**: read nested-then-flat-then-default; both layouts work, so older configs and this
  repo's own published `config.json` are unaffected.

### 2.4 A wheel installed from `dist/` shipped a CLI that could not start

`llm.cli` builds every command on typer, but typer was **never declared** in
`project.dependencies` — it arrived transitively via `fastapi[standard]`, so nothing
constrained it. A clean install resolved typer 0.29.2, which removed the `click_type=`
argument `training/train.py` passes to `typer.Option`, and **every subcommand died at import**
— including `llm --version`.

- **Detection**: ran `just build`, then installed the artefact into a clean venv and used it.
  The dev environment had typer 0.27.3 and worked, which is exactly why no test caught it:
  **nothing ever installed the built artefact.**
- **Fix**: `typer>=0.27.3,<0.29` as a direct dependency.
- **Regression guard**: a test asserting the declaration exists with a ceiling below the
  breaking release, plus one importing `llm.cli.main.app`.

### 2.5 Two ranks bound to the same GPU

`rank_cuda_mapping` ordered GPUs by free memory, so two processes reading slightly different
values produced mirrored orderings and both ranks bound `cuda:1`, colliding in NCCL.

- **Detection**: running the two-rank path repeatedly and observing the collision.
- **Fix**: deterministic mapping.
- **Verification**: 7/7 consecutive successful runs after the fix.

### 2.6 A too-small `vocab_size` aborted the process instead of raising

`model.vocab_size` is a config number, while a `simple` tokenizer built without
`tokenizer_path` takes its vocabulary from `DEFAULT_SIMPLE_CORPUS` — a fixed **103-entry**
printable-ASCII set unrelated to the training corpus. A small model (`vocab_size: 64`)
emitted ids up to 102. On CPU this raised `IndexError: index out of range in self`; **on CUDA
it tripped an assertion inside `vectorized_gather_kernel` and killed the interpreter with
SIGABRT (exit 134)**, printing dozens of `ind >=0 && ind < ind_dim_size` lines and never
mentioning the vocabulary.

- **Detection**: running the real CLI (`llm train`) rather than the in-process engine.
- **Fix**: `_check_tokenizer_fits_model_vocabulary` at the training entry point, next to the
  existing `_check_data_window_fits_model_context`, raising a message that names both numbers,
  explains why 103 regardless of dataset, and gives both remedies.
- **Regression guard**: seven tests, including a **source-level wiring test** — see §3.4.

### 2.7 `data.hash_algo` rejected algorithms its own description promised

The field said *"Any name accepted by `hashlib.new` works"* but validated with
`^[a-z0-9_]+$`, which excludes hyphens. `md5-sha1` **is** in `hashlib.algorithms_available`
and `hashlib.new("md5-sha1")` hashes with it, yet the config refused it.

- **Detection**: comparing the docstring's promise against the actual pattern, then testing
  every real hashlib name against the validator.
- **Fix**: `^[a-z0-9]+([_-][a-z0-9]+)*$` — accepts every `hashlib.algorithms_available` entry
  while still refusing `UPPER`, `-lead`, `trail-`, `has space`, `a--b`, and the empty string.

### 2.8 Smaller correctness fixes worth naming

- `.std()` defaulted to `unbiased=True`, producing **NaN** for a one-element PPO buffer →
  advantage and reward normalisation use `unbiased=False`.
- BLEU/chrF scored only the first sentence (`[[r] for r in references]`) → `[list(references)]`.
- `HFTokenizer.vocab_size` ignored `add_tokens` → `len(tokenizer)`.
- Resume replayed the RNG stream → persist/restore RNG state.
- Epoch summary printed the *next* epoch's LR → capture the LR before `scheduler.step()`.
- `B-Time` was a running average (319.1 vs real 2.6) → the current batch's duration.
- `Grad Norm: 0.00` on 40% of lines → `None` + `n/a` rather than a fake zero.
- Persisted-mode dedup `skip` resurrected duplicates → survivor counting.

---

## 3. Detection techniques: what worked, what lied

### 3.1 External oracles — highest yield

Cross-validating against code the project does not control found the defects that
self-consistent tests structurally cannot. Used successfully with `gguf-py`,
`transformers`, the `openai` SDK, `sacrebleu`, `rouge_score`, and PyTorch reference ops.

**Rule**: if both writer and reader are ours, a green round trip proves only self-consistency.

### 3.2 Mutation testing — highest confidence per change

Break the fix, confirm the test goes red, restore. This repeatedly exposed **fake-passing
tests**:
- DPO `ln(2)` sign-symmetric — passed with the sign flipped.
- SimPO comparing two independently-built models.
- A speculative-decoding end-to-end test forcing a fixed proposal instead of sampling.
- `Literal` closure untested — widening to `str` changed nothing.
- Six tests calling a helper **directly** stayed green when its call site was deleted.

### 3.3 Sibling / mirror-function scanning

When one implementation is fixed, immediately check its twins. Found:
- the same non-causal prefill bug in `flash_attn.py`;
- the same `.std(unbiased=...)` bug in two RLHF modules;
- two regex-validated config fields accepting values the implementation refuses.

### 3.4 Source-level assertions for untestable behaviour

Some code cannot be exercised here (CUDA-only backends, 6-GPU paths). Rather than leave it
unguarded, assert on the **source text**:
- `test_causal_flag_derivation.py` — neither backend contains `has_past = True`.
- the vocabulary guard's **wiring** test — the helper exists *and* is called after
  `data_module.setup()`, because the six behavioural tests could not tell the difference.

### 3.5 Degenerate-input scanning

Single-sample batches, zero-length sequences, empty eval splits, non-finite values. Found the
PPO NaN, the `inf`-to-`null` JSON normalisation gap, and several empty-input crashes.

### 3.6 Techniques that produced false positives

Recorded so a future reviewer does not repeat them. **Every one of these was my error, not the
product's:**

| Probe | Why it lied |
|---|---|
| Reading `EXPORT_REGISTRY._registry` | That attribute does not exist; the real one is `_entries`. |
| Testing `fsdp_mixed_precision='fp16'` on CPU | That path never reaches `_fsdp_mixed_precision`. |
| `parallel_strategy='3d'` alone | Conditionally valid — needs `dp_size`/`pp_size`/`tp_size`. |
| Counting zeros in `weight` for pruning | Pruning keeps `W` intact and applies a mask in forward. |
| Reading a decompose error as "too high" | It equalled the theoretical SVD optimum. |
| Asserting the NF4 table is symmetric | 16 levels include an exact 0.0, so 15 cannot pair. |
| `data/demo.txt` missing | A documented prerequisite, not a broken config. |
| Low coverage in `training/distributed/` | Those tests are `heavy`; `test-fast` excludes them. |

**Pattern**: each failure was *an unverified assumption about where the code path goes*, not a
missing test.

---

## 4. Checklist for a future reviewer

1. **Run every gate, including ones CI does not**: `test-fast`, `test-heavy`, `test-e2e`,
   `test-gpu`, `test-doc`, `check`, `docs`, `build`.
2. **Install the built artefact** and use it. A green suite says nothing about the wheel.
3. **Diff two full runs** to establish determinism before trusting any single failure.
4. **For each fix, mutate it** and confirm the test goes red.
5. **Cross-validate against an external implementation** wherever one exists.
6. **Check twins** of every function you touch.
7. **Re-derive the bound** before calling a number suspicious (SVD optimum, quantisation
   error, expected sparsity).
8. **Distinguish "unreachable but dangerous" from "live defect"** and say which it is.
9. When a test fails, **suspect the test first** — this happened ten-plus times in this series.

---

## 5. Audited and found clean

Recorded so these need not be redone from scratch. Each was verified empirically, not by
reading:

- All 8 gates, on a loaded host.
- Wheel contents: 229 `.py` in the wheel vs 229 in `src/`, 16/16 subpackages, entry points
  registered, core-only install fully functional.
- Real `llm serve` + official `openai` SDK over TCP: chat, streaming, `/health`, and the
  422 refusal of unsupported parameters.
- Real `llm train` end to end.
- `llm migrate-ckpt`, `llm prune`, `llm decompose`, and all six subcommands' `--help`.
- All 15 accepted ADRs have live implementations; both `Proposed` ones are genuinely unbuilt
  (`EXPORT_REGISTRY` has no TensorRT target).
- ROADMAP: 178 completed / 14 open, every open entry honestly labelled.
- All 73 numeric `Field` constraints and all 7 `pattern=` fields.
- Public API: 188 modules import, 8/8 entry points resolve, `llm.__all__` complete.
- 3/3 `Usage::` docstring examples run verbatim.
- Two full `test-fast` runs, 4823 tests, zero outcome differences.

---

## 6. Remaining known gaps (need a decision, not more auditing)

| Item | Status | Blocker |
| --- | --- | --- |
| ZeRO Stage-3 | ADR-015 `Proposed`, no `ZeroStage3` class | Needs an ADR decision |
| RoPE ONNX generation export | Refused with a measured drift reason | Needs an ADR-016 change: thread an explicit position input through `DecoderModel.forward` |
| Three-way inference benchmark | `mha`/`mla` compared | `flash-attn` has no Python 3.14 wheel |
| TensorRT | Prerequisites verified | No cp314 wheel; needs Python ≤3.13 or an NGC container |
| Core ML | ADR-016 records it as feasible | Needs Apple hardware to verify any artefact |
| Docker image (`just image`) | **Unverified** | `docker` is not usable on this host |
