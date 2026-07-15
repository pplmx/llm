# 2026-07-12 Audit — Issue Backlog (Tier 1 + Tier 2 + Tier 3)

These are the Tier 1 (1–2 week), Tier 2 (1–2 month), and Tier 3 (3–6 month)
issues derived from
[docs/audits/2026-07-12-technical-due-diligence.md](../2026-07-12-technical-due-diligence.md).

Tier 1 + Tier 2 tickets are self-contained markdown files ready to be pasted
into a GitHub issue (with the title on the first `#` line). Tier 3 tickets
follow the same format but each item is a foundation slice — full
implementation is multi-week per the audit.

## How to import

### Option A — Using `gh` after authentication

```bash
# 1. Authenticate (one-time)
gh auth login

# 2. From repo root, run this loop:
for f in docs/audits/2026-07-12-tickets/*.md; do
  [ "$f" = "README.md" ] && continue
  # First H1 line becomes the title; rest is the body
  title=$(head -1 "$f" | sed 's/^# //')
  body=$(tail -n +2 "$f")
  gh issue create \
    --title "$title" \
    --body "$body" \
    --label "audit-2026-07"
done
```

> **Note**: create the milestone `v0.0.6-audit-followup` first and the label
> `audit-2026-07` first; then add `--milestone v0.0.6-audit-followup` to the loop.

### Option B — Manual paste

Open each `.md` file, copy the body (everything after the first `# title` line), and
create an issue in GitHub Issues with the matching title and the listed labels.

## Tier 1 — Index (1–2 weeks)

| # | File | Title (short) | Severity |
|---|---|---|---|
| 1 | `01-fix-security-md.md` | Fix SECURITY.md | HIGH |
| 2 | `02-hmac-api-key.md` | hmac.compare_digest for API key | HIGH |
| 3 | `03-drop-bin-from-hf-loader.md` | Drop `.bin` from HF allow_patterns | HIGH |
| 4 | `04-raise-on-paged-attention.md` | Raise on `use_paged_attention=True` | HIGH |
| 5 | `05-attn-use-cache-validator.md` | Add attn_impl × use_cache validator | MEDIUM |
| 6 | `06-remove-reload-true.md` | Remove `reload=True` from `main()` | LOW |
| 7 | `07-refuse-no-auth-public-host.md` | Refuse no-auth public host | MEDIUM |
| 8 | `08-ty-in-ci.md` | Add `ty` to CI lint job | MEDIUM |
| 9 | `09-pip-audit-bandit-ci.md` | Add `pip-audit` + `bandit` to CI | MEDIUM |
| 10 | `10-cap-transformers-torch.md` | Cap `transformers<6` and `torch<3` | HIGH |
| 11 | `11-coverage-fail-under.md` | Add coverage `--fail-under` gate | MEDIUM |
| 12 | `12-log-startup-config.md` | Log model version + config on startup | HIGH |
| 13 | `13-eval-logging-dep-groups.md` | Split heavy deps into groups | MEDIUM |
| 14 | `14-threadlock-batch-engine.md` | Add threading.Lock around `step()` | HIGH |

## Tier 2 — Index (1–2 months)

| # | File | Title (short) | Severity |
|---|---|---|---|
| 15 | `15-structured-api-errors.md` | Structured APIError envelope + request IDs (K) | MEDIUM |
| 16 | `16-split-serving-api.md` | Split `serving/api.py` into focused modules (H) | MEDIUM |
| 17 | `17-custom-loop-callbacks.md` | Custom-loop task callback bridge (F, RLHF) | MEDIUM |
| 18 | `18-kv-cache-prefill-optim.md` | Optimize `KVCache.update_at_indices` for prefill (AM) | MEDIUM |
| 19 | `19-hypothesis-invariant-tests.md` | Hypothesis invariant tests for core invariants (AD) | MEDIUM |
| 20 | `20-mkdocstrings-api-reference.md` | `mkdocstrings` API reference (AJ) | HIGH |
| 21 | `21-docs-github-pages.md` | Deploy docs to GitHub Pages (AI) | MEDIUM |
| 22 | `22-custom-prometheus-metrics.md` | Custom Prometheus metrics for serving (AX) | MEDIUM |
| 23 | `23-async-batch-engine.md` | Make `ContinuousBatchingEngine.step` truly async (M) | HIGH |

## Tier 3 — Index (3–6 months, foundation slices)

Tier 3 items are larger architectural work. The audit lists 9 items
(Tier 3 #1–9); the table below shows the foundation-slice tickets
that have been carved out as in-scope for one focused iteration. The
audit references each item by the letter code from the due-diligence
document (e.g. Finding AO = Flash Attention).

| # | File | Title (short) | Audit ref |
|---|---|---|---|
| 24 | `24-flash-attention-registry.md` | Flash Attention 2 via ATTENTION_REGISTRY | Finding AO |
| 25 | `25-hf-hub-publish.md` | HF Hub publish pipeline (`save_pretrained` + `push_to_hub`) | Tier 3 #7 |
| 26 | `26-speculative-decoding.md` | Speculative decoding (Leviathan 2023) for serving | Tier 3 #9 |
| 27 | `27-lm-eval-pipeline.md` | lm-eval-harness pipeline (LlamaLmEvalLM + presets) | Tier 3 #6 |
| 28 | `28-data-presets.md` | Data preset files (C4 / Pile / RedPajama) | Tier 3 #8 |
| 29 | `29-fsdp-e2e-docs.md` | FSDP end-to-end wiring + documentation | Tier 3 #2 |
| 30 | `30-paged-attention-forward.md` | Paged Attention full forward path (replace `list[KVCache]`) | Tier 3 #3 |
| 31 | `31-mla-kv-cache.md` | MLA + linear + paged KV cache wiring | Tier 3 #10 |
| 32 | `32-export-registry.md` | Export registry (`EXPORT_REGISTRY`, parity with `BACKEND_REGISTRY`) | Finding BH |
| 33 | `33-torchscript-export.md` | TorchScript export target via `llm.export_backends` entry point | Tier 3 #11 |
| 34 | `34-export-discovery.md` | Export discovery doc sync (architecture tree, ADR-005, ROADMAP checkboxes) | Tier 3 #32/#33 follow-up |
| 35 | `35-frequency-penalty.md` | OpenAI-compat `frequency_penalty` end-to-end (sampling helper + every backend + chat router) | OpenAI-compat gap |
| 37 | `37-presence-penalty.md` | OpenAI-compat `presence_penalty` with flat-per-token semantics (drop the `1.0 + presence_penalty` alias) | OpenAI-compat gap |
| 38 | `38-logit-bias.md` | OpenAI-compat `logit_bias` (additive per-token biases via `index_add_`, applied after the penalty helpers) | OpenAI-compat gap |
| 39 | `39-data-dedup-source.md` | `DedupTextSource` wrapper (exact-dedup via content hash, optional cross-run seen-hash persistence) — `data_source="dedup_local"` / `"dedup_hf"` compose with the existing builders; `DataConfig` gains `seen_hashes_path` / `write_seen_hashes` / `hash_algo` | P0 pretraining productization |
| 40 | `40-adalora-foundation.md` | AdaLoRA foundation slice: SVD-form `ΔW = P · diag(λ) · Q` parameterization with QR-based orthonormalization on every forward; `mask` buffer registered for the future pruning slice; `orth_reg_loss()`; merge/unmerge; helper surface mirrors `apply_lora` / `merge_lora` / etc. by name. Adaptive pruning + importance scoring is the explicit follow-up (T3 #41). | P2 efficient fine-tuning |
| 41 | `41-adalora-pruning.md` | AdaLoRA pruning slice: `compute_importance_scores` (default `|λ|`, combined `|λ|·|∇L/∂λ|` with gradient EMA), `prune_to_rank` mutates the mask buffer (idempotent, raises on un-pruning), `update_budget` returns a linearly interpolated rank from `init_rank` at `tinit` to `target_rank` at `tfinal`; module-level `prune_adalora` walks every `AdaLoRALinear` and accepts either `target_rank` or `schedule=(tinit, tfinal) + current_step`. Trainer-side EMA smoothing and SFT/DPO integration are deliberate follow-ups. | P2 efficient fine-tuning |

Remaining Tier 3 items (self-hosted GPU runner, MLA + KV cache) are
tracked in [ROADMAP.md](../../ROADMAP.md) §阶段十/十四.

## Status snapshot (2026-07-15)

- **Tier 1**: 14/14 implemented in commits `23b3018`–`817dd86` (main).
- **Tier 2**: 14/14 already in main (`#3, #4, #5, #11, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23`).
- **Tier 3**: 9/9 audit foundation slices + 9 post-audit slices shipped — `#24` (Flash Attention 2 registry entry), `#25` (HF Hub publish), `#26` (Speculative decoding), `#27` (lm-eval-harness pipeline), `#28` (data presets for C4 / Pile / RedPajama), `#29` (FSDP end-to-end wiring + docs), `#30` (Paged Attention full forward path through DecoderModel + ContinuousBatchingEngine), `#31` (MLA + KV cache: linear + paged, with `set_attention_kv_cache_capability("mla", supports=True)`; placeholder-architecture caveat documented), `#32` (Export registry: `EXPORT_REGISTRY` mirrors `BACKEND_REGISTRY`; built-in `onnx` target + `llm.export_backends` entry-point group; `export_to_onnx` preserved as the stable ONNX API), `#33` (TorchScript export target registered through the `llm.export_backends` entry point — first non-built-in backend to exercise the registry's plug-in path end-to-end; trace method supported, script method xfail-tracked for `DecoderModel`), `#34` (Export discovery doc sync: `architecture.md` tree + plugin-kernel table; new ADR-005; ADR README + ROADMAP checkboxes synced to current state), `#35` (OpenAI-compat `frequency_penalty` wired end-to-end: new `apply_frequency_penalty` sampling helper, `GenerationConfig` + every backend (Eager / Batched / Speculative), chat + generate routers, and schemas — the `(not implemented)` hint in the schema drops), `#37` (OpenAI-compat `presence_penalty` with flat-per-token semantics: new `apply_presence_penalty` helper, full plumbing through `GenerationConfig` + every backend + the chat router, `ChatCompletionRequest` widened to `[-2.0, 2.0]`, and the `1.0 + presence_penalty` alias hack at `routers/chat.py:81` removed), `#38` (OpenAI-compat `logit_bias` wired end-to-end: new `apply_logit_bias` helper using `index_add_`, applied after the penalty helpers per OpenAI's reference ordering; string-key coercion handled at the helper boundary; full plumbing through `GenerationConfig` + every backend + chat / generate / batch_generate routers + schemas), `#39` (`DedupTextSource` wrapper: hash-based exact-dedup over any inner `TextSource`; case-sensitive by default; optional cross-run `seen_hashes_path` load + append-only `write_seen_hashes=True` for monotonic dedup state; `source_fingerprint()` includes inner + dedup config so checkpoint-resume validation still catches drift; two new `SOURCE_REGISTRY` entries `dedup_local` / `dedup_hf` compose the wrapper with the existing `local` / `hf` builders; `DataConfig.data_source` regex widens to `^(local|hf|dedup_local|dedup_hf)$` and gains three optional fields `seen_hashes_path` / `write_seen_hashes` / `hash_algo`), `#40` (AdaLoRA foundation slice: `AdaLoRALinear` class with SVD-form parameterization `ΔW = P · diag(λ · mask) · Q` and QR-based orthonormalization on every forward; `mask: (init_rank,)` registered as a buffer so the future pruning slice (T3 #41) can zero out low-importance components without a public-API break; `orth_reg_loss()` exposes `||PᵀP − I||²_F + ||QQᵀ − I||²_F`; init defaults match the paper (`init_rank=12`, `target_rank=init_rank // 2`, `alpha=32.0`); helper surface (`apply_adalora` / `merge_adalora` / `unmerge_adalora` / `get_adalora_parameters` / `count_adalora_parameters` / `disable_adalora` / `enable_adalora`) mirrors LoRA by name for one-import swap; adaptive pruning + importance scoring explicitly deferred to T3 #41), `#41` (AdaLoRA pruning slice: `compute_importance_scores(gradient_ema=None)` returns the paper's per-component importance — default `|λ|`, combined `|λ| · |∂L/∂λ|` when a gradient EMA is supplied; `prune_to_rank(target_rank, scores=None)` mutates the mask to keep the top-k components (idempotent, raises on un-pruning); `update_budget(current_step, tinit, tfinal)` returns a linearly interpolated rank from `init_rank` at `tinit` to `target_rank` at `tfinal`; module-level `prune_adalora(model, target_rank=None, schedule=(tinit, tfinal), current_step=None, gradient_emas=None)` walks every `AdaLoRALinear` and applies the prune call — accepts either a uniform `target_rank` or a `(tinit, tfinal)` schedule; trainer-side EMA smoothing and SFT/DPO task integration are deliberate follow-ups).
