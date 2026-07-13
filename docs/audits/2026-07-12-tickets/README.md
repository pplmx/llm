# 2026-07-12 Audit — Issue Backlog (Tier 1 + Tier 2)

These are the Tier 1 (1–2 week) and Tier 2 (1–2 month) issues derived from
[docs/audits/2026-07-12-technical-due-diligence.md](../2026-07-12-technical-due-diligence.md).

Each ticket is a self-contained markdown file ready to be pasted into a GitHub issue
(with the title on the first `#` line).

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

## Status snapshot (2026-07-12)

- **Tier 1**: 14/14 implemented in commits `23b3018`–`817dd86` (main).
- **Tier 2**: 13/14 already in main (`#3, #4, #5, #11, #14, #15, #16, #17, #18, #19, #20, #21, #22`).
- **Remaining**: ticket 23.
