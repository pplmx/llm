# 2026-07-12 Audit — Tier 1 Issue Backlog

These are the Tier 1 (1–2 week) issues derived from
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
    --label "audit-2026-07,v0.0.6-audit-followup"
done
```

> **Note**: create the milestone `v0.0.6-audit-followup` first (via
> `gh api -X POST .../milestones -f title=v0.0.6-audit-followup ...`) and the label
> `audit-2026-07` first (via `gh label create audit-2026-07`). Then add `--milestone
> v0.0.6-audit-followup` to the loop.

### Option B — Manual paste

Open each `.md` file, copy the body (everything after the first `# title` line), and
create an issue in GitHub Issues with the matching title and the listed labels.

## Index

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
