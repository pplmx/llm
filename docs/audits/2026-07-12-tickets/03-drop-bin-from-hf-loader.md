# Drop `*.bin` from HF `from_pretrained` allow_patterns (Finding AR)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AR (HIGH)

## Description
`src/llm/compat/hf_loader.py:152-156` downloads HF Hub models with
`allow_patterns=["*.json", "*.safetensors", "*.bin"]`. PyTorch `.bin` files are pickled
and can execute arbitrary code on load. Most modern HF models ship safetensors; `.bin`
is legacy. Drop `.bin` from the allow list to eliminate the attack surface.

## Acceptance criteria
- [ ] `_load_from_hub` only downloads `*.json` and `*.safetensors`
- [ ] `list_supported_architectures()` document string updated to reflect safetensors-only
- [ ] Add test that verifies a fake HF repo with only `.bin` files fails gracefully
      (clear error, not a pickle-load crash)

## Estimate
~10 minutes

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `security`
