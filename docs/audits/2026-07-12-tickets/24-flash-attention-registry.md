# Flash Attention 2 integration via registry (Finding AO)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding AO (MEDIUM),
Tier 3 #4

## Description
PyTorch's SDPA backend already auto-selects Flash Attention 2 on
supported hardware (Hopper/Ampere), but the project should expose
Flash Attention 2 explicitly through the `ATTENTION_REGISTRY` so users
can opt in, gain access to sliding-window patterns, and unlock longer
sequences. This ticket establishes the **registry entry** and the
optional-dependency contract; full `flash_attn_varlen_func` /
ring-attention integration is a follow-up.

The existing design spec at
`docs/superpowers/specs/2026-03-26-flash-attention-design.md` covers
the long-term backend abstraction; this ticket is the **first
shippable slice**: register `FlashAttention` as an opt-in
`attn_impl` value, mirroring how `mha` and `mla` are registered.

## Acceptance criteria
- [ ] New `src/llm/core/attn/flash_attn.py` exposing
      `class FlashAttention(nn.Module)` registered via
      `@register_attention("flash_attn")`.
- [ ] `flash_attn` is an **optional dependency**: importing
      `llm.core.attn.flash_attn` does NOT raise when `flash-attn`
      is not installed; only `FlashAttention()` instantiation raises
      a clear `ImportError` with the install command.
- [ ] `set_attention_kv_cache_capability("flash_attn", supports=True)`
      matches `mha`'s contract (KV cache integration same shape).
- [ ] `src/llm/core/attn/__init__.py` exports `FlashAttention` so
      `from llm.core.attn import FlashAttention` works.
- [ ] New `tests/core/test_flash_attn_registry.py` covering:
      - Registry contains "flash_attn"
      - `FlashAttention` class is exported
      - Capability map entry exists with value True
      - Instantiating without `flash-attn` installed raises
        `ImportError` with the remediation hint.
- [ ] `pyproject.toml` adds `flash-attn` to an optional dependency
      group (e.g. `[perf]` or `[attn]`) — NOT the runtime
      dependencies list.
- [ ] Doc: short section in `docs/guides/inference.md` describing
      how to opt in (`attn_impl="flash_attn"`) and the install
      command.

## Estimate
~1 week (foundation slice only; full backend abstraction is a
follow-up).

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `perf`, `attn`,
`correctness`, `optional-dep`
