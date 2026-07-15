# Add AdaLoRA foundation slice (T3 #40)

## Source

`ROADMAP.md` §4 (高效微调, P2) and §15.3 (efficient fine-tuning
techniques) — both list **AdaLoRA (Adaptive LoRA)** as unchecked.
Builds directly on the existing `LoRALinear` module
(`src/llm/core/lora.py`) and its helper surface.

## Description

The project ships LoRA + QLoRA (`src/llm/core/lora.py`,
`src/llm/core/qlora.py`) for parameter-efficient fine-tuning. Both use
a **fixed rank**: every adapted linear layer carries the same number of
trainable low-rank components regardless of how important that matrix
is to the task. AdaLoRA (Zhang et al., 2023 — *Adaptive Budget
Allocation for Parameter-Efficient Fine-Tuning*, arXiv:2303.10512)
addresses this by parameterizing the increment matrix in **SVD form**
(``ΔW = P · diag(λ) · Q`` with orthogonal P, Q) and re-allocating the
rank budget across modules based on per-component importance.

The full algorithm has two distinct concerns:

1. **SVD-form parameterization + orthogonality regularization** (this
   slice)
2. **Importance scoring + periodic pruning** (follow-up slice)

This ticket carves out the foundation slice. It is enough to:

- ship `AdaLoRALinear` — a drop-in alternative to `LoRALinear` that
  carries the SVD-form parameterization,
- expose the **mask hook** so the future pruning slice can zero out
  low-importance components without a public-API break,
- compute the **orthogonality regularization** (`||PᵀP − I||²_F +
  ||QQᵀ − I||²_F`) so trainers can add it to the loss with one call,
- ship the usual merge / unmerge / get-parameters helpers so users
  get the same ergonomics as LoRA.

A minimal but realistic reference (matching Algorithm 1 in the paper):

```text
P, λ, Q  ← init P via random orthogonal, λ via small random, Q via zeros
forward(x):
    P̃ ← orthonormalize(P)              # QR decomposition
    Q̃ ← orthonormalize(Q.T).T          # QR on the transpose
    ΔW ← P̃ · diag(λ · mask) · Q̃
    return base(x) + scaling · (x · ΔWᵀ)
orth_reg_loss():
    return ||P̃ᵀ P̃ − I||²_F + ||Q̃ Q̃ᵀ − I||²_F
```

Why SVD-form vs the `A @ B` form used by LoRA: the diagonal λ is the
**spectrum** of the increment. The mask slots into the spectrum, so
pruning "zeroes out a component" maps cleanly to setting one `λ_i` to
zero — instead of the rank-rearrangement surgery that
pruning-trimming `A[:, kept]` would force.

## Acceptance criteria

- [ ] New module `src/llm/core/adalora.py` with:
    - `class AdaLoRALinear(nn.Module)` — SVD-form parameterization
      (`P: (out, init_rank)`, `lambda: (init_rank,)`, `Q: (init_rank,
      in)`); mask `mask: (init_rank,)` initialised to all-ones;
      QR-based orthonormalization inside `forward`; `merge_weights()`
      and `unmerge_weights()` so the layer can be folded back into the
      base for inference; `orth_reg_loss()` returning a non-negative
      scalar; `effective_rank` property reading through the mask.
    - `apply_adalora(model, init_rank, target_rank, alpha,
      target_modules=None)` — mirrors `apply_lora`'s signature;
      `target_rank` is stored on each layer but does **not** trigger
      pruning in this slice (the follow-up slice owns pruning).
    - `merge_adalora`, `unmerge_adalora`, `get_adalora_parameters`,
      `count_adalora_parameters` — mirror the LoRA helpers by name so
      swapping LoRA → AdaLoRA in user code is a one-import change.
- [ ] Init defaults match the paper's worked example:
    `init_rank=12`, `target_rank=init_rank // 2`, `alpha=32.0`. (`rank`
    is renamed to `init_rank` everywhere; LoRA's `rank` keeps its
    name — AdaLoRA's "rank" is a *budget*, not the count of active
    components.)
- [ ] `forward(x)` matches the base layer's output **at initialization**
    (λ is initialised small enough that the masked ΔW contribution is
    numerically zero to ≤1e-6 tolerance).
- [ ] Base layer weight and bias are frozen, matching `LoRALinear`.
- [ ] `orth_reg_loss()` is **zero** when P and Q are themselves
    orthonormal — verified by a unit test that orthonormalizes P/Q
    manually before calling.
- [ ] `merge_weights` followed by `unmerge_weights` returns the base
    weight to its pre-merge value (roundtrip identity) — same
    invariant as `LoRALinear`.
- [ ] `apply_adalora` with `target_modules=[...]` replaces only
    matching `nn.Linear` layers (verified by counting the layers
    before/after).
- [ ] Tests in `tests/core/test_adalora.py`: init shape, base frozen,
    trainable params, forward shape, initial-output matches base, λ
    non-zero produces different output, mask pruning zeroes
    contribution, QR orthonormality holds, orth_reg_loss positive
    when P/Q perturbed, merge/unmerge roundtrip, apply_adalora
    targeting, count_adalora_parameters accounting.
- [ ] `CHANGELOG.md` `[Unreleased] ### Added` gets an entry
    cross-referencing T3 #40.
- [ ] `ROADMAP.md` §4 and §15.3 AdaLoRA boxes are checked with a
    one-line note about the follow-up pruning slice.
- [ ] `docs/audits/2026-07-12-tickets/README.md` adds ticket #40 to
    the index and the status snapshot.

## Non-goals (deliberately deferred)

- **Importance scoring + periodic pruning** — separate ticket (T3 #41)
  on top of this foundation. The `mask` attribute and
  `effective_rank` property are designed so that slice is an additive
  change with no public-API break.
- **Integration with the SFT / DPO training tasks** — T3 #42 (or
  later). The current SFT task already accepts a custom parameters
  callback for the optimizer, so wiring AdaLoRA in is a one-liner
  once the foundation + pruning ship.
- **AdaLoRA-specific initialisation schedule** (orthogonal P, small
  λ, zero Q) is implemented here because it's part of the layer
  contract. Anything beyond that (β₁/β₂ EMA on importance, rank
  warming schedule, etc.) belongs to T3 #41.

## Estimate

~1 focused iteration. Slightly larger than the LoRA module because
of QR decomposition + orthogonality regularization, but bounded — no
external dependencies, no training-loop integration.

## References

- Zhang et al., 2023 — *Adaptive Budget Allocation for Parameter-
  Efficient Fine-Tuning*, arXiv:2303.10512.

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `core`, `p2-finetuning`
