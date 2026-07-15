# Add AdaLoRA pruning + importance scoring (T3 #41)

## Source

Direct follow-up to [T3 #40](40-adalora-foundation.md) — the
foundation slice shipped `AdaLoRALinear` with SVD-form parameterization,
QR-based orthonormalization, an `effective_rank` property, and a
`mask` buffer explicitly reserved for this slice. ROADMAP §4 (P2) and
§15.3 already list AdaLoRA as done; the **adaptive** behaviour
(periodic reallocation of the rank budget across modules) is what this
ticket closes.

## Description

The T3 #40 layer is functionally a higher-rank LoRA with the
orthogonality regularization. It only becomes *adaptive* once
importance scoring + periodic pruning are wired in — that is, once the
rank budget can be reallocated from low-importance components to
high-importance ones during training.

Per Algorithm 1 in Zhang et al. 2023, the paper's recipe is:

```text
I_i = |λ_i| · |∂L/∂λ_i|           # per-component importance
I_avg_i = α · I_avg_i + (1−α) · I_i   # EMA smoothing (trainer-side)
b_t = linear(init_rank, target_rank, t)  # budget schedule
mask = top-k(I_avg, k=b_t)          # keep the top-b_t components
```

This slice ships the **layer-side** half of that recipe (importance
score, prune-to-rank, budget schedule, module-level helper). The
**trainer-side** half (the EMA smoothing loop, the periodic timer,
the loss fan-out) stays outside this ticket — it's how the existing
SFT / DPO tasks opt in, and that integration is its own follow-up.

Concrete API additions on top of T3 #40:

```python
# Per-layer
scores = layer.compute_importance_scores(gradient_ema=None)  # (init_rank,)
layer.prune_to_rank(target_rank)                            # mutates mask
scheduled = layer.update_budget(current_step, tinit, tfinal) # int

# Module-level
from llm.core.adalora import prune_adalora

# Same target rank for every AdaLoRALinear layer
prune_adalora(model, target_rank=8)

# Per-step budget schedule (linear from init_rank to target_rank)
prune_adalora(model, schedule=(tinit, tfinal), current_step=step)
```

The signature mirrors the LoRA-style "apply to whole model" helpers
(`apply_lora`, `merge_lora`, etc.) so swapping LoRA → AdaLoRA in user
code keeps the same ergonomics.

## Acceptance criteria

- [ ] `AdaLoRALinear.compute_importance_scores(gradient_ema=None)`
      returns a tensor of shape ``(init_rank,)``:
        - default (``gradient_ema=None``): ``|λ_i|``;
        - with ``gradient_ema`` provided: ``|λ_i| · gradient_ema_i``
          (element-wise product), per the paper's combined score.
- [ ] `AdaLoRALinear.prune_to_rank(target_rank)` mutates ``self.mask``
      in-place so that exactly ``target_rank`` entries remain
      ``1.0`` and the rest are ``0.0``. The kept entries are the
      ``target_rank`` components with **highest** importance score.
      Idempotent: calling it twice with the same target is a no-op
      after the first call.
- [ ] `prune_to_rank` raises ``ValueError`` if
      ``target_rank > self.effective_rank`` (you cannot un-prune —
      the dropped components would need their λ values restored,
      which is out of scope for the algorithm).
- [ ] `prune_to_rank(target_rank=init_rank)` is a no-op (every
      component kept).
- [ ] `AdaLoRALinear.update_budget(current_step, tinit, tfinal)`
      returns ``init_rank`` when ``current_step ≤ tinit``,
      ``target_rank`` when ``current_step ≥ tfinal``, and a
      linearly interpolated integer rank otherwise.
- [ ] `update_budget` raises ``ValueError`` if
      ``tinit >= tfinal`` or ``current_step < 0``.
- [ ] Module-level ``prune_adalora(model, target_rank=None,
      schedule=None, current_step=None)`` walks every
      ``AdaLoRALinear`` layer and calls ``prune_to_rank`` on it.
      Either ``target_rank`` or ``schedule=(tinit, tfinal)`` (with
      ``current_step``) must be provided; providing neither raises.
- [ ] `effective_rank` reflects pruning immediately (already true in
      T3 #40; the slice just exercises it under realistic schedules).
- [ ] Tests in `tests/core/test_adalora.py` (new section): the eight
      acceptance items above, plus a small integration test that
      trains λ briefly, prunes, and verifies that the kept components
      are the ones whose magnitude grew most.
- [ ] `CHANGELOG.md` `[Unreleased] ### Added` gets an entry
      cross-referencing T3 #41.
- [ ] `ROADMAP.md` §4 + §15.3 AdaLoRA note is updated from
      "foundation slice, pruning is follow-up" to "foundation +
      pruning slices shipped".
- [ ] `docs/audits/2026-07-12-tickets/README.md` adds ticket #41 to
      the index and the status snapshot.

## Non-goals (deliberately deferred)

- **Trainer-side EMA smoothing** — the running average of
  ``|∂L/∂λ_i|`` is computed in the training loop, not in the layer.
  This slice provides the *contract* (``gradient_ema`` argument) so
  the trainer can plug in any smoothing strategy it likes (PyTorch
  optimizer hook, manual loop, etc.).
- **SFT / DPO task integration** — wiring ``step_adalora_pruning`` into
  the existing SFT/DPO callbacks is a separate ticket (T3 #42 in
  this iteration's plan). The slice today is the *layer-side* half
  only.
- **Per-module rank allocation** — the paper allocates different rank
  budgets to different modules based on their importance distribution.
  The current slice applies the same ``target_rank`` to every layer;
  per-module allocation is a meaningful follow-up but doubles the
  API surface and can ship later.
- **Un-pruning** — once a component is masked out, restoring it
  requires un-zeroing its λ entry, which has been overwritten by
  the optimizer. The slice accepts that pruning is one-way and
  raises on illegal ``target_rank > effective_rank``.

## Estimate

~1 focused iteration. The layer-side additions are a few dozen lines;
the trainer-side wiring (next slice) is the larger piece.

## References

- Zhang et al., 2023 — *Adaptive Budget Allocation for Parameter-
  Efficient Fine-Tuning*, arXiv:2303.10512. Algorithm 1 (page 4).

## Labels

`audit-2026-07`, `v0.0.6-audit-followup`, `core`, `p2-finetuning`
