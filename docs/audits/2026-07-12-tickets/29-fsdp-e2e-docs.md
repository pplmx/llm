# FSDP end-to-end wiring + documentation (Tier 3 #2)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Tier 3 #2
("FSDP end-to-end + docs"), ROADMAP §5 高级分布式训练:
"FSDP — ``parallel_strategy=fsdp`` + ``wrap_model_for_training()`` 已接线,
待 e2e 与文档"

## Description
The framework already accepts ``parallel_strategy="fsdp"`` via
``DistributedConfig`` and ``wrap_model_for_training()`` constructs an
:class:`torch.distributed.fsdp.FullyShardedDataParallel` wrapper, but
the integration is leaky:

* The FSDP wrapper is created with **no config** — no
  ``mixed_precision``, no ``auto_wrap_policy``, no ``cpu_offload``.
  A user who picks FSDP gets a near-default wrapper that won't
  actually save memory (because every parameter ends up in the
  default FSDP unit).
* ``model_state_dict`` / ``load_model_state_dict`` always use
  ``FULL_STATE_DICT`` for FSDP, which materializes the entire
  model on rank 0 — defeating the memory benefit of FSDP when
  saving / loading checkpoints.
* ``docs/guides/distributed.md`` explicitly says FSDP is "即将支持"
  (coming soon), which contradicts the actual ``parallel_strategy="fsdp"``
  support.

This ticket closes the gap as a foundation slice:

1. Add FSDP config fields to :class:`DistributedConfig`:
   - ``fsdp_mixed_precision: Literal["fp32", "bf16", "fp16"]`` (default ``"bf16"``)
   - ``fsdp_auto_wrap_min_params: int`` (default ``10_000_000`` — only wrap
     modules larger than this)
   - ``fsdp_cpu_offload: bool`` (default ``False``)
2. Wire those into ``wrap_model_for_training`` so the FSDP wrapper
   applies the config (size-based ``auto_wrap_policy``,
   ``MixedPrecision``, optional ``CPUOffload``).
3. Add an enum value for the state-dict strategy so
   ``model_state_dict`` / ``load_model_state_dict`` can pick
   between ``FULL_STATE_DICT`` (single-host save/load) and
   ``SHARDED_STATE_DICT`` (memory-efficient multi-rank save/load).
4. Update :file:`docs/guides/distributed.md` to document FSDP
   properly: when to use it vs DDP, the config knobs,
   memory/speed tradeoffs, and a runnable example.
5. Tests: config validation (``fsdp_mixed_precision`` values,
   min-params non-negative), ``wrap_model_for_training`` returns
   ``None`` gracefully when ``fsdp`` is requested on CPU (the
   actual FSDP init needs CUDA + a process group, so we keep the
   test to the dispatch logic, not the real FSDP runtime).

## Acceptance criteria
- [ ] :class:`DistributedConfig` accepts ``fsdp_mixed_precision``,
      ``fsdp_auto_wrap_min_params``, ``fsdp_cpu_offload`` with
      reasonable defaults.
- [ ] ``wrap_model_for_training`` builds an FSDP wrapper that uses
      the size-based auto-wrap policy and (optionally) mixed precision
      and CPU offload per the config.
- [ ] ``model_state_dict`` / ``load_model_state_dict`` accept a
      ``state_dict_type`` argument (``"full"`` or ``"sharded"``).
- [ ] Tests in ``tests/training/distributed/test_parallel.py``:
      config validation + ``wrap_model_for_training`` dispatch
      (cpu/cuda single-rank paths).
- [ ] :file:`docs/guides/distributed.md` rewritten to remove the
      "coming soon" note and document FSDP as a first-class strategy.

## Estimate
~2 weeks

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `distributed`, `fsdp`,
`docs`
