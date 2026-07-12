# Custom-loop task callback bridge (Finding F, RLHF correctness)

## Source
docs/audits/2026-07-12-technical-due-diligence.md §Finding F (MEDIUM),
Tier 2 #7

## Description
`TrainingTask` supports a "custom loop" mode (`uses_standard_loop = False`)
for PPO/RLHF. `PPOTask.run_training` reimplements the loop and currently
emits **no callback hooks** — `MetricsLogger`, `TensorBoardLogger`,
`EarlyStopping`, `EvaluationCallback`, `LRSchedulerCallback` are all
silently skipped for RLHF. This makes RLHF invisible to standard
observability tooling and means `should_stop_training` cannot be honored.

## Acceptance criteria
- [ ] New helper `BaseTask.run_with_callbacks(engine, loop_fn)` in
      `src/llm/training/tasks/base_task.py` that:
      1. Calls `engine._run_callbacks("on_train_start")`
      2. For each epoch: `on_epoch_start` → loop_fn(epoch) →
         `on_epoch_end`
      3. Calls `engine._run_callbacks("on_train_end")`
      4. Wraps the loop in try/except that calls
         `engine._run_callbacks("on_exception", exc)` then re-raises
- [ ] Inside the epoch, `loop_fn` is expected to call
      `task._emit_step_callbacks(...)` after each optimizer step
      (so `on_train_step_end` fires with `(loss, metrics)`).
- [ ] `PPOTask.run_training` refactored to call
      `self.run_with_callbacks(engine, loop_fn)`; `loop_fn` calls
      `_emit_step_callbacks` after each `ppo_trainer.train_step(prompts)`.
- [ ] Verify `should_stop_training` is honored: setting it from a
      callback causes RLHF training to exit cleanly.
- [ ] Tests:
      - `tests/training/rlhf/test_ppo_callbacks.py` — fake PPOTrainer,
        assert `MetricsLogger.on_epoch_end` fires with correct `logs`.
      - `tests/training/rlhf/test_ppo_stop.py` — assert a callback that
        sets `should_stop_training` halts the loop after the current
        epoch.

## Estimate
~2 hours

## Labels
`audit-2026-07`, `v0.0.6-audit-followup`, `rlhf`, `training`,
`correctness`, `observability`
