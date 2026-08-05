---
name: add-task
description: >
  Workflow for adding a new training task to this repo (Python LLM training
  framework): TrainingTask subclass + paired DataModule + TASK_REGISTRY
  registration + optional streaming TextSource. Use when the user wants a new
  training objective, a new DataModule, a new streaming data source, or asks
  "how do I add a task for llm-train --task" (新训练任务 / 新数据模块 /
  新流式数据源).
---

# Add a Training Task

`llm-train --task <name>` dispatches through `TASK_REGISTRY`
(`src/llm/training/task_registry.py`), which binds a `TrainingTask` subclass
to its `BaseDataModule` factory as a frozen `TaskSpec`. Built-ins live in
`src/llm/training/tasks/builtin.py`: `regression`, `lm`, `stream_lm`, `sft`,
`dpo`, `reward`, `ppo`.

## Workflow

1. **DataModule first** (`src/llm/data/modules/<name>.py`): subclass
   `BaseDataModule` (`src/llm/data/base.py`) and implement:
    - `prepare_data()`, `setup(stage)`
    - `train_dataloader(rank, world_size) -> (DataLoader, sampler|None)`
    - `val_dataloader(rank, world_size) -> (DataLoader|None, sampler|None)`
    - streaming tasks must also implement `validate_streaming_config()`,
   `get_checkpoint_state()`, `load_checkpoint_state()` (checkpoint-resume
   contract; see `StreamingTextDataModule` as the reference).
   Map-style datasets live in `src/llm/data/datasets/`.

2. **Task class** (`src/llm/training/tasks/<name>_task.py`): subclass
   `TrainingTask` (`tasks/base_task.py`) and override what differs:
    - `build_model()` (route construction through `ModelFactory` / registry
   builders; apply PEFT wrapping here when opt-in config flags are set —
   see `LanguageModelingTask` for the pattern)
    - `build_optimizer(model)`, `build_scheduler(optimizer)`,
   `build_criterion()`
    - `train_step(batch, model, criterion) -> (loss, metrics_dict)` and
   `validation_step(...)`
    - `build_callbacks()` for task-specific callbacks (e.g. AdaLoRA pruning,
   PEFT adapter save)
    - custom loops: `uses_standard_training_loop()` returning `False` plus
   `run_training(engine)` (see PPO); prefer the standard loop when possible
    - extra resume state: `get_checkpoint_state()` / `load_checkpoint_state()`
   (`CheckpointContributor` protocol)

3. **Register** in `src/llm/training/tasks/builtin.py`:

   ```python
   TASK_REGISTRY.register("my_task", MyTask, MyDataModule, description="...")
   ```

   Third-party tasks use the `llm.tasks` entry-point hook group loaded by
   `train.py` (`load_entry_point_hooks("llm.tasks")`). The CLI `--task`
   choices are generated from `TASK_REGISTRY.names()` automatically.

4. **New streaming text sources** (when the task consumes text corpora):
   subclass `TextSource` in `src/llm/data/sources.py`
   (`iter_texts(skip=0)` + `source_fingerprint()`), add a
   `_build_<name>_source(data_config)` builder, register it in
   `ensure_sources_registered()`, widen the `DataConfig.data_source` regex,
   and expose config fields on `DataConfig`. Entry-point group:
   `llm.data_sources`. Reference: `DedupTextSource` wrapper + `dedup_local` /
   `dedup_hf`.

5. **Config**: add opt-in fields to `TrainingConfig` / `DataConfig`
   (`src/llm/training/core/config.py`) with defaults that preserve current
   behavior; add Pydantic/dataclass validators for bad combinations.

6. **Tests** (see `write-test` skill): `tests/training/` for the task
   (forward/backward step, checkpoint round-trip when the task carries extra
   state), `tests/data/` for the DataModule/source. Add at least one e2e test
   exercising `llm-train --task <name>` end-to-end on tiny fixtures.

## Pitfalls

- A task and its DataModule are registered **together** — never register a
  task without its data module factory.
- Streaming resume: `source_fingerprint()` must include everything that makes
  the stream non-interchangeable, or checkpoint resume validation silently
  accepts drifted sources.
- Next-token LM losses must apply the shift inside the task's loss path (see
  commit d8306cb) — do not double-shift in both dataset and loss.
