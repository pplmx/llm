"""Optional periodic in-training evaluation (RIL TASK-329).

Wires the (previously never-instantiated) :class:`EvaluationCallback` into
training: every ``training.eval_interval`` optimizer steps, a lightweight LM
evaluation (perplexity/accuracy over a text corpus, via the metric registry)
runs and its results are surfaced through ``engine.log_metrics`` at rank 0.

Guarded so a rank-0-only eval can never hang a collective-forward training
run: ``tp`` / ``fsdp`` / ``pp`` / ``3d`` use collectives (or stage-scheduled
forwards) inside ``model(...)``, so refusing them at construction (fail
fast) beats a mid-run deadlock. ``ddp`` / ``zero`` and single-process runs
forward locally, so rank 0 evaluating its own copy is safe.
"""

from __future__ import annotations

import logging
from typing import Any

from llm.training.core.callbacks import Callback, EvaluationCallback
from llm.training.core.config import Config as TrainConfig

logger = logging.getLogger(__name__)

#: Parallel strategies whose ``model(...)`` forward is collective /
#: stage-scheduled — a rank-0-only periodic eval would hang on the gather.
_COLLECTIVE_FORWARD_STRATEGIES = frozenset({"fsdp", "tp", "pp", "3d"})


def build_periodic_eval_callback(
    config: TrainConfig,
    *,
    model_max_seq_len: int,
    tokenizer: Any,
    world_size: int,
) -> Callback | None:
    """Build the periodic-eval callback, or ``None`` when disabled.

    Returns ``None`` when ``training.eval_interval == 0`` (the default). When
    enabled it resolves the eval corpus as
    ``data.eval_dataset_path -> data.val_dataset_path -> data.dataset_path``
    and raises a clear ``ValueError`` for a missing corpus or an unsupported
    collective-forward parallel strategy.
    """
    interval = config.training.eval_interval
    if not interval:
        return None

    strategy = config.distributed.parallel_strategy
    if world_size > 1 and strategy in _COLLECTIVE_FORWARD_STRATEGIES:
        raise ValueError(
            f"training.eval_interval={interval} is not supported with "
            f"parallel_strategy={strategy!r}: its model forward is collective "
            "or stage-scheduled, so a rank-0-only periodic eval would hang. "
            "Set training.eval_interval=0 (default) or use world_size=1 / "
            "'ddp' / 'zero'."
        )

    corpus = config.data.eval_dataset_path or config.data.val_dataset_path or config.data.dataset_path
    if not corpus:
        raise ValueError(
            "training.eval_interval > 0 requires a text corpus to evaluate: "
            "set data.eval_dataset_path, data.val_dataset_path, or "
            "data.dataset_path."
        )

    from llm.evaluation.eval_tasks.lm_task import LMTask
    from llm.evaluation.runner import EvaluationRunner

    max_seq = config.training.eval_max_seq_len or model_max_seq_len
    task = LMTask(dataset_path=corpus, tokenizer=tokenizer, max_seq_len=max_seq)
    runner = EvaluationRunner(task, metric_names=config.training.eval_metric_names)

    logger.info(
        "Periodic eval enabled: every %d steps on corpus=%r metrics=%s",
        interval,
        corpus,
        config.training.eval_metric_names or "task default",
    )
    return EvaluationCallback(runner, eval_interval=interval)
