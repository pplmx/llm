"""Supervised Fine-Tuning (SFT) task — an alias of :class:`LanguageModelingTask`.

SFT is next-token language modeling on dictionary batches
(``input_ids`` / ``labels`` / ``attention_mask``), which
:class:`llm.training.tasks.lm_task.LanguageModelingTask` already handles:
its ``train_step``/``validation_step`` accept dict batches, shift the
next-token prediction, and raise on degenerate (length-1) sequences.

This class exists only so the registry can name the SFT entry point
(``TASK_REGISTRY.register("sft", SFTTask, ...)``) and so
``SFTTask`` stays a stable import for users and tests. It is deliberately
a **pure alias** — no overrides.

RIL ISS-339: the previous ``train_step``/``validation_step``/``build_criterion``
copies silently diverged from the parent:

- they threaded the batch ``attention_mask`` into ``model(input_ids,
  attn_mask=...)``, but the flash_attn backend explicitly ignores
  ``attn_mask`` (``llm/models/decoder.py``) and the pipeline-parallel
  scheduler drops it (``engine._pp_extract_inputs`` only binds
  ``input_ids``/``labels``), so SFT behavior differed by attention backend
  and by parallel strategy;
- they lacked the parent's ``numel() == 0`` guard, so a length-1/empty row
  produced NaN and was silently swallowed as a 0.0 loss instead of raising.

Delegating every step to the parent makes SFT behave identically under SDPA,
flash_attn, and pipeline parallelism, and inherit the loud empty-sequence
guard.
"""

from __future__ import annotations

from llm.training.tasks.lm_task import LanguageModelingTask


class SFTTask(LanguageModelingTask):
    """Supervised fine-tuning: causal LM on ``SFTDataModule`` batches.

    All behavior (build_model/model wrapping, optimizer/scheduler, loss,
    train/validation steps, pipeline-parallel support) is inherited from
    :class:`~llm.training.tasks.lm_task.LanguageModelingTask`.
    """
