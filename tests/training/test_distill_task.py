"""Knowledge-Distillation task tests (RIL TASK-222/223, DEC-055).

Covers three layers of the ``distill`` training path:

1. Registration + the fresh-seeded dev-convenience teacher (student trainable,
   teacher frozen + different init).
2. Loading a real frozen teacher from a CheckpointManager checkpoint.
3. A full (CPU, ``slow``) e2e: pre-train a teacher on a small deterministic
   next-token corpus, save it, then distill a fresh student against it and
   verify the student's KD loss and its KL-to-teacher both drop.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import llm.training.tasks.builtin  # noqa: F401 - registers the built-in tasks
from llm.data.base import BaseDataModule
from llm.runtime import ModelFactory
from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import CheckpointManager, Logger
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks.lm_task import LanguageModelingTask
from tests.support.data import DummyLMDataModule


# Fixed, learnable next-token corpus: each of the ``num_samples`` sequences is
# ``[(i + j) % vocab]`` so token_{j+1} = (token_j + 1) % vocab. Unlike the fully
# random DummyLMDataModule this gives the model (teacher AND student) a stable,
# real function to reproduce — a necessary condition for demonstrating that
# distillation actually transfers the teacher's distribution.
class _FixedNextTokenModule(BaseDataModule):
    """Deterministic tiny corpus: labels == inputs (predict next token)."""

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    def _tensors(self, n: int) -> TensorDataset:
        vocab = self.config.model.vocab_size
        seq_len = self.config.model.max_seq_len
        rows = [(torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len] + i).fmod(vocab).long() for i in range(n)]
        inputs = torch.stack(rows)
        return TensorDataset(inputs, inputs)  # labels == inputs -> teach next token

    def train_dataloader(self, rank, world_size, device=None):
        dataset = self._tensors(self.config.training.num_samples)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        return (
            DataLoader(dataset, batch_size=self.config.training.batch_size, sampler=sampler, drop_last=True),
            sampler,
        )

    def val_dataloader(self, rank, world_size):
        return None, None


def _config(**training) -> Config:
    defaults = {
        "batch_size": 8,
        "epochs": 1,
        "num_samples": 64,
        "distill_temperature": 4.0,
        "distill_alpha": 0.0,
        "lr": 1e-3,
        "warmup_epochs": 0,
    }
    defaults.update(training)
    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=3, num_heads=3, max_seq_len=24),
        training=TrainingConfig(**defaults),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def _kl_to_teacher(engine: TrainingEngine, task, temperature: float) -> float:
    """KL(teacher_T || student_T) on the fixed cyclic probe sequence."""
    task.teacher.eval()
    engine.model.eval()
    vocab = engine.config.model.vocab_size
    seq_len = engine.config.model.max_seq_len
    x = torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len].unsqueeze(0)
    with torch.no_grad():
        s_logp = torch.log_softmax(engine.model(x) / temperature, dim=-1)
        t_p = torch.softmax(task.teacher(x) / temperature, dim=-1)
        return torch.nn.functional.kl_div(s_logp, t_p, reduction="batchmean").item()


def _save_ckpt(tmp_path, model, config) -> str:
    """Save ``model`` as a CheckpointManager checkpoint; return the 'best' stem."""
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path / "ckpts"), save_interval=1, keep_last_n=1, save_best=True),
        rank=0,
        logger=Logger(rank=0, config=LoggingConfig(log_level="ERROR")),
    )
    manager.save_checkpoint(
        epoch=1,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        loss=0.0,
        model_config=config.model.model_dump(),
    )
    return str(tmp_path / "ckpts" / "best")


def test_distill_task_registered():
    spec = TASK_REGISTRY.get("distill")
    assert spec is not None
    assert spec.task_cls is not None
    assert spec.task_cls.__name__ == "DistillationTask"


def test_distill_builds_frozen_teacher_and_trainable_student():
    from llm.training.tasks.distill_task import DistillationTask

    config = _config()
    task = DistillationTask(config, DummyLMDataModule(config))
    student = task.build_model()
    assert task.teacher is not None
    assert any(p.requires_grad for p in student.parameters())
    assert all(not p.requires_grad for p in task.teacher.parameters())
    # Different construction seeds -> student starts far from the teacher.
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        assert not torch.allclose(student(x), task.teacher(x), atol=1e-6)


def test_distill_loads_frozen_teacher_from_checkpoint(tmp_path):
    from llm.training.tasks.distill_task import DistillationTask

    config = _config()
    teacher = ModelFactory.from_config(config.model)
    saved = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    ckpt = _save_ckpt(tmp_path, teacher, config)

    task = DistillationTask(_config(distill_teacher_path=ckpt), DummyLMDataModule(config))
    student = task.build_model()
    assert task.teacher is not None
    assert all(not p.requires_grad for p in task.teacher.parameters())
    # The loaded teacher reproduces the checkpointed weights exactly.
    loaded = {k: v.detach() for k, v in task.teacher.state_dict().items()}
    assert set(loaded) == set(saved)
    assert all(torch.equal(loaded[k], saved[k]) for k in saved)
    # The student is fresh, so it starts away from the loaded teacher.
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        assert not torch.allclose(student(x), task.teacher(x), atol=1e-6)


def test_distill_engine_student_moves_toward_teacher(tmp_path):
    """Pre-train a teacher, save it, distill a fresh student against it."""

    # Tiny CPU models are hammered by 128-thread oversubscription; the default
    # thread pool makes even this small run pathological. Cap to a single thread
    # for the duration of the run, then restore.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_distill_e2e(tmp_path)
    finally:
        torch.set_num_threads(prev_threads)


def _run_distill_e2e(tmp_path) -> None:
    from llm.training.tasks.distill_task import DistillationTask

    # 1) Pre-train a confident teacher on the fixed corpus via the standard LM task.
    torch.manual_seed(7)
    teacher_train_cfg = _config(epochs=10)
    teacher_module = _FixedNextTokenModule(teacher_train_cfg)
    teacher_task = LanguageModelingTask(teacher_train_cfg, teacher_module)
    teacher_engine = TrainingEngine(
        config=teacher_train_cfg, task=teacher_task, rank=0, world_size=1, data_module=teacher_module
    )
    for epoch in range(10):
        teacher_engine._run_epoch(epoch)
    teacher_ckpt = _save_ckpt(tmp_path, teacher_engine.model, teacher_train_cfg)

    # 2) Distill a FRESH student against the frozen teacher loaded from disk.
    distill_cfg = _config(epochs=20, distill_teacher_path=teacher_ckpt)
    distill_module = _FixedNextTokenModule(distill_cfg)
    distill_task = DistillationTask(distill_cfg, distill_module)
    distill_engine = TrainingEngine(
        config=distill_cfg, task=distill_task, rank=0, world_size=1, data_module=distill_module
    )

    kl_before = _kl_to_teacher(distill_engine, distill_task, distill_cfg.training.distill_temperature)
    kd_losses = [distill_engine._run_epoch(epoch) for epoch in range(20)]
    kl_after = _kl_to_teacher(distill_engine, distill_task, distill_cfg.training.distill_temperature)

    # The student provably moves toward the (frozen, pre-trained) teacher:
    # both the KD loss and the KL-to-teacher must drop substantially.
    assert all(math.isfinite(loss) for loss in kd_losses)
    assert kd_losses[-1] < kd_losses[0] * 0.5, f"KD loss did not decrease: {kd_losses[0]:.4f} -> {kd_losses[-1]:.4f}"
    assert kl_after < kl_before * 0.5, f"student did not approach teacher: KL {kl_before:.4f} -> {kl_after:.4f}"
