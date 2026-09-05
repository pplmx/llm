"""Integration + CPU e2e for the multimodal training task (TASK-227)."""

from __future__ import annotations

import torch

import llm.training.tasks.builtin  # noqa: F401 - registers built-in tasks
from llm.multimodal.data import MultimodalDataModule
from llm.multimodal.model import MultimodalModel
from llm.training.task_registry import TASK_REGISTRY


def _config(num_samples: int = 32, use_rope: bool = True):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24, use_rope=use_rope),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_multimodal_task_registered():
    spec = TASK_REGISTRY.get("multimodal")
    assert spec is not None
    from llm.multimodal.task import MultimodalTask

    assert spec.task_cls is MultimodalTask


def test_multimodal_task_builds_fusion_model_and_ce_criterion():
    from llm.multimodal.task import MultimodalTask

    config = _config()
    module = MultimodalDataModule(config, modality="linear", input_dim=16)
    task = MultimodalTask(config, module)
    model = task.build_model()
    assert isinstance(model, MultimodalModel)
    assert isinstance(task.build_criterion(), torch.nn.CrossEntropyLoss)


def test_multimodal_task_build_model_threads_datamodule_embed_dim():
    """Regression (RIL ISS-354): build_model() must thread the datamodule's
    embed_dim into the fusion projection. It previously defaulted modal_dim to
    decoder.hidden_size, so a datamodule with embed_dim != hidden_size crashes
    with a cryptic nn.Linear mismatch at forward time (loud, but only at the
    first train step)."""
    from llm.multimodal.task import MultimodalTask

    config = _config()  # hidden_size=24
    module = MultimodalDataModule(config, modality="linear", input_dim=8, embed_dim=17)
    task = MultimodalTask(config, module)
    model = task.build_model()
    assert isinstance(model, MultimodalModel)
    assert model.fusion.modal_dim == 17

    # A real forward on a 17-dim modal batch works end to end.
    batch = {"input_ids": torch.randint(0, 32, (2, 6)), "modal_embeds": torch.randn(2, 17)}
    logits = model(batch["input_ids"], batch["modal_embeds"])
    assert logits.shape == (2, 6, 32)


def test_multimodal_model_encoder_embed_dim_is_source_of_truth():
    """In-forward (trainable-tower) encoders declare their own embed_dim; the
    fusion projection must adopt it, and an explicit modal_dim that contradicts
    the encoder must fail loudly instead of silently misprojecting."""
    import pytest

    from llm.multimodal.encoders import LinearModalityEncoder
    from llm.runtime import ModelFactory

    decoder = ModelFactory.from_config(_config().model)
    encoder = LinearModalityEncoder(input_dim=4, embed_dim=17)
    model = MultimodalModel(decoder, encoder=encoder)
    assert model.fusion.modal_dim == 17

    with pytest.raises(ValueError, match="embed_dim"):
        MultimodalModel(decoder, modal_dim=24, encoder=encoder)


def _next_token_accuracy(model, module):
    model.eval()
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    # Move the loader batch to the model's device: the engine does this inside
    # train_step, but this standalone probe doesn't (CPU batch vs CUDA model
    # would raise on GPU machines).
    device = next(model.parameters()).device
    batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["modal_embeds"])
        pred = logits.argmax(-1)
    target = batch["labels"][:, 1:]
    return (pred[:, :-1] == target).float().mean().item()


def test_multimodal_engine_fused_training_converges():
    """CPU e2e: the fused modal-conditioned decoder trains (loss drops) and its
    text next-token accuracy improves — proof the modal path is wired through
    the real engine."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = _config()
        module = MultimodalDataModule(config, modality="linear", input_dim=16)
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

        losses = [engine._run_epoch(epoch) for epoch in range(30)]
        assert all(loss == loss for loss in losses)  # finite
        assert losses[-1] < losses[0] * 0.5, f"multimodal loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert losses[-1] < 1.5

        acc = _next_token_accuracy(engine.model, module)
        assert acc > 0.7, f"fused text accuracy too low: {acc:.3f}"
    finally:
        torch.set_num_threads(prev)
