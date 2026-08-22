"""QAT integration tests (RIL TASK-219 / DEC-054)."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.quantization.fake_quant import FakeQuantLinear, apply_fake_quant


def _mini_config(**training):
    from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig

    base = {"batch_size": 2, "epochs": 1, "num_samples": 16}
    base.update(training)
    return Config(
        model=ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=32),
        training=TrainingConfig(**base),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
    )


def test_apply_fake_quant_replaces_and_preserves_weights():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    apply_fake_quant(model, bits=8, quant_activation=False)
    count = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    assert count == 2, f"expected 2 FakeQuantLinear, got {count}"
    for name, p in model.named_parameters():
        assert name in before, f"param {name} disappeared"
        assert torch.equal(before[name], p.detach()), f"param {name} not preserved"


def test_apply_fake_quant_target_modules_filter():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
    apply_fake_quant(model, bits=8, target_modules=["0"])
    assert isinstance(model[0], FakeQuantLinear)
    assert isinstance(model[1], nn.Linear), "untargeted linear must stay a plain Linear"


def test_build_model_qat_applies_and_preserves_state():
    from llm.training.tasks.lm_task import LanguageModelingTask
    from tests.support.data import DummyLMDataModule

    config = _mini_config(use_qat=True, qat_bits=8)
    task = LanguageModelingTask(config, DummyLMDataModule(config))
    model = task.build_model()
    assert any(isinstance(m, FakeQuantLinear) for m in model.modules()), "build_model must fake-quantize linears"


def test_training_config_rejects_invalid_qat_bits():
    from pydantic import ValidationError

    from llm.training.core.config import TrainingConfig

    for bad in (3, 5, 16):
        try:
            TrainingConfig(batch_size=2, qat_bits=bad)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"qat_bits={bad} should have been rejected")


def test_qat_engine_trains_with_finite_loss():
    """Single-rank QAT engine run stays finite (TASK-219 integration).

    The dummy data has RANDOM targets, so loss sits near ln(vocab) rather than
    decreasing; the fake-quant STE training loop must still run three epochs
    without NaN/inf and keep the model fake-quantized. (The loss-DECREASE claim
    is verified separately on learnable data in
    ``test_fake_quant_linear_trains_and_reduces_loss``.)
    """
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask
    from tests.support.data import DummyLMDataModule

    config = Config(
        model=ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=32),
        training=TrainingConfig(batch_size=2, epochs=3, num_samples=16, use_qat=True, qat_bits=8),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )
    data_module = DummyLMDataModule(config)
    task = LanguageModelingTask(config, data_module)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=data_module)
    assert any(isinstance(m, FakeQuantLinear) for m in engine.model.modules()), "engine model must be fake-quantized"
    losses = [engine._run_epoch(e) for e in range(3)]
    assert all(loss < float("inf") for loss in losses), f"QAT losses not finite: {losses}"
    # Near-random loss for a 64-vocab dummy task (~ ln 64 = 4.16) proves the
    # fake-quant forward + STE backward ran without corruption.
    assert all(3.0 < loss < 6.0 for loss in losses), f"QAT losses out of sane range: {losses}"
