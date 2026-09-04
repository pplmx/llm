import pytest
import torch
import torch.nn as nn

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.lm_task import LanguageModelingTask


@pytest.fixture
def mock_config():
    config = Config()
    config.model = ModelConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
        use_glu=True,
    )
    config.training = TrainingConfig(batch_size=4, epochs=1, lr=1e-4)
    return config


@pytest.mark.heavy
def test_lm_task_initialization(mock_config):
    data_module = SyntheticDataModule(mock_config)  # Just for interface
    task = LanguageModelingTask(mock_config, data_module)

    model = task.build_model()
    assert model.hidden_size == 128
    assert len(model.transformer_blocks) == 2

    # Check if SwiGLU is passed (MLP should have gate_proj)
    assert hasattr(model.transformer_blocks[0].mlp, "gate_proj")


@pytest.mark.heavy
def test_lm_task_step(mock_config):
    data_module = SyntheticDataModule(mock_config)
    task = LanguageModelingTask(mock_config, data_module)

    model = task.build_model()
    criterion = task.build_criterion()

    # Mock batch: (input_ids, labels)
    batch_size = 4
    seq_len = 16
    vocab_size = 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    batch = (input_ids, labels)

    loss, metrics = task.train_step(batch, model, criterion)

    assert not torch.isnan(loss)
    assert metrics["loss"] > 0
    assert metrics["ppl"] >= 1.0


@pytest.mark.heavy
def test_lm_task_step_uses_next_token_shift(mock_config):
    """Regression: the LM loss compares ``logits[s]`` with ``labels[s+1]``.

    Without the shift, a model can drive the loss to ~0 by copying the
    current token (its own embedding leaks through the residual stream)
    while scoring at random on real next-token evaluation — training
    reports ppl≈1.0 but the checkpoint is useless. This test pins the
    loss to the manual shifted-CE reference.
    """
    mock_config.model.dropout = 0.0  # deterministic forward for the loss identity
    data_module = SyntheticDataModule(mock_config)
    task = LanguageModelingTask(mock_config, data_module)
    model = task.build_model()
    criterion = task.build_criterion()

    batch_size, seq_len, vocab_size = 4, 16, 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    batch = (input_ids, labels)

    loss, metrics = task.train_step(batch, model, criterion)

    logits = model(input_ids)
    expected = criterion(
        logits[..., :-1, :].contiguous().view(-1, vocab_size),
        labels[..., 1:].contiguous().view(-1),
    )
    assert torch.allclose(loss, expected, atol=1e-6)
    assert metrics["ppl"] == pytest.approx(torch.exp(expected).item())

    # Same contract in the validation path.
    val_loss, val_metrics = task.validation_step(batch, model, criterion)
    assert torch.allclose(val_loss, expected, atol=1e-6)
    assert val_metrics["val_ppl"] == pytest.approx(torch.exp(expected).item())


@pytest.mark.heavy
def test_lm_task_validation(mock_config):
    data_module = SyntheticDataModule(mock_config)
    task = LanguageModelingTask(mock_config, data_module)

    model = task.build_model()
    criterion = task.build_criterion()

    batch_size = 2
    seq_len = 8
    vocab_size = 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    batch = (input_ids, labels)

    _loss, metrics = task.validation_step(batch, model, criterion)

    assert "val_loss" in metrics
    assert "val_ppl" in metrics


@pytest.mark.quick
def test_lm_task_nan_loss_is_surfaced_loudly(caplog, mock_config):
    """Regression (RIL ISS-343): a NaN step must be surfaced with a warning,
    not silently reported as a perfect loss 0.0 / ppl 1.0 as if the model
    trained."""

    class _NaNModel(nn.Module):
        def forward(self, input_ids):
            return torch.full((*input_ids.shape, 8), float("nan"))

    class _NanCriterion(nn.Module):
        def forward(self, logits, targets):
            return torch.tensor(float("nan"), requires_grad=True)

    data_module = SyntheticDataModule(mock_config)
    task = LanguageModelingTask(mock_config, data_module)

    batch = (
        torch.randint(0, 128, (2, 8)),  # seq_len >= 2 so the shift guard passes
        torch.randint(0, 128, (2, 8)),
    )

    with caplog.at_level("WARNING", logger="llm.training.tasks.lm_task"):
        loss, metrics = task.train_step(batch, _NaNModel(), _NanCriterion())

    # The 0.0 ``requires_grad`` fallback keeps training alive...
    assert loss.item() == 0.0
    assert metrics == {"loss": 0.0, "ppl": 1.0}
    # ...but the divergence must be visible in the logs, not swallowed.
    assert any("NaN" in message for message in caplog.messages), "NaN loss branch must emit a warning (RIL ISS-343)"
