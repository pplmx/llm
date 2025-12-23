import pytest
import torch

from llm.data.synthetic_data_module import SyntheticDataModule
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


def test_lm_task_initialization(mock_config):
    data_module = SyntheticDataModule(mock_config)  # Just for interface
    task = LanguageModelingTask(mock_config, data_module)

    model = task.build_model()
    assert model.hidden_size == 128
    assert len(model.transformer_blocks) == 2

    # Check if SwiGLU is passed (MLP should have gate_proj)
    assert hasattr(model.transformer_blocks[0].mlp, "gate_proj")


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

    assert isinstance(loss, torch.Tensor)
    assert "loss" in metrics
    assert "ppl" in metrics
    assert metrics["loss"] > 0
    assert metrics["ppl"] >= 1.0


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

    loss, metrics = task.validation_step(batch, model, criterion)

    assert "val_loss" in metrics
    assert "val_ppl" in metrics
