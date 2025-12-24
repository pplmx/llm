import math

import pytest
import torch

from llm.training.core.config import Config, ModelConfig
from llm.training.tasks.lm_task import LanguageModelingTask


@pytest.fixture
def task_context():
    model_cfg = ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=64)
    cfg = Config(model=model_cfg)
    # SyntheticDataModule is fine for providing dummy structure
    from llm.data.synthetic_data_module import SyntheticDataModule

    dm = SyntheticDataModule(cfg)
    task = LanguageModelingTask(cfg, dm)
    return task, model_cfg


@pytest.mark.heavy
def test_lm_task_perplexity_masked_entirely(task_context):
    """Test perplexity calculation when targets are entirely masked (ignore_index)."""
    task, model_cfg = task_context
    model = task.build_model()
    criterion = task.build_criterion()

    # ignore_index is -100 by default in nn.CrossEntropyLoss
    targets = torch.full((1, 4), -100, dtype=torch.long)

    loss, metrics = task.train_step((torch.zeros(1, 4, dtype=torch.long), targets), model, criterion)

    assert loss.item() == 0.0
    assert metrics["ppl"] == 1.0


@pytest.mark.heavy
def test_lm_task_perplexity_extreme_values(task_context):
    """Test perplexity with high loss."""
    task, model_cfg = task_context
    model = task.build_model()
    criterion = task.build_criterion()

    # Use real inputs and targets
    input_ids = torch.zeros(1, 1, dtype=torch.long)
    targets = torch.zeros(1, 1, dtype=torch.long)

    # Put model in eval mode to avoid dropout affecting loss calculation comparison
    model.eval()

    loss, metrics = task.train_step((input_ids, targets), model, criterion)

    # loss and ppl should be present and related
    assert loss.item() >= 0
    assert "ppl" in metrics
    assert metrics["ppl"] == pytest.approx(math.exp(loss.item()), rel=1e-5)


@pytest.mark.heavy
def test_lm_task_target_creation(task_context):
    """Verify that targets are handled correctly in train_step."""
    task, model_cfg = task_context
    model = task.build_model()
    criterion = task.build_criterion()

    # In LM task, targets are usually provided by the dataloader as shifted versions.
    # train_step just passes them to model and criterion.
    input_ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])

    loss, metrics = task.train_step((input_ids, targets), model, criterion)
    assert "loss" in metrics
    assert "ppl" in metrics
