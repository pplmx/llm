import torch

from llm.training.core.config import Config, ModelConfig
from llm.training.tasks.dpo_task import DPOTask


def test_dpo_task_init_and_build():
    config = Config(model=ModelConfig(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16))
    task = DPOTask(config, data_module=None)

    # build_model creates BOTH policy and ref model (reconstruction)
    model = task.build_model()  # This returns policy, creates task.ref_model internally

    assert task.ref_model is not None
    assert task.ref_model is not model

    # Check freezing
    for p in task.ref_model.parameters():
        assert not p.requires_grad

    # Check policy trainable
    assert any(p.requires_grad for p in model.parameters())


def test_dpo_task_train_step():
    config = Config(model=ModelConfig(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16))
    task = DPOTask(config, data_module=None)

    # Needs a built model with ref model set
    model = task.build_model()

    criterion = None  # DPOTask doesn't use criterion for DPO loss

    B, S, V = 2, 4, 100
    chosen_ids = torch.randint(0, V, (B, S))
    rejected_ids = torch.randint(0, V, (B, S))
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100  # Mask first token
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, 0] = -100

    batch = {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
    }

    loss, metrics = task.train_step(batch, model, criterion)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "reward_acc" in metrics
    assert "reward_margin" in metrics
