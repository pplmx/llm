import torch

from llm.training.tasks.dpo_task import DPOTask


def test_dpo_task_init_and_build(tiny_config):
    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()

    assert task.ref_model is not None
    assert task.ref_model is not model

    # Check freezing
    for p in task.ref_model.parameters():
        assert not p.requires_grad

    # Check policy trainable
    assert any(p.requires_grad for p in model.parameters())


def test_dpo_task_train_step(tiny_config):
    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()
    criterion = None  # DPOTask doesn't use criterion for DPO loss

    B, S, V = 2, 4, tiny_config.model.vocab_size
    chosen_ids = torch.randint(0, V, (B, S))
    rejected_ids = torch.randint(0, V, (B, S))
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100
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
