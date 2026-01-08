import torch

from llm.training.tasks.sft_task import SFTTask


def test_sft_task_train_step(tiny_config, tiny_model):
    # Use fixtures for config and model
    task = SFTTask(tiny_config, data_module=None)
    criterion = task.build_criterion()

    # Batch: 2 sequences, len 4
    batch = {
        "input_ids": torch.randint(0, 100, (2, 4)),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]]),
        "attention_mask": torch.ones(2, 4),
    }

    loss, metrics = task.train_step(batch, tiny_model, criterion)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss" in metrics
    assert "ppl" in metrics
