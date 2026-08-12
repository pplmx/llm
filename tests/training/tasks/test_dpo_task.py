from unittest import mock

import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.dpo_task import DPOTask


def test_dpo_task_init_and_build(tiny_config):
    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()

    assert task.ref_model is not model

    for param in task.ref_model.parameters():
        assert not param.requires_grad

    # Check policy trainable
    assert any(p.requires_grad for p in model.parameters())


def test_dpo_ref_model_is_broadcast_by_engine(tiny_config):
    """Regression (RIL ISS-038): DPOTask clones its frozen reference inside
    ``build_model`` — *before* the engine broadcasts the policy from rank 0.
    Without an extra sync the reference copy carries each rank's own
    RNG-initialised weights and multi-GPU DPO computes logps against a
    rank-divergent reference. The engine must broadcast the reference too.
    """
    data_module = SyntheticDataModule(tiny_config)
    data_module.setup()
    task = DPOTask(tiny_config, data_module)
    broadcast_calls: list = []

    with (
        mock.patch("llm.training.core.engine.broadcast_parameters", side_effect=broadcast_calls.append),
        mock.patch("llm.training.core.engine._cuda_usable", return_value=False),
    ):
        engine = TrainingEngine(tiny_config, task, rank=0, world_size=1, data_module=data_module)

    policy_model = engine.model
    assert len(broadcast_calls) == 2, f"expected policy + ref broadcast, got {len(broadcast_calls)}"
    assert broadcast_calls[0] is policy_model
    assert broadcast_calls[1] is task.ref_model
    assert task.ref_model is not policy_model
    # Policy and reference are identical copies (DPOTask snapshot the policy
    # state into the reference at build time); the engine must sync both so
    # every rank agrees after the (no-op on single rank) broadcast.
    ref_sd = task.ref_model.state_dict()
    pol_sd = policy_model.state_dict()
    assert set(ref_sd) == set(pol_sd)
    for key in ref_sd:
        assert torch.equal(ref_sd[key], pol_sd[key]), f"ref/policy diverge at {key}"


def test_dpo_task_train_step(tiny_config):
    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()
    criterion = None  # DPOTask doesn't use criterion for DPO loss

    batch_size, seq_len, vocab_size = 2, 4, tiny_config.model.vocab_size
    chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
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

    assert not torch.isnan(loss)
    assert 0.0 <= metrics["reward_acc"] <= 1.0
