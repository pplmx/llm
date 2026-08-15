from pathlib import Path
from unittest import mock

import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.engine import TrainingEngine
from llm.training.core.logger import Logger
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


def test_dpo_ref_model_persisted_and_restored(tiny_config):
    """RIL round-60 deep-dive Finding 1: the frozen DPO reference must be
    checkpointed with the run and restored verbatim on resume.

    ``DPOTask.build_model`` snapshots the policy into ``ref_model`` BEFORE
    the engine loads any checkpoint into the policy, and the checkpoint
    never carried the reference — so a resumed DPO run computed every
    log-ratio against a freshly-random reference (silently wrong loss).
    The fix persists ``ref_model`` in checkpoint extra_state and restores
    it.
    """
    task = DPOTask(tiny_config, data_module=None)
    task.build_model()
    assert task.ref_model is not None

    # Persist (snapshot ref weights so we can compare post-restore).
    state = task.get_checkpoint_state()
    assert state is not None, "DPOTask should contribute ref to checkpoint state"
    assert "dpo_ref_model" in state
    orig = {k: v.clone() for k, v in task.ref_model.state_dict().items()}

    # Restore into a fresh task (simulating a resume) — the ref must come
    # back bit-identical, not as a fresh random copy.
    task2 = DPOTask(tiny_config, data_module=None)
    task2.build_model()
    task2.load_checkpoint_state(state)
    restored = task2.ref_model.state_dict()
    assert set(restored) == set(orig)
    for key in orig:
        assert torch.equal(restored[key], orig[key]), f"ref not restored at {key}"


def test_dpo_ref_model_synced_from_loaded_checkpoint(tiny_config, tmp_path):
    """REG-RED: resuming DPO from an SFT/base checkpoint must align the
    frozen reference with the LOADED policy — the standard way DPO runs
    (``resume_from_checkpoint=<SFT ckpt>``). Before the fix the reference
    stayed at its random build-time init while the policy carried the base
    weights, so every DPO log-ratio was computed against a random model.

    End-to-end through the engine: build a base checkpoint, then construct
    a DPO engine resuming from it. The reference must equal the loaded
    base weights.
    """
    import torch as _torch

    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.checkpoint import CheckpointManager
    from llm.training.core.engine import TrainingEngine

    cfg = tiny_config.model_copy(deep=True)
    cfg.checkpoint.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.distributed.backend = "gloo"

    # 1. Build a "base" (SFT) model: a fresh policy whose weights we then
    #    distinguish from random init by adding a constant marker.
    dm = SyntheticDataModule(cfg)
    dm.setup()
    base_task = DPOTask(cfg, dm)
    base_model = base_task.build_model()
    with _torch.no_grad():
        for p in base_model.parameters():
            p.add_(0.5)  # marker: base != any fresh random init

    # 2. Save it as a plain checkpoint WITHOUT a persisted ref (an SFT ckpt).
    manager = CheckpointManager(cfg.checkpoint, 0, Logger(0, cfg.logging))
    manager.save_checkpoint(
        epoch=0,
        model=base_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        loss=1.0,
        extra_state=None,
        model_config=cfg.model.model_dump(),
    )

    # 3. Resume DPO from that base checkpoint.
    cfg.checkpoint.resume_from_checkpoint = str(Path(cfg.checkpoint.checkpoint_dir) / "latest")
    task = DPOTask(cfg, dm)
    engine = TrainingEngine(cfg, task, rank=0, world_size=1, data_module=dm)

    # 4. The loaded policy equals the base; the reference must equal it too.
    #    (Compare on CPU — the pipeline moves the policy to CUDA but the
    #    reference stays wherever it was built.)
    policy_sd = {k: v.detach().to("cpu") for k, v in engine.model.state_dict().items()}
    ref_sd = {k: v.detach().to("cpu") for k, v in task.ref_model.state_dict().items()}
    assert set(ref_sd) == set(policy_sd)
    for key in ref_sd:
        assert _torch.equal(ref_sd[key], policy_sd[key]), (
            f"DPO ref diverged from checkpoint-loaded base at {key}: reference was not synced to the loaded policy"
        )


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
