"""Tests for PPO trainer checkpoint state."""

import pytest
import torch

from llm.training.core.config import PPOConfig
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.rlhf.value_model import ValueModel
from tests.support.devices import DEFAULT_DEVICE


@pytest.fixture
def tiny_setup(tiny_model):
    from llm.training.tasks.reward_task import RewardModel

    class SimpleTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 100 for c in text[:10]]

        def decode(self, ids: list[int]) -> str:
            return "x"

        eos_token_id = None

    policy = tiny_model
    reward_model = RewardModel(tiny_model)
    value_model = ValueModel(tiny_model)
    config = PPOConfig(value_coef=0.5, ppo_epochs=1, mini_batch_size=1, response_max_len=2)

    trainer = PPOTrainer(
        policy_model=policy,
        reward_model=reward_model,
        tokenizer=SimpleTokenizer(),
        config=config,
        value_model=value_model,
        device=str(DEFAULT_DEVICE),
    )
    return trainer


def test_ppo_trainer_checkpoint_roundtrip(tiny_setup):
    trainer = tiny_setup
    trainer.global_step = 3

    state = trainer.get_checkpoint_state()
    assert state["global_step"] == 3
    assert "value_model" in state
    assert "value_optimizer" in state

    # Snapshot state is moved to CPU by _snapshot_state; keep before on
    # the same device for a like-for-like comparison.
    before = trainer.value_model.value_head.weight.detach().cpu().clone()
    trainer.value_model.value_head.weight.data.fill_(0.0)
    # Saved snapshot must not alias live parameters.
    assert not torch.allclose(state["value_model"]["value_head.weight"], torch.zeros_like(before))

    trainer.load_checkpoint_state(state)
    assert trainer.global_step == 3
    assert torch.allclose(trainer.value_model.value_head.weight.detach().cpu(), before)


def test_ppo_trainer_ref_model_persisted_and_restored(tiny_setup):
    """RIL round-60 deep-dive Finding 1 (PPO): the frozen KL reference must be
    checkpointed and restored verbatim on resume.

    ``PPOTrainer.__init__`` deep-copies the policy into ``ref_model`` inside
    ``prepare_training`` — BEFORE the engine loads any checkpoint into the
    policy — and the checkpoint never carried the reference, so a resumed PPO
    run computed its KL penalty against a stale model.
    """
    trainer = tiny_setup
    assert trainer.ref_model is not None, "use_ref_model default should build a ref"

    state = trainer.get_checkpoint_state()
    assert "ref_model" in state, "PPOTrainer should persist the reference model"

    orig = {k: v.detach().cpu().clone() for k, v in trainer.ref_model.state_dict().items()}
    # Perturb the live ref so a restore is observable.
    with torch.no_grad():
        for p in trainer.ref_model.parameters():
            p.add_(1.0)

    trainer.load_checkpoint_state(state)
    restored = trainer.ref_model.state_dict()
    for key in orig:
        assert torch.equal(restored[key].detach().cpu(), orig[key]), f"ref not restored at {key}"


def test_ppo_trainer_on_checkpoint_loaded_syncs_ref_to_policy(tiny_setup):
    """RIL round-60 deep-dive Finding 1 (PPO): when the loaded checkpoint did
    NOT carry a persisted ref (an SFT/base checkpoint — the standard RLHF
    flow), the reference must equal the checkpoint-loaded policy."""

    trainer = tiny_setup
    assert trainer.ref_model is not None

    # Simulate: the checkpoint load moved the policy to a known state, the ref
    # was NOT restored from extra_state (fresh SFT-base resume).
    base_policy = trainer.policy
    with torch.no_grad():
        for p in base_policy.parameters():
            p.add_(0.5)

    trainer.on_checkpoint_loaded(base_policy)

    ref_sd = trainer.ref_model.state_dict()
    pol_sd = base_policy.state_dict()
    for key in ref_sd:
        assert torch.equal(ref_sd[key].detach().cpu(), pol_sd[key].detach().cpu()), (
            f"PPO ref diverged from loaded base at {key}"
        )


def test_ppo_trainer_on_checkpoint_loaded_keeps_persisted_ref(tiny_setup):
    """Once a persisted ref was restored (mid-PPO resume), on_checkpoint_loaded
    must NOT overwrite it with the now-moved policy."""
    trainer = tiny_setup
    assert trainer.ref_model is not None

    # Persist a ref, then move the policy.
    state = trainer.get_checkpoint_state()
    orig_ref = {k: v.detach().cpu().clone() for k, v in trainer.ref_model.state_dict().items()}
    with torch.no_grad():
        for p in trainer.policy.parameters():
            p.add_(1.0)

    # A mid-PPO resume restores the persisted ref...
    trainer.load_checkpoint_state(state)
    # ...then the engine's post-load hook fires — it must keep the restored
    # base, not re-point at the moved policy.
    trainer.on_checkpoint_loaded(trainer.policy)
    for key in orig_ref:
        assert torch.equal(trainer.ref_model.state_dict()[key].detach().cpu(), orig_ref[key]), (
            f"restored ref overwritten by moved policy at {key}"
        )
