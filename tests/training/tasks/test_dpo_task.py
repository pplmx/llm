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
    """Regression: DPOTask clones its frozen reference inside
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
    """deep-dive Finding 1: the frozen DPO reference must be
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


def test_dpo_batch_logps_shape_guard_raises(tiny_config):
    """The shared _get_batch_logps raises a clear error on logits/labels seq-len mismatch.

       The guard lives on LanguageModelingTask (shared by DPO and SimPO since the
    dedup); a mismatched batch would otherwise gather garbage windows
       with an opaque downstream error.
    """
    import pytest

    task = DPOTask(tiny_config, data_module=None)
    logits = torch.zeros(2, 4, tiny_config.model.vocab_size)  # seq_len 4
    labels = torch.full((2, 5), -100)  # seq_len 5 — mismatched
    with pytest.raises(ValueError, match="Logits seq_len 4 != Labels seq_len 5"):
        task._get_batch_logps(logits, labels)


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
        # The data pipeline always supplies these; DPOTask now
        # feeds them to the model forwards.
        "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    loss, metrics = task.train_step(batch, model, criterion)

    assert not torch.isnan(loss)
    assert 0.0 <= metrics["reward_acc"] <= 1.0


def test_dpo_attention_masks_reach_the_model(tiny_config):
    """Regression for DPOTask must pass the per-row attention
    masks the data pipeline builds into the policy and reference forwards.

    Zeroing a real content token's mask bit must change the DPO loss — before
    the fix the masks never reached the model, so the loss was completely
    mask-insensitive. (The standard causal right-padded layout masks trailing
    pads by causality regardless, which is why the old discard was 'bounded
    impact', but any non-trailing padding is silently corrupted without it.)
    """
    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()
    # Make policy != ref so the BT logit is non-trivial (an identical
    # policy/ref yields a constant -log(sigmoid(0)) that is mask-invariant).
    torch.manual_seed(123)
    task.ref_model.load_state_dict(task.build_model().state_dict())
    model.eval()
    task.ref_model.eval()

    batch_size, seq_len = 2, 6
    vocab_size = tiny_config.model.vocab_size
    chosen = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected = torch.randint(0, vocab_size, (batch_size, seq_len))

    def make_batch(attn: torch.Tensor) -> dict:
        return {
            "chosen_input_ids": chosen,
            "chosen_labels": chosen.clone(),
            "rejected_input_ids": rejected,
            "rejected_labels": rejected.clone(),
            "chosen_attention_mask": attn,
            "rejected_attention_mask": attn.clone(),
        }

    ones = torch.ones(batch_size, seq_len, dtype=torch.long)
    content_masked = ones.clone()
    content_masked[:, 1] = 0  # mask out the 2nd (real) token of every row
    with torch.no_grad():
        loss_ones, _ = task.train_step(make_batch(ones), model, None)
        loss_masked, _ = task.train_step(make_batch(content_masked), model, None)
    # Same layout, only the mask differs — a mask-transparent DPO would return
    # identical losses (the pre-fix behavior this guards against).
    assert not torch.allclose(loss_ones, loss_masked), (
        "DPO loss is insensitive to the attention mask — the masks never reach the model"
    )


def test_dpo_batch_logps_rejects_length_one_sequences(tiny_config):
    """A length-1 sequence has no next-token target, so it must be refused.

    ``_get_batch_logps`` shifts logits by one, so ``seq_len=1`` yields an EMPTY
    target tensor and the helper used to return ``0.0`` for every row. DPO and
    SimPO both consume that number: a zero log-prob makes the implicit reward
    and the chosen-vs-rejected margin vanish, so the loss sits at exactly its
    initial value while the run looks healthy — a silent no-op. The sibling
    LM/SFT/CST shift sites already raise this message; this pins the shared
    preference helper to the same contract so a degenerate batch cannot take
    the quiet path.
    """
    import pytest

    task = DPOTask(tiny_config, data_module=None)
    logits = torch.zeros(2, 1, tiny_config.model.vocab_size)
    labels = torch.zeros(2, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="sequence length must be > 1"):
        task._get_batch_logps(logits, labels)

    # The same guard must fire on SimPO's length-normalised path, which passes
    # ``average_log_prob=True`` and divides by the (empty) mask sum.
    with pytest.raises(ValueError, match="sequence length must be > 1"):
        task._get_batch_logps(logits, labels, average_log_prob=True)


def test_dpo_batch_logps_still_accepts_a_real_sequence(tiny_config):
    """Counterpart: the new guard must not reject a usable batch.

    Without this, a guard that fired on every input would satisfy the test
    above while breaking DPO entirely.
    """
    task = DPOTask(tiny_config, data_module=None)
    logits = torch.zeros(1, 4, tiny_config.model.vocab_size)
    labels = torch.zeros(1, 4, dtype=torch.long)
    out = task._get_batch_logps(logits, labels)
    assert out.shape == (1,), f"expected one log-prob per batch row, got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


# --- the DPO loss must match its published formula --------------------------
#
# The existing train_step test asserts only `not isnan` and that reward_acc is in
# [0, 1] — both true for a loss that is numerically WRONG. The DPO objective
# (Rafailov et al. 2023) is
#
#     L = -log sigmoid(beta * [(log pi_w - log pi_l) - (log ref_w - log ref_l)])
#
# with two checkable consequences: the longhand form must agree with
# `logsigmoid`, and a policy identical to the reference must give exactly ln(2).


def test_the_dpo_objective_matches_the_published_formula():
    """Our loss expression equals the paper's, computed a different way."""
    import torch.nn.functional as functional

    beta = 0.1
    pi_w, pi_l = torch.tensor([-2.0, -1.5]), torch.tensor([-3.0, -2.5])
    ref_w, ref_l = torch.tensor([-2.2, -1.8]), torch.tensor([-2.9, -2.4])

    ours = (-functional.logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))).mean()
    # log sigmoid(z) = -log(1 + exp(-z)), expanded by hand.
    z = beta * ((pi_w - pi_l) - (ref_w - ref_l))
    longhand = (torch.log(1 + torch.exp(-z))).mean()

    assert torch.allclose(ours, longhand, atol=1e-6), f"logsigmoid form {float(ours)} != longhand {float(longhand)}"


def test_an_identical_policy_and_reference_give_exactly_ln_two():
    """At initialisation the policy equals the reference, so the loss is ln(2).

    This is the value DPO runs are expected to START at; getting anything else
    means the beta scaling, the log-ratio ordering, or the shift is wrong — all
    of which still produce a plausible-looking number and a healthy log.
    """
    import math

    import torch.nn.functional as functional

    beta = 0.1
    policy_chosen, policy_rejected = torch.tensor([-2.0, -1.5]), torch.tensor([-3.0, -2.5])
    # Policy == reference.
    ref_chosen, ref_rejected = policy_chosen.clone(), policy_rejected.clone()

    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    loss = (-functional.logsigmoid(beta * logits)).mean()

    assert abs(float(loss) - math.log(2)) < 1e-6, (
        f"an identical policy/reference pair gave {float(loss)}, expected ln(2) = {math.log(2)}"
    )


def test_the_loss_is_zero_when_the_chosen_side_dominates():
    """As the chosen-rejected margin grows the loss must vanish, not explode.

    A sign flip in the log-ratio (chosen minus rejected vs. the reverse) leaves
    the loss bounded and plausible, so the DIRECTION is what this pins.
    """
    import torch.nn.functional as functional

    beta = 0.1
    # Chosen hugely preferred over rejected, policy identical to reference.
    big = torch.tensor([50.0])
    loss = (-functional.logsigmoid(beta * big)).mean()
    # -log sigmoid(0.1 * 50) = -log sigmoid(5) ~ 0.0067, so the bound is 0.01:
    assert float(loss) < 0.01, f"a strongly preferred chosen side gave loss {float(loss)}"

    # Reversed margin: penalised, so the loss is large.
    reversed_loss = (-functional.logsigmoid(beta * -big)).mean()
    assert float(reversed_loss) > 5.0, f"a reversed margin gave loss {float(reversed_loss)}"


def test_beta_scales_the_margin():
    """Doubling beta must equal doubling the log-ratio inside the sigmoid."""
    import torch.nn.functional as functional

    margin = torch.tensor([1.5])
    at_beta_1 = (-functional.logsigmoid(0.1 * margin)).mean()
    at_beta_2 = (-functional.logsigmoid(0.2 * margin)).mean()
    at_double_margin = (-functional.logsigmoid(0.1 * (2 * margin))).mean()

    assert torch.allclose(at_beta_2, at_double_margin, atol=1e-6)
    assert float(at_beta_2) < float(at_beta_1), "a larger beta must reduce the loss for a positive margin"


# --- the loss must be ln(2) AT INIT, measured through train_step ------------
#
# The formula tests above recompute the objective from tensors they build
# themselves, so they pin the MATHS but never execute DPOTask. Verified: flipping
# the sign of ``pi_logratios - ref_logratios`` inside train_step left all of them
# — and the pre-existing train_step test — passing. The check that reaches the
# implementation is structural: ``build_model`` deep-copies the policy into the
# reference, so at initialisation the two models are IDENTICAL and the log-ratio
# difference is exactly zero. Any real DPO loss must then be ln(2).


def test_train_step_matches_the_formula_when_policy_differs_from_reference(tiny_config):
    """Drive train_step with a PERTURBED reference and compare against the formula.

    The ln(2) identity alone cannot detect a reversed log-ratio: when the policy
    equals the reference the difference is zero either way, so flipping the sign
    leaves the loss at ln(2) (verified -- it does). The sign only shows up once
    the two models disagree, so the reference is perturbed here and the loss is
    recomputed independently from the same log-probabilities.
    """
    import torch.nn.functional as functional

    task = DPOTask(tiny_config, data_module=None)
    model = task.build_model()

    # Perturb the reference so policy != reference.
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in task.ref_model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))

    batch_size, seq_len, vocab_size = 2, 5, tiny_config.model.vocab_size
    torch.manual_seed(0)
    chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, 0] = -100
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    batch = {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "chosen_attention_mask": mask,
        "rejected_attention_mask": mask,
    }

    beta = task.beta
    model.eval()
    with torch.no_grad():
        loss, _metrics = task.train_step(batch, model, None)
        # Independent recomputation from the two models' own log-probabilities.
        pi_w = task._get_batch_logps(model(chosen_ids, attn_mask=mask), chosen_labels)
        pi_l = task._get_batch_logps(model(rejected_ids, attn_mask=mask), rejected_labels)
        ref_w = task._get_batch_logps(task.ref_model(chosen_ids, attn_mask=mask), chosen_labels)
        ref_l = task._get_batch_logps(task.ref_model(rejected_ids, attn_mask=mask), rejected_labels)
    expected = (-functional.logsigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))).mean()

    assert abs(float(loss) - float(expected)) < 1e-6, (
        f"train_step returned {float(loss):.6f} but the formula gives {float(expected):.6f}; "
        "a reversed log-ratio would differ here only because policy != reference"
    )
    assert abs(float(loss) - 0.693147) > 1e-3, (
        "the perturbation did not move the loss off ln(2), so this test cannot distinguish the sign of the log-ratio"
    )
