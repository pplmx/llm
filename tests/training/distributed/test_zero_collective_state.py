"""ZeRO shard-gathering envelope without a process group.

The multi-rank resume path is covered end-to-end by
``tests/e2e/test_zero_resume.py``, but that test spawns two processes and is
therefore ``heavy`` — it never runs in the CPU daily gate. The envelope
handling itself (``collective_state_dict`` → ``load_state_dict``) is pure
bookkeeping that needs no rendezvous at ``world_size=1``, so it is pinned here
where every push runs it.

The invariant that matters: gathering must be transparent to a single-rank job.
A ZeRO run that saves through the envelope must resume to exactly the state an
uninterrupted run would have reached — if the envelope lost the moments or the
loader silently ignored them, the parameters diverge on the next step.
"""

from __future__ import annotations

import pytest
import torch

from llm.training.distributed.zero import ZERO_RANK_STATES_KEY, ZeroOptimizer

STEPS = 6
STEPS_PRE = 3
STEPS_POST = 3
SEED = 7


def _build_model() -> torch.nn.Sequential:
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))


def _make_data() -> torch.Tensor:
    torch.manual_seed(999)
    return torch.randn(6, 4)


def _make_optimizer(model: torch.nn.Sequential, rank: int = 0, world_size: int = 1) -> ZeroOptimizer:
    # SGD with momentum: the state dict then carries real per-parameter
    # buffers, so a resume that dropped them would be visible in the weights.
    return ZeroOptimizer(
        lambda owned: torch.optim.SGD(owned, lr=0.05, momentum=0.9),
        model.parameters(),
        rank=rank,
        world_size=world_size,
    )


def _train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, steps: int) -> None:
    x = _make_data()
    for _ in range(steps):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()


@pytest.mark.quick
def test_single_rank_envelope_round_trips_the_whole_state():
    """``collective_state_dict`` at world_size=1 wraps exactly this rank's shard."""
    model = _build_model()
    optimizer = _make_optimizer(model)
    _train(model, optimizer, STEPS_PRE)

    envelope = optimizer.collective_state_dict()
    shards = envelope[ZERO_RANK_STATES_KEY]
    assert sorted(shards) == ["0"], f"a single-rank job must store one shard, got {sorted(shards)}"
    shard = shards["0"]
    assert shard["_zero_stage1"] is True
    assert shard["rank"] == 0
    assert shard["global_indices"] == list(range(len(list(model.parameters()))))
    assert shard["inner"]["state"], "SGD momentum buffers must be present before this test can prove anything"


@pytest.mark.quick
def test_resume_through_the_envelope_matches_an_uninterrupted_run():
    """Save via the envelope, reload into a fresh optimizer, land where a full
    run would have: the moments (not just the weights) must survive."""
    # Reference: one uninterrupted run.
    torch.manual_seed(SEED)
    reference_model = _build_model()
    reference_optimizer = _make_optimizer(reference_model)
    _train(reference_model, reference_optimizer, STEPS)

    # Interrupted: STEPS_PRE, save, rebuild from scratch, reload, STEPS_POST.
    model = _build_model()
    optimizer = _make_optimizer(model)
    _train(model, optimizer, STEPS_PRE)
    envelope = optimizer.collective_state_dict()
    saved_weights = [p.detach().clone() for p in model.parameters()]

    torch.manual_seed(SEED)
    resumed_model = _build_model()
    resumed_optimizer = _make_optimizer(resumed_model)
    resumed_optimizer.load_state_dict(envelope)
    # The weights themselves live in the model checkpoint, not the optimizer
    # payload; restore them the way the checkpoint manager does.
    with torch.no_grad():
        for param, weight in zip(resumed_model.parameters(), saved_weights, strict=True):
            param.copy_(weight)
    _train(resumed_model, resumed_optimizer, STEPS_POST)

    for index, (got, want) in enumerate(zip(resumed_model.parameters(), reference_model.parameters(), strict=True)):
        assert torch.equal(got, want), f"parameter {index} diverged after resume — the moments were dropped"


@pytest.mark.quick
def test_bare_shard_still_loads():
    """The per-rank shard remains a valid input (hand-built / single-rank callers)."""
    model = _build_model()
    optimizer = _make_optimizer(model)
    _train(model, optimizer, STEPS_PRE)
    bare = optimizer.state_dict()
    assert bare["inner"]["state"], "SGD momentum buffers must be present before this test can prove anything"

    torch.manual_seed(SEED)
    other = _build_model()
    other_optimizer = _make_optimizer(other)
    other_optimizer.load_state_dict(bare)
    # ``load_state_dict`` rebuilds the inner optimizer's groups/state, so the
    # wrapper must re-adopt them rather than keep pointing at the fresh,
    # empty ones it was constructed with.
    assert other_optimizer.state, "a bare shard must still restore the optimizer state"
    assert other_optimizer.state is other_optimizer.optimizer.state


@pytest.mark.quick
@pytest.mark.parametrize("save_first", [False, True], ids=["fresh", "after_resume"])
def test_scheduler_updates_reach_the_inner_optimizer(save_first):
    """A resumed run must keep honouring its LR schedule.

    ``Optimizer.load_state_dict`` rebuilds ``param_groups`` instead of mutating
    them, so before this was fixed a resume left the wrapper aliasing orphaned
    dicts: the engine's ``LRScheduler`` wrote each step's learning rate into
    structures the optimizer never read, training silently kept the pre-resume
    LR, and the logs reported the scheduled one. Parametrised against the fresh
    (never-loaded) case so the resume path is compared to correct behaviour
    rather than to itself.
    """
    model = _build_model()
    optimizer = _make_optimizer(model)
    _train(model, optimizer, STEPS_PRE)
    if save_first:
        optimizer.load_state_dict(optimizer.collective_state_dict())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler.step()

    scheduled_lr = optimizer.param_groups[0]["lr"]
    stepped_lr = optimizer.optimizer.param_groups[0]["lr"]
    assert stepped_lr == scheduled_lr, (
        f"the scheduled LR {scheduled_lr} never reached the optimizer that steps (it holds {stepped_lr})"
    )
    assert scheduler.get_last_lr()[0] == stepped_lr
