"""Gradient accumulation: the maths, and the engine's step accounting.

Every engine test in this tree ran with ``gradient_accumulation_steps = 1``, so
none of the accumulation arithmetic was exercised at all — yet getting it wrong
does not crash, it silently changes the effective learning rate:

- every micro-batch loss is divided by ``accum_steps``, so a FULL window
  accumulates one global-batch average gradient;
- when an epoch's tail is a PARTIAL window the engine rescales the gradient by
  ``accum_steps / accum_counter``, otherwise that step carries only
  ``counter/accum`` of the intended magnitude.

The property tested here is the one the design claims: an accumulated window
equals a single large batch of the same samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.regression_task import RegressionTask


def _two_equal_micro_batches():
    torch.manual_seed(0)
    model = nn.Linear(4, 2, bias=False)
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(5)
    inputs = [torch.randn(4, 4) for _ in range(2)]
    targets = [torch.randint(0, 2, (4,)) for _ in range(2)]
    return model, criterion, inputs, targets


def test_a_full_window_equals_one_large_batch():
    """accum=2 over equal micro-batches == the same 8 samples in one batch."""
    model, criterion, inputs, targets = _two_equal_micro_batches()
    accum = 2

    torch.manual_seed(1)
    model.zero_grad()
    for x, y in zip(inputs, targets, strict=True):
        criterion(model(x), y).div(accum).backward()
    accumulated = model.weight.grad.clone()

    torch.manual_seed(1)
    model.zero_grad()
    criterion(model(torch.cat(inputs)), torch.cat(targets)).backward()
    reference = model.weight.grad.clone()

    assert torch.allclose(accumulated, reference, atol=1e-6), (
        f"accumulated gradient diverges from the equivalent large batch by "
        f"{float((accumulated - reference).abs().max()):.3e}"
    )


def test_a_partial_window_is_rescaled_to_full_batch_magnitude():
    """The ``accum_steps / accum_counter`` rescale, and why it exists.

    A 2-of-4 window would otherwise carry exactly half the intended gradient —
    a silent effective-LR reduction on the epoch's tail step, which is what the
    engine's comment describes and this test pins.
    """
    model, criterion, inputs, targets = _two_equal_micro_batches()
    accum, counter = 4, 2

    torch.manual_seed(1)
    model.zero_grad()
    for x, y in zip(inputs, targets, strict=True):
        criterion(model(x), y).div(accum).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(accum / counter)
    rescaled = model.weight.grad.clone()

    torch.manual_seed(1)
    model.zero_grad()
    criterion(model(torch.cat(inputs)), torch.cat(targets)).backward()
    reference = model.weight.grad.clone()

    assert torch.allclose(rescaled, reference, atol=1e-6), (
        f"the rescaled partial window does not match the equivalent batch; "
        f"max|d| = {float((rescaled - reference).abs().max()):.3e}"
    )


def test_without_the_rescale_a_partial_window_is_half_magnitude():
    """Quantify what the rescale corrects, so its removal cannot look harmless."""
    model, criterion, inputs, targets = _two_equal_micro_batches()
    accum = 4  # only 2 of 4 micro-batches arrive: a 2/4 window

    torch.manual_seed(1)
    model.zero_grad()
    for x, y in zip(inputs, targets, strict=True):
        criterion(model(x), y).div(accum).backward()
    unrescaled = model.weight.grad.clone()

    torch.manual_seed(1)
    model.zero_grad()
    criterion(model(torch.cat(inputs)), torch.cat(targets)).backward()
    reference = model.weight.grad.clone()

    ratio = float(unrescaled.norm() / reference.norm())
    assert abs(ratio - 0.5) < 1e-3, f"expected the unscaled window to be half magnitude, got {ratio:.4f}"


def _engine(accum_steps: int, batch_size: int, num_samples: int, lr: float = 1e-3) -> TrainingEngine:
    config = Config(
        training=TrainingConfig(task="regression", epochs=1, batch_size=batch_size, num_samples=num_samples, lr=lr),
        model=ModelConfig(hidden_size=16, num_layers=1),
        optimization=OptimizationConfig(amp_dtype="auto", gradient_accumulation_steps=accum_steps, use_compile=False),
    )
    data_module = SyntheticDataModule(config)
    data_module.setup()
    return TrainingEngine(config, RegressionTask(config, data_module), rank=0, world_size=1, data_module=data_module)


def test_the_engine_takes_ceil_of_batches_over_accum_steps():
    """5 batches with accum=2 must produce 3 optimizer steps, not 2 and not 5.

    The third step is the partial window (1 of 2); dropping it silently trains on
    8 of 10 samples per epoch, and stepping per micro-batch trains 5 times on
    gradients 1/2 the intended size.
    """
    engine = _engine(accum_steps=2, batch_size=2, num_samples=10)

    n_batches = sum(1 for _ in engine._iter_training_batches())
    assert n_batches == 5, f"test precondition: expected 5 batches, got {n_batches}"

    engine._run_epoch(0)

    assert engine.global_step == 3, (
        f"5 batches with accum=2 advanced {engine.global_step} steps; expected ceil(5/2) = 3"
    )


def test_an_exact_window_needs_no_rescale():
    """4 batches with accum=2 is exactly two full windows: 2 steps, no tail."""
    engine = _engine(accum_steps=2, batch_size=2, num_samples=8)
    n_batches = sum(1 for _ in engine._iter_training_batches())
    assert n_batches == 4, f"test precondition: expected 4 batches, got {n_batches}"

    engine._run_epoch(0)

    assert engine.global_step == 2


def test_the_engine_rescales_a_partial_window_in_practice():
    """The engine must apply the rescale to the gradients it actually uses.

    The formula is tested in isolation above, but nothing would notice if the
    engine stopped calling it: the step COUNT is unaffected, so a step-only
    assertion passes either way. This captures each optimizer step's gradient
    norm and checks the epoch TAIL.

    The learning rate is tiny on purpose. Gradients are computed from the
    post-step weights, so with a normal lr the tail's 2x factor is masked by the
    weight change between steps — measured with lr=1e-3 the tail/full ratio moved
    only from 1.049 to 1.018 when the rescale was disabled, which is why a "tail
    is comparable to full" assertion cannot catch it. At lr=1e-12 the weights are
    effectively frozen and the factor is exact: 0.029475 with the rescale versus
    0.014738 without.
    """
    engine = _engine(accum_steps=2, batch_size=2, num_samples=10)
    for group in engine.optimizer.param_groups:
        group["lr"] = 1e-12

    norms: list[float] = []
    real_step = engine.optimizer.step

    def spy(*args, **kwargs):
        norms.append(sum(float(p.grad.norm()) for p in engine.model.parameters() if p.grad is not None))
        return real_step(*args, **kwargs)

    engine.optimizer.step = spy  # type: ignore[method-assign]
    engine._run_epoch(0)

    assert len(norms) == 3, f"expected 3 steps (2 full windows + tail), got {len(norms)}"
    tail_ratio = norms[-1] / norms[1]
    full_ratio = norms[0] / norms[1]
    # A full window and the (rescaled) partial window must be the same
    # magnitude. Without the accum/counter factor the tail is exactly half.
    assert abs(tail_ratio - full_ratio) < 0.15, (
        f"the epoch tail's gradient is {tail_ratio / full_ratio:.3f}x the full-window magnitude "
        f"(tail {norms[-1]:.6f}, full {norms[0]:.6f}); the partial-window rescale was not applied"
    )
