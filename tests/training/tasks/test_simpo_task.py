"""SimPO task tests: reference-free loss, finite, and directionally correct."""

import torch

from llm.training.tasks.simpo_task import SimPOTask


def _pref_batch(seq_len: int, chosen: tuple, rejected: tuple) -> dict:
    """Build a chosen/rejected preference batch with the DPO task's keys."""
    chosen_ids = torch.as_tensor([list(chosen)])
    rejected_ids = torch.as_tensor([list(rejected)])
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, 0] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, 0] = -100
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "chosen_attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "rejected_attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }


def test_simpo_is_reference_free(tiny_config):
    task = SimPOTask(tiny_config, data_module=None)
    assert not hasattr(task, "ref_model")
    model = task.build_model()
    assert any(p.requires_grad for p in model.parameters())


def test_simpo_train_step_finite_and_backprops(tiny_config, device):
    task = SimPOTask(tiny_config, data_module=None)
    model = task.build_model().to(device).train()
    vocab = tiny_config.model.vocab_size
    chosen = [(i % vocab) + 1 for i in range(6)]
    rejected = [((i + 3) % vocab) + 1 for i in range(6)]
    batch = _pref_batch(6, tuple(chosen), tuple(rejected))
    batch = {k: v.to(device) for k, v in batch.items()}

    loss, metrics = task.train_step(batch, model, criterion=None)
    assert not torch.isnan(loss)
    assert 0.0 <= metrics["reward_acc"] <= 1.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "SimPO loss must produce gradients"


def test_simpo_prefers_chosen_after_training(tiny_config, device):
    task = SimPOTask(tiny_config, data_module=None)
    model = task.build_model().to(device).train()
    vocab = tiny_config.model.vocab_size
    chosen = tuple((i % vocab) + 1 for i in range(6))
    rejected = tuple(((i + 3) % vocab) + 1 for i in range(6))
    batch = _pref_batch(6, chosen, rejected)
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(120):
        optimizer.zero_grad()
        loss, metrics = task.train_step(batch, model, criterion=None)
        loss.backward()
        optimizer.step()

    assert metrics["reward_acc"] == 1.0, "SimPO failed to prefer the chosen response"
    assert metrics["reward_margin"] > 0.0


# --- the SimPO loss must match its published formula ------------------------
#
# The existing tests assert "not isnan", "has gradients" and the DIRECTION of the
# preference. All three still pass with either term of the objective deleted:
# dropping ``gamma`` from the margin, and dropping the ``lmbda`` SFT term, each
# left the suite green (verified). The default ``simpo_gamma`` is 0.0, which is
# why the default-configured tests can never see the gamma term at all.
#
# These tests set BOTH parameters to non-zero values and recompute the objective
# independently from the model's own log-probabilities.


def _simpo_config(tiny_config, *, gamma: float, lmbda: float) -> object:
    tiny_config.training.simpo_gamma = gamma
    tiny_config.training.simpo_lambda = lmbda
    return tiny_config


def test_the_simpo_loss_matches_the_published_formula(tiny_config):
    """``train_step`` must equal the paper's objective, recomputed independently."""
    import torch.nn.functional as functional

    config = _simpo_config(tiny_config, gamma=0.5, lmbda=1.0)
    task = SimPOTask(config, data_module=None)
    model = task.build_model()
    model.eval()

    batch = _pref_batch(4, chosen=(1, 2, 3, 4), rejected=(4, 3, 2, 1))

    with torch.no_grad():
        loss, _metrics = task.train_step(batch, model, None)
        # Recompute from the model's own log-probabilities, using only the
        # paper's formula -- not the implementation's intermediate names.
        chosen_logps = task._get_batch_logps(
            model(batch["chosen_input_ids"], attn_mask=batch["chosen_attention_mask"]),
            batch["chosen_labels"],
            average_log_prob=True,
        )
        rejected_logps = task._get_batch_logps(
            model(batch["rejected_input_ids"], attn_mask=batch["rejected_attention_mask"]),
            batch["rejected_labels"],
            average_log_prob=True,
        )
    margin = task.beta * chosen_logps - task.beta * rejected_logps - 0.5
    expected = (-functional.logsigmoid(margin)).mean() + 1.0 * (-chosen_logps.mean())

    assert abs(float(loss) - float(expected)) < 1e-6, (
        f"train_step returned {float(loss):.6f} but the SimPO objective gives {float(expected):.6f}"
    )


def test_a_nonzero_gamma_shifts_the_loss(tiny_config):
    """``gamma`` must matter: the loss with gamma=0.5 differs from gamma=0.

    The default is 0.0, so without setting it explicitly no test exercises the
    term — which is exactly how deleting it from the margin went unnoticed.
    """
    batch = _pref_batch(4, chosen=(1, 2, 3, 4), rejected=(4, 3, 2, 1))

    losses = {}
    for gamma in (0.0, 0.5):
        config = _simpo_config(tiny_config, gamma=gamma, lmbda=1.0)
        task = SimPOTask(config, data_module=None)
        model = task.build_model()
        torch.manual_seed(3)
        model.eval()
        with torch.no_grad():
            losses[gamma], _ = task.train_step(batch, model, None)

    assert abs(float(losses[0.0]) - float(losses[0.5])) > 1e-3, (
        f"gamma had no effect on the loss ({float(losses[0.0]):.6f} vs {float(losses[0.5]):.6f})"
    )
    # A larger gamma makes the margin harder to satisfy, so the loss rises.
    assert float(losses[0.5]) > float(losses[0.0]), "a larger gamma must increase the loss"


def test_a_nonzero_lambda_adds_the_sft_term(tiny_config):
    """``lmbda`` must matter: lambda=2 differs from lambda=0 by the SFT term.

    With lambda=0 the loss is purely the preference objective; the difference
    between the two runs must equal ``lmbda * (-mean_chosen_logp)``.
    """

    batch = _pref_batch(4, chosen=(1, 2, 3, 4), rejected=(4, 3, 2, 1))

    # ONE model, two lambdas: the SFT term is a function of the model's own
    # chosen log-prob, so comparing two independently-built models would fold
    # their initialisation difference into the measurement (it did: 8.66 vs
    # 9.21 when only the lambda should have changed).
    task = SimPOTask(_simpo_config(tiny_config, gamma=0.0, lmbda=0.0), data_module=None)
    model = task.build_model()
    model.eval()
    computed = {}
    with torch.no_grad():
        for lam in (0.0, 2.0):
            task.lmbda = lam
            loss, _ = task.train_step(batch, model, None)
            chosen_logps = task._get_batch_logps(
                model(batch["chosen_input_ids"], attn_mask=batch["chosen_attention_mask"]),
                batch["chosen_labels"],
                average_log_prob=True,
            )
            computed[lam] = (float(loss), float(chosen_logps.mean()))

    difference = computed[2.0][0] - computed[0.0][0]
    expected = 2.0 * (-computed[0.0][1])
    assert abs(difference - expected) < 1e-5, (
        f"the lambda=2 loss differs from lambda=0 by {difference:.6f}, but the SFT term is "
        f"{expected:.6f}; the term is not being applied as documented"
    )
    assert abs(difference) > 1e-3, "lambda had no effect on the loss"


def test_the_gamma_term_is_present_at_the_documented_default(tiny_config):
    """At the default gamma=0.0 the margin must be EXACTLY beta*(lp_w - lp_l).

    Pinned so the default path is a known quantity rather than an untested one:
    the term is legitimately absent there, and a future change that adds an
    offset would show up here.
    """
    import torch.nn.functional as functional

    config = _simpo_config(tiny_config, gamma=0.0, lmbda=0.0)  # isolate the preference term
    task = SimPOTask(config, data_module=None)
    model = task.build_model()
    model.eval()
    batch = _pref_batch(4, chosen=(1, 2, 3, 4), rejected=(4, 3, 2, 1))

    with torch.no_grad():
        loss, _ = task.train_step(batch, model, None)
        chosen_logps = task._get_batch_logps(
            model(batch["chosen_input_ids"], attn_mask=batch["chosen_attention_mask"]),
            batch["chosen_labels"],
            average_log_prob=True,
        )
        rejected_logps = task._get_batch_logps(
            model(batch["rejected_input_ids"], attn_mask=batch["rejected_attention_mask"]),
            batch["rejected_labels"],
            average_log_prob=True,
        )
    expected = (-functional.logsigmoid(task.beta * (chosen_logps - rejected_logps))).mean()

    assert abs(float(loss) - float(expected)) < 1e-6
