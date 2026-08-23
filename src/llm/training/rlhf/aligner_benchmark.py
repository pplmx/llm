"""DPO vs PPO/RLHF vs GRPO alignment benchmark on a shared preference task (TASK-233).

A CPU-verifiable harness that runs ALL aligners against the same synthetic
preference signal — a judge whose "good" response ends in a target token — and
reports comparable metrics:

- **DPO** (off-policy, preference pairs): ``preference_fraction`` =
  ``P(chosen-logp > rejected-logp)`` on the shared preference set, where the
  chosen response ends in the target token and the rejected one does not.
- **PPO** (on-policy, rollout): mean rollout reward plus the same
  ``preference_fraction`` evaluated on the shared set.
- **GRPO** (group-relative, fixed synthetic groups): the group-reward fraction
  (fraction of groups whose most-likely response is the rewarded target) plus
  the shared ``preference_fraction``.

This is a head-to-head benchmark, not a claim that one aligner is always
better. On this small CPU setup the off-policy DPO path converges stably
(preference_fraction 0 -> ~1) within a modest step budget, whereas on-policy
PPO with rollout and a sparse end-token reward remains noisy and does not
reach the same convergence in an equivalent budget — a real, documented
observation (see ``docs/guides/dpo_vs_ppo.md``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TargetTokenReward(nn.Module):
    """Rule-based judge reward: 1.0 iff the last real token equals the target.

    Mirrors :class:`llm.training.rlhf.aifeedback.TargetTokenJudge` as an
    ``nn.Module`` so it can act as PPO's reward model. ``forward`` follows the
    reward-model call contract ``(input_ids: [B, L], attention_mask: [B, L]) ->
    rewards: [B]``.
    """

    def __init__(self, target_token: int = 0) -> None:
        super().__init__()
        self._target = int(target_token)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is not None:
            # Last NON-padding token (length-1 sequences stay at index 0).
            last_idx = attention_mask.sum(dim=1, keepdim=True).long().clamp(min=1) - 1
            last = input_ids.gather(1, last_idx)
        else:
            last = input_ids[:, -1:]
        target = torch.as_tensor(self._target, dtype=torch.long, device=input_ids.device)
        hit = torch.eq(last.long(), target).to(torch.float32)
        return hit.squeeze(-1)


def _seq_log_prob(model: nn.Module, ids: torch.Tensor) -> float:
    """Sum of per-token log-probs ``model`` assigns to a ``[L]`` response."""
    model.eval()
    with torch.no_grad():
        logits = model(ids.unsqueeze(0))  # [1, L, V]
    lp = functional.log_softmax(logits.float(), dim=-1)
    resp_logp = lp[:, :-1].gather(-1, ids[1:].unsqueeze(0).unsqueeze(-1)).squeeze(-1).sum(-1)
    return float(resp_logp.item())


def preference_fraction(model: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor) -> float:
    """Fraction of ``(chosen, rejected)`` pairs where the model prefers chosen.

    ``chosen`` / ``rejected`` are ``[B, L]``. A pair counts if the model's
    cumulative log-prob of ``chosen`` exceeds that of ``rejected``.
    """
    if chosen.ndim != 2 or rejected.ndim != 2 or chosen.shape != rejected.shape:
        raise ValueError(
            f"preference_fraction expects equal [B, L] tensors, got {tuple(chosen.shape)} vs {tuple(rejected.shape)}"
        )
    hits = 0
    for i in range(chosen.size(0)):
        if _seq_log_prob(model, chosen[i]) > _seq_log_prob(model, rejected[i]):
            hits += 1
    return hits / chosen.size(0)


def _snapshot_reference(model: nn.Module) -> nn.Module:
    """Return a frozen deep-copy of ``model`` (the initial policy)."""
    import copy

    ref = copy.deepcopy(model)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    return ref


def _reference_kl(policy: nn.Module, reference: nn.Module, sequences: torch.Tensor) -> float:
    """Mean per-sequence KL of the policy vs a frozen reference over ``seqs``.

    ``sequences`` is ``[B, L]``. KL is summed over every predicted position
    (the next-token distribution at each prefix) and averaged over the batch —
    the standard reward-over-optimization / drift diagnostic.
    """
    policy.eval()
    reference.eval()
    with torch.no_grad():
        p_logits = policy(sequences)
        r_logits = reference(sequences)
    p = functional.softmax(p_logits.float(), dim=-1)
    r_log = functional.log_softmax(r_logits.float(), dim=-1)
    p_pos = p[:, :-1]
    r_pos = r_log[:, :-1]
    kl = (p_pos * (p_pos.clamp_min(1e-12).log() - r_pos)).sum(-1)  # [B, L-1]
    return float(kl.sum(-1).mean().item())


def _dpo_config(base: Any, target_token: int) -> tuple[Any, Any]:
    from llm.data.modules.aifeedback import AIFeedbackDataModule
    from llm.training.rlhf.aifeedback import TargetTokenJudge
    from llm.training.tasks.dpo_task import DPOTask

    module = AIFeedbackDataModule(base, judge=TargetTokenJudge(target_token=target_token))
    module.prepare_data()
    module.setup()
    task = DPOTask(base, module)
    return module, task


def run_dpo(
    config: Any, epochs: int, target_token: int = 1, eval_sequences: torch.Tensor | None = None
) -> dict[str, Any]:
    """Train DPO on the synthetic judge-labeled pairs; return the trajectory.

    Uses :class:`AIFeedbackDataModule` (chosen ends in the target token) +
    the existing ``--task dpo`` loop. Reports the shared preference-fraction
    trajectory, the DPO loss trajectory, and the policy's KL from a frozen
    reference over training (reward-over-optimization diagnostic).
    """
    from llm.training.core.engine import TrainingEngine

    module, task = _dpo_config(config, target_token)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
    reference = _snapshot_reference(engine.model)
    eval_seqs = eval_sequences if eval_sequences is not None else module.chosen

    pref_traj: list[float] = []
    loss_traj: list[float] = []
    kl_traj: list[float] = []
    for epoch in range(epochs):
        loss = engine._run_epoch(epoch)
        loss_traj.append(float(loss))
        pref_traj.append(preference_fraction(engine.model, module.chosen, module.rejected))
        kl_traj.append(_reference_kl(engine.model, reference, eval_seqs))

    return {
        "preference_fraction_trajectory": pref_traj,
        "final_preference_fraction": pref_traj[-1],
        "dpo_loss_trajectory": loss_traj,
        "reference_kl_trajectory": kl_traj,
        "final_reference_kl": kl_traj[-1],
    }


def _grpo_group_reward_fraction(model: nn.Module, module: Any) -> float:
    """Fraction of GRPO groups whose most-likely response is the rewarded target."""
    model.eval()
    with torch.no_grad():
        logits = model(module.response_tokens)
    lp = functional.log_softmax(logits.float(), dim=-1)
    lp = lp.gather(-1, module.response_tokens.unsqueeze(-1)).squeeze(-1).sum(-1)
    lp = lp.reshape(-1, module.group_size)
    return float((lp.argmax(-1) == 0).float().mean().item())


def run_grpo(config: Any, epochs: int, eval_sequences: torch.Tensor | None = None) -> dict[str, Any]:
    """Run group-relative GRPO on its synthetic group task; return trajectories.

    Uses the existing :class:`GRPODataModule` + ``--task grpo`` loop (the group
    whose first response is the all-zero target is rewarded 1.0). Reports the
    group-reward fraction, GRPO loss, and reference-KL trajectories. Returns the
    policy too so the comparison can evaluate it on the shared preference set.
    """
    from llm.data.modules.grpo import GRPODataModule
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.grpo_task import GRPOTask

    module = GRPODataModule(config)
    module.prepare_data()
    module.setup()
    task = GRPOTask(config, module)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
    reference = _snapshot_reference(engine.model)
    eval_seqs = eval_sequences if eval_sequences is not None else module.response_tokens

    group_traj: list[float] = []
    loss_traj: list[float] = []
    kl_traj: list[float] = []
    for epoch in range(epochs):
        loss = engine._run_epoch(epoch)
        loss_traj.append(float(loss))
        group_traj.append(_grpo_group_reward_fraction(engine.model, module))
        kl_traj.append(_reference_kl(engine.model, reference, eval_seqs))

    return {
        "group_reward_fraction_trajectory": group_traj,
        "final_group_reward_fraction": group_traj[-1],
        "grpo_loss_trajectory": loss_traj,
        "reference_kl_trajectory": kl_traj,
        "final_reference_kl": kl_traj[-1],
        "policy": engine.model,
    }


def run_ppo(
    config: Any,
    steps: int,
    prompts: list[str],
    target_token: int = 1,
    eval_sequences: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run on-policy PPO with the target-token judge reward.

    Builds a :class:`PPOTask` whose reward model is the rule-based
    :class:`TargetTokenReward` (the same judge signal DPO consumes as
    preference pairs), rolls out ``steps`` optimizer updates, and returns the
    mean-reward trajectory and (when ``eval_sequences`` is given) the policy's
    reference-KL after each step.
    """
    import json
    import tempfile
    from pathlib import Path

    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.ppo_task import PPOTask

    torch.manual_seed(0)
    from llm.data.modules.prompt import PromptDataModule

    if not prompts:
        raise ValueError("run_ppo requires at least one prompt")
    if not getattr(config.data, "dataset_path", None):
        config.data.dataset_path = str(Path(tempfile.mkdtemp(prefix="ppo_bench_")) / "prompts.jsonl")
    with Path(config.data.dataset_path).open("w", encoding="utf-8") as fp:
        for prompt in prompts:
            fp.write(json.dumps({"prompt": prompt}) + "\n")
    # Force a SHORT rollout: the config default response length is far too
    # long for a tiny-CPU benchmark and would overrun the model's positional
    # cache during autoregression (prompt + response must fit max_seq_len).
    config.ppo.response_max_len = min(3, max(1, int(config.model.max_seq_len) - 1))

    module = PromptDataModule(config)
    module.prepare_data()
    module.setup()

    task: PPOTask = PPOTask(config, module)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
    task.prepare_training(engine)
    trainer = task.ppo_trainer
    if trainer is None:
        raise RuntimeError("PPO trainer was not initialized by PPOTask.prepare_training()")
    # Swap in the rule-based judge reward (the benchmark's shared preference
    # signal) instead of the freshly-initialized learned reward head.
    trainer.reward_model = TargetTokenReward(target_token=target_token)
    reference = _snapshot_reference(engine.model)

    reward_traj: list[float] = []
    kl_traj: list[float] = []
    for _ in range(max(steps, 1)):
        metrics = trainer.train_step(prompts)
        reward_traj.append(float(metrics.get("reward_mean", 0.0)))
        if eval_sequences is not None:
            kl_traj.append(_reference_kl(engine.model, reference, eval_sequences))

    result: dict[str, Any] = {
        "mean_reward_trajectory": reward_traj,
        "final_mean_reward": reward_traj[-1],
        "policy": engine.model,
    }
    if eval_sequences is not None:
        result["reference_kl_trajectory"] = kl_traj
        result["final_reference_kl"] = kl_traj[-1]
    return result


def compare_dpo_vs_ppo(
    config: Any,
    dpo_epochs: int,
    ppo_steps: int,
    prompts: list[str],
    target_token: int = 0,
    grpo_epochs: int = 40,
) -> dict[str, Any]:
    """Run DPO, PPO and GRPO on a shared target-token preference task.

    The shared judge target is ``target_token`` (default 0, matching GRPO's
    built-in group target). Returns a dict with the three aligner results, each
    evaluated on the same shared preference set, plus a short summary.
    """
    from llm.data.modules.aifeedback import AIFeedbackDataModule
    from llm.training.rlhf.aifeedback import TargetTokenJudge

    shared = AIFeedbackDataModule(config, judge=TargetTokenJudge(target_token=target_token))
    shared.prepare_data()
    shared.setup()
    shared_eval = shared.chosen

    dpo_result = run_dpo(config, dpo_epochs, target_token=target_token, eval_sequences=shared_eval)

    ppo = run_ppo(config, ppo_steps, prompts, target_token=target_token, eval_sequences=shared_eval)
    ppo_policy = ppo.pop("policy")
    ppo_pref = preference_fraction(ppo_policy, shared.chosen, shared.rejected)
    ppo_result = {**ppo, "final_preference_fraction": ppo_pref}

    grpo = run_grpo(config, grpo_epochs, eval_sequences=shared_eval)
    grpo_policy = grpo.pop("policy")
    grpo_pref = preference_fraction(grpo_policy, shared.chosen, shared.rejected)
    grpo_result = {**grpo, "final_preference_fraction": grpo_pref}
    grpo_first = float(grpo["group_reward_fraction_trajectory"][0])
    grpo_final = float(grpo["final_group_reward_fraction"])
    grpo_kl_first = float(grpo["reference_kl_trajectory"][0])
    grpo_kl_final = float(grpo["final_reference_kl"])

    summary = (
        f"DPO pref-fraction {dpo_result['preference_fraction_trajectory'][0]:.2f} -> "
        f"{dpo_result['final_preference_fraction']:.2f} over {dpo_epochs} epochs; "
        f"PPO mean reward {ppo_result['final_mean_reward']:.3f}, "
        f"pref-fraction {ppo_result['final_preference_fraction']:.2f} after {ppo_steps} rollout step(s); "
        f"GRPO group-reward {grpo_first:.2f} -> {grpo_final:.2f} over {grpo_epochs} epochs; "
        f"reference-KL DPO {dpo_result['final_reference_kl']:.3f} / "
        f"PPO {ppo_result['final_reference_kl']:.3f} / "
        f"GRPO {grpo_kl_first:.3f} -> {grpo_kl_final:.3f}."
    )
    return {"dpo": dpo_result, "ppo": ppo_result, "grpo": grpo_result, "summary": summary}
