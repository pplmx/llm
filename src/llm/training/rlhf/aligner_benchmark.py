"""DPO vs PPO/RLHF alignment benchmark on a shared synthetic preference task (TASK-232).

A CPU-verifiable harness that runs BOTH aligners against the same synthetic
preference signal — a judge whose "good" response ends in a target token — and
reports comparable metrics:

- **DPO** (off-policy, preference pairs): ``preference_fraction`` =
  ``P(chosen-logp > rejected-logp)`` on the shared preference set, where the
  chosen response ends in the target token and the rejected one does not.
- **PPO** (on-policy, rollout): mean rollout reward plus the same
  ``preference_fraction`` evaluated on the shared set.

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


def _dpo_config(base: Any, target_token: int) -> tuple[Any, Any]:
    from llm.data.modules.aifeedback import AIFeedbackDataModule
    from llm.training.rlhf.aifeedback import TargetTokenJudge
    from llm.training.tasks.dpo_task import DPOTask

    module = AIFeedbackDataModule(base, judge=TargetTokenJudge(target_token=target_token))
    module.prepare_data()
    module.setup()
    task = DPOTask(base, module)
    return module, task


def run_dpo(config: Any, epochs: int, target_token: int = 1) -> dict[str, Any]:
    """Train DPO on the synthetic judge-labeled pairs; return the trajectory.

    Uses :class:`AIFeedbackDataModule` (chosen ends in the target token) +
    the existing ``--task dpo`` loop. Reports the shared preference-fraction
    trajectory and the DPO loss trajectory.
    """
    from llm.training.core.engine import TrainingEngine

    module, task = _dpo_config(config, target_token)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)

    pref_traj: list[float] = []
    loss_traj: list[float] = []
    for epoch in range(epochs):
        loss = engine._run_epoch(epoch)
        loss_traj.append(float(loss))
        pref_traj.append(preference_fraction(engine.model, module.chosen, module.rejected))

    return {
        "preference_fraction_trajectory": pref_traj,
        "final_preference_fraction": pref_traj[-1],
        "dpo_loss_trajectory": loss_traj,
    }


def run_ppo(config: Any, steps: int, prompts: list[str], target_token: int = 1) -> dict[str, Any]:
    """Run on-policy PPO with the target-token judge reward.

    Builds a :class:`PPOTask` whose reward model is the rule-based
    :class:`TargetTokenReward` (the same judge signal DPO consumes as
    preference pairs), rolls out ``steps`` optimizer updates, and returns the
    mean-reward trajectory.
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

    reward_traj: list[float] = []
    for _ in range(max(steps, 1)):
        metrics = trainer.train_step(prompts)
        reward_traj.append(float(metrics.get("reward_mean", 0.0)))

    return {
        "mean_reward_trajectory": reward_traj,
        "final_mean_reward": reward_traj[-1],
        "policy": engine.model,
    }


def compare_dpo_vs_ppo(
    config: Any, dpo_epochs: int, ppo_steps: int, prompts: list[str], target_token: int = 1
) -> dict[str, Any]:
    """Run both aligners on the shared target-token preference task.

    Returns a dict with the DPO result, the PPO result (evaluated on the same
    shared preference set), and a short human-readable summary.
    """
    from llm.data.modules.aifeedback import AIFeedbackDataModule
    from llm.training.rlhf.aifeedback import TargetTokenJudge

    shared = AIFeedbackDataModule(config, judge=TargetTokenJudge(target_token=target_token))
    shared.prepare_data()
    shared.setup()

    dpo_result = run_dpo(config, dpo_epochs, target_token=target_token)

    ppo = run_ppo(config, ppo_steps, prompts, target_token=target_token)
    ppo_policy = ppo.pop("policy")
    ppo_pref = preference_fraction(ppo_policy, shared.chosen, shared.rejected)
    ppo_result = {**ppo, "final_preference_fraction": ppo_pref}

    summary = (
        f"DPO pref-fraction {dpo_result['preference_fraction_trajectory'][0]:.2f} -> "
        f"{dpo_result['final_preference_fraction']:.2f} over {dpo_epochs} epochs; "
        f"PPO mean reward {ppo_result['final_mean_reward']:.3f}, "
        f"pref-fraction {ppo_result['final_preference_fraction']:.2f} after {ppo_steps} rollout step(s)."
    )
    return {"dpo": dpo_result, "ppo": ppo_result, "summary": summary}
