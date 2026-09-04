"""
PPO Trainer for RLHfunctional.

Implements Proximal Policy Optimization for language model alignment.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from llm.generation.sampling import sampling_probs as _sampling_probs
from llm.models.decoder import DecoderModel
from llm.training.core.config import PPOConfig
from llm.training.rlhf.rollout_buffer import RolloutBatch, RolloutBuffer
from llm.training.rlhf.value_model import ValueModel

logger = logging.getLogger(__name__)


def _normalize_eos_ids(eos_token_id: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    """Return the tokenizer's EOS id(s) as a plain tuple of ints.

    HF tokenizers can expose ``eos_token_id`` as a list/sequence; a raw
    ``next_token.item() == eos_id`` int-vs-list comparison silently never
    matches, so every rollout runs to ``response_max_len`` with post-EOS junk
    folded into the training signal (RIL ISS-116/334).
    """
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, (list, tuple)):
        return tuple(int(t) for t in eos_token_id)
    return (int(eos_token_id),)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHfunctional.

    Trains a policy model using PPO with rewards from a reward model.
    Optionally uses a reference model for KL divergence penalty.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: Any,
        config: PPOConfig,
        ref_model: nn.Module | None = None,
        value_model: ValueModel | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: The language model to train (policy).
            reward_model: Frozen reward model for scoring responses.
            tokenizer: Tokenizer for encoding/decoding.
            config: PPO configuration.
            ref_model: Frozen reference model for KL penalty (optional).
            value_model: Value function model (optional, uses policy if None).
            device: Device to run training on.
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)

        # Reference model (frozen copy of initial policy)
        if config.use_ref_model:
            if ref_model is not None:
                self.ref_model = ref_model
            else:
                # Create a frozen copy
                self.ref_model = self._create_ref_model()
        else:
            self.ref_model = None

        # Value model (separate critic when value_coef > 0)
        if value_model is not None:
            self.value_model = value_model
        elif config.value_coef > 0:
            self.value_model = self._create_value_model()
        else:
            self.value_model = None

        # Move models to device
        self.policy.to(self.device)
        self.reward_model.to(self.device)
        self.reward_model.eval()
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            # RIL ISS-334 (mirrors the critic's ISS-173 handling): the frozen
            # reference is reconstructed from the policy (which the engine
            # broadcasts before prepare_training), but broadcast it explicitly
            # too — if the copy ran before a rank-0 sync, multi-GPU ranks
            # would diverge in the KL-penalty target and the averaged gradients
            # would fit no single reference.
            from llm.training.core.distributed import broadcast_parameters

            broadcast_parameters(self.ref_model)
        if self.value_model is not None:
            self.value_model.to(self.device)
            # RIL ISS-173: the critic deep-copies the policy (which the engine
            # broadcasts before prepare_training), but broadcast it explicitly
            # too — if the copy ran before a rank-0 sync, multi-GPU ranks
            # would start with divergent critic weights and GAE bootstraps.
            from llm.training.core.distributed import broadcast_parameters

            broadcast_parameters(self.value_model)

        # Optimizers
        policy_lr = config.policy_lr or 1e-5
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=policy_lr,
        )
        if self.value_model is not None:
            value_lr = config.value_lr or policy_lr
            self.value_optimizer = AdamW(
                self.value_model.parameters(),
                lr=value_lr,
            )
        else:
            self.value_optimizer = None

        # Rollout buffer
        self.buffer = RolloutBuffer(
            gae_lambda=config.gae_lambda,
            gamma=config.gamma,
            normalize_advantages=config.normalize_advantages,
        )

        # Training stats
        self.global_step = 0
        self.kl_ctl = config.kl_coef

    @staticmethod
    def _snapshot_state(state: Any) -> Any:
        """Deep-copy checkpoint payloads so later in-place updates do not alias saved tensors."""
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, dict):
            return {key: PPOTrainer._snapshot_state(value) for key, value in state.items()}
        if isinstance(state, list):
            return [PPOTrainer._snapshot_state(value) for value in state]
        return state

    def get_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"global_step": self.global_step}
        if self.value_model is not None:
            state["value_model"] = self._snapshot_state(self.value_model.state_dict())
        if self.value_optimizer is not None:
            state["value_optimizer"] = self._snapshot_state(self.value_optimizer.state_dict())
        if self.ref_model is not None:
            state["ref_model"] = self._snapshot_state(self.ref_model.state_dict())
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.global_step = int(state.get("global_step", self.global_step))
        if self.value_model is not None and "value_model" in state:
            self.value_model.load_state_dict(state["value_model"])
            # A persisted critic was restored — keep it (the critic's base at
            # the moment the checkpoint was saved) instead of re-syncing to
            # the now-moved policy in on_checkpoint_loaded (RIL ISS-175).
            self._value_restored_from_ckpt = True
        if self.value_optimizer is not None and "value_optimizer" in state:
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        if self.ref_model is not None and "ref_model" in state:
            self.ref_model.load_state_dict(state["ref_model"])
            # A persisted reference was restored — keep THIS base (the
            # original at the moment the checkpoint was saved) instead of
            # re-syncing to the now-moved policy in on_checkpoint_loaded.
            self._ref_restored_from_ckpt = True

    def on_checkpoint_loaded(self, model: nn.Module) -> None:
        """Align the frozen companions with a checkpoint-loaded base policy.

        Called by the engine right after ``load_checkpoint`` applies resumed
        weights to the policy (RIL round-60 deep-dive Finding 1, same contract
        as ``DPOTask.on_checkpoint_loaded``).

        - Resuming from an SFT/base checkpoint (no persisted ``ref_model`` /
          ``value_model`` in extra_state): the reference AND the critic must
          equal the loaded policy — that IS the base the policy is
          regularised / bootstrapped against. Both are deep-copied inside
          ``prepare_training`` BEFORE the engine loads the checkpoint, so
          without this hook the KL penalty and the GAE value bootstrap use
          stale (pre-resume) models (RIL round-60 Finding 1 + round-62
          surface-B Finding 3 / ISS-175).
        - Resuming from a mid-PPO checkpoint: ``load_checkpoint_state`` already
          restored the original companion weights; keep them.
        """
        if self.ref_model is not None and not getattr(self, "_ref_restored_from_ckpt", False):
            self.ref_model.load_state_dict(model.state_dict())
        # The critic's base is deep-copied from the policy at build time too;
        # re-align it with the loaded policy unless the checkpoint carried a
        # persisted value model (mid-PPO resume), which is authoritative.
        if self.value_model is not None and not getattr(self, "_value_restored_from_ckpt", False):
            self.value_model.base_model.load_state_dict(model.state_dict())

    def _sync_value_grads(self) -> None:
        """All-reduce the critic's gradients across ranks (RIL ISS-173).

        No-op when not running distributed. Mirrors what DDP does for the
        policy: gradients are summed then divided by the world size so every
        rank's ``value_optimizer.step()`` applies the SAME mean gradient and
        the critic stays a single shared model across ranks.
        """
        if self.value_model is None:
            return
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return
        world = torch.distributed.get_world_size()
        for param in self.value_model.parameters():
            if param.grad is None:
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            param.grad.div_(world)

    def _policy_base(self) -> nn.Module:
        """The bare module behind a possibly DDP-wrapped policy.

        ``self.policy`` keeps its ``DistributedDataParallel`` wrapper for
        training (so gradients all-reduce on multi-GPU); read-only copies
        (reference model, critic base) must deep-copy the *unwrapped*
        module — deep-copying a DDP wrapper drags along its process-group
        references and the isinstance(DecoderModel) guards below would
        otherwise fail (RIL round-47 deep-dive).
        """
        return self.policy.module if isinstance(self.policy, DistributedDataParallel) else self.policy

    def _create_ref_model(self) -> nn.Module:
        """Create a frozen copy of the policy model."""
        import copy

        ref_model = copy.deepcopy(self._policy_base())
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def _create_value_model(self) -> ValueModel:
        """Create a trainable critic with the same architecture as the policy."""
        import copy

        value_base = copy.deepcopy(self._policy_base())
        if not isinstance(value_base, DecoderModel):
            raise TypeError(f"critic base must be a DecoderModel, got {type(value_base).__name__}")
        return ValueModel(value_base)

    def _extract_response_values(
        self,
        all_values: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        max_response_len: int,
    ) -> torch.Tensor:
        """Extract per-response-token value estimates from full-sequence critic output."""
        batch_size = all_values.size(0)
        response_values = torch.zeros(
            batch_size,
            max_response_len,
            device=all_values.device,
            dtype=all_values.dtype,
        )

        for i in range(batch_size):
            resp_len = int(response_mask[i].sum().long())
            if resp_len > 0:
                # ``attention_mask`` marks every real token (prompt +
                # response) with 1 and trailing padding with 0, so the real
                # prompt length is (real tokens) - (response tokens). The
                # old ``(1 - response_mask[i]).sum()`` counted trailing
                # padding as prompt tokens, offsetting every non-longest
                # sample's response region into padding (RIL ISS-043).
                prompt_len = int(attention_mask[i].sum().long()) - resp_len
                positions = prompt_len - 1 + torch.arange(resp_len, device=all_values.device)
                response_values[i, :resp_len] = all_values[i, positions]

        return response_values

    def compute_response_values(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replay a prompt-response pair and collect critic values before each response token.
        """
        if self.value_model is None:
            raise RuntimeError("value_model is required to compute response values")

        self.value_model.eval()
        values: list[torch.Tensor] = []
        input_ids = prompt_ids.unsqueeze(0)

        with torch.no_grad():
            for token in response_ids:
                attention_mask = torch.ones_like(input_ids)
                token_values = self.value_model(input_ids, attention_mask)
                values.append(token_values[0, -1])
                input_ids = torch.cat(
                    [input_ids, token.reshape(1, 1)],
                    dim=1,
                )

        self.value_model.train()
        return torch.stack(values)

    def compute_terminal_value(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Critic's bootstrap value V(s_L) at the state AFTER the response.

        ``compute_response_values`` emits a value before each generated token
        but never scores the post-response state s_L (prompt + full response,
        i.e. the state that would predict one more token). That state is the
        continuation point of a max-len-truncated episode — GAE must bootstrap
        from V(s_L) there instead of treating the cut as terminal (RIL
        ISS-178). Returns a scalar tensor.
        """
        if self.value_model is None:
            raise RuntimeError("value_model is required to compute the terminal value")

        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)

        self.value_model.eval()
        with torch.no_grad():
            terminal_value = self.value_model(full_ids, attention_mask)[0, -1]
        self.value_model.train()
        return terminal_value

    def generate_responses(
        self,
        prompts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[bool]]:
        """
        Generate responses for a batch of prompts.

        Returns:
            prompt_ids: List of prompt token tensors
            response_ids: List of response token tensors
            log_probs: List of log probability tensors for responses
            truncated: List of bools — True when the response hit
                ``response_max_len`` instead of terminating at EOS. Truncated
                episodes are not genuinely terminal and must be bootstrapped
                with V(s_L) in GAE (RIL ISS-178). With a tokenizer that has
                no EOS every rollout is truncated.
        """
        self.policy.eval()
        # The tokenizer API is ``eos_token_id`` (SimpleCharacterTokenizer,
        # HF wrapper, and every generation backend use it). The old guard
        # ``hasattr(self.tokenizer, "eos_id")`` was always False — no
        # tokenizer exposes ``eos_id`` — so rollouts never stopped at EOS and
        # ran to ``response_max_len`` with post-EOS junk folded into the
        # training signal (RIL ISS-116).
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        # Normalize list-valued tokenizers so EOS termination works (RIL ISS-334).
        eos_ids = _normalize_eos_ids(eos_id)

        all_prompt_ids = []
        all_response_ids = []
        all_log_probs = []
        all_truncated: list[bool] = []

        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = torch.tensor(
                    self.tokenizer.encode(prompt),
                    dtype=torch.long,
                    device=self.device,
                )

                # Generate response autoregressively
                response_ids = []
                log_probs = []

                # The episode is truncated iff generation exhausts
                # ``response_max_len`` without emitting EOS (RIL ISS-178).
                # Assume truncated; an EOS hit below clears it.
                truncated = True

                input_ids = prompt_ids.unsqueeze(0)  # [1, prompt_len]

                # Cap the response budget to the model's positional context.
                # Without this, a prompt longer than ``max_seq_len -
                # response_max_len`` drives the embedding past the positional
                # table: learned PE raises IndexError, sinusoidal silently
                # truncates the position slice and corrupts hidden states
                # (RIL ISS-334). ``budget`` is the number of response tokens.
                budget = self.config.response_max_len
                model_max_seq_len = getattr(self.policy, "max_seq_len", None)
                if model_max_seq_len is not None:
                    capacity = int(model_max_seq_len) - int(prompt_ids.numel())
                    if capacity <= 0:
                        raise ValueError(
                            f"prompt of {prompt_ids.numel()} tokens already reaches the "
                            f"model's max_seq_len={model_max_seq_len}; there is no room "
                            f"to generate a response. Shorten the prompt or raise "
                            f"max_seq_len."
                        )
                    budget = min(budget, capacity)

                for _ in range(budget):
                    logits = self.policy(input_ids)  # [1, seq_len, vocab_size]
                    next_token_logits = logits[0, -1, :]  # [vocab_size]

                    # The stored ``old_log_probs`` must be log-probs of the
                    # RAW policy — ``ppo_step`` recomputes the current
                    # ``new_log_probs`` from the raw ``log_softmax`` of the
                    # (unscaled) shift logits, so the importance ratio
                    # ``exp(new - old)`` is only a valid IS ratio if both
                    # sides use the raw distribution. Temperature is applied
                    # ONLY to the sampling distribution (softmax over
                    # ``logits / T``), never to the recorded log-prob — with
                    # ``T != 1`` the old code logged ``log_softmax(logits/T)``
                    # and silently mixed two differently-scaled policies in
                    # the ratio (RIL ISS-053).
                    # Sample from the temperature- AND top-k/top-p-filtered
                    # distribution (RIL ISS-176: the config declared these
                    # knobs but generation honored only temperature — pure
                    # multinomial over the full distribution otherwise). The
                    # recorded ``old_log_prob`` below stays the RAW unscaled
                    # log-softmax so the IS ratio ``exp(new - old)`` matches
                    # ppo_step's raw re-score (RIL ISS-053).
                    if self.config.temperature == 0:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        probs = _sampling_probs(
                            next_token_logits,
                            temperature=self.config.temperature,
                            top_k=self.config.top_k,
                            top_p=self.config.top_p,
                        )
                        next_token = torch.multinomial(probs, num_samples=1)

                    # Get log probability under the RAW (unscaled) policy.
                    log_prob = functional.log_softmax(next_token_logits, dim=-1)[next_token]

                    response_ids.append(next_token.item())
                    log_probs.append(log_prob.item())

                    # Update input
                    input_ids = torch.cat(
                        [input_ids, next_token.unsqueeze(0)],
                        dim=1,
                    )

                    # Stop at EOS so post-EOS continuation is not folded into
                    # the response training signal (and rollout compute is not
                    # wasted on the tail). Only an EOS termination makes the
                    # episode genuinely terminal — anything else is a
                    # truncation whose last state must be bootstrapped (RIL
                    # ISS-178).
                    if eos_ids and next_token.item() in eos_ids:
                        truncated = False
                        break

                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(torch.tensor(response_ids, device=self.device))
                all_log_probs.append(torch.tensor(log_probs, device=self.device))
                all_truncated.append(truncated)

        self.policy.train()
        return all_prompt_ids, all_response_ids, all_log_probs, all_truncated

    def compute_rewards(
        self,
        prompt_ids: list[torch.Tensor],
        response_ids: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Compute rewards for prompt-response pairs using the reward model.
        """
        rewards = []

        with torch.no_grad():
            for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True):
                # Concatenate prompt and response
                full_ids = torch.cat([p_ids, r_ids]).unsqueeze(0)  # [1, total_len]
                attention_mask = torch.ones_like(full_ids)

                # Get reward
                reward = self.reward_model(full_ids, attention_mask)  # [1]
                rewards.append(reward.squeeze())

        return rewards

    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            ref_logits = self.ref_model(input_ids)

        policy_logits = self.policy(input_ids)

        # Compute KL divergence on response tokens only
        ref_log_probs = functional.log_softmax(ref_logits, dim=-1)
        policy_log_probs = functional.log_softmax(policy_logits, dim=-1)

        # KL = sum(p * (log p - log q))
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

        # Mask to the states that PREDICT response tokens. ``kl`` at state
        # ``k`` is the KL between the next-token distributions at ``k``, i.e.
        # the state generating ``input[k+1]``; response tokens occupy
        # ``[prompt_len, total_len)`` so their generating states are
        # ``[prompt_len-1, total_len-1)``. Masking with the *unshifted*
        # ``response_mask`` covered the state AFTER the last response token
        # and skipped the first response token's predictor — a one-position
        # bias that misregularized the shift-aligned policy loss (RIL
        # ISS-118). ``response_mask[:, 1:]`` selects exactly the predicting
        # states; drop ``kl[:, -1]`` (the state predicting beyond the window).
        state_mask = response_mask[:, 1:]
        kl = (kl[:, :-1] * state_mask).sum() / state_mask.sum().clamp(min=1)

        return kl

    def ppo_step(self, batch: RolloutBatch) -> dict[str, float]:
        """
        Perform a single PPO update step.

        Returns:
            Dictionary of training metrics.
        """
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        response_mask = batch.response_mask
        old_log_probs = batch.old_log_probs
        advantages = batch.advantages
        returns = batch.returns

        # Deterministic re-scoring (RIL ISS-174): ``old_log_probs`` were
        # computed under ``policy.eval()`` during rollout generation. Re-
        # scoring the SAME inputs in train mode applies dropout (default
        # p=0.1), so even at epoch 0 ``ratio = exp(new - old) != 1``, the
        # approx_kl is spuriously nonzero, and a configured ``target_kl``
        # early-stops without any policy movement (clip fires on dropout
        # noise, not divergence). Keep autograd (the surrogate is the
        # training objective) but score every model in eval mode; restore
        # train mode before returning.
        self.policy.eval()
        if self.value_model is not None:
            self.value_model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()

        # Forward pass
        logits = self.policy(input_ids)

        # Get log probs for actual tokens (shifted)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_response_mask = response_mask[:, 1:]

        new_log_probs = functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            new_log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Compute ratio
        # Note: old_log_probs needs to be aligned with token positions
        response_len = old_log_probs.size(1)
        # Extract log probs for response portion only
        batch_size = input_ids.size(0)
        new_response_log_probs = torch.zeros_like(old_log_probs)
        # Mask over the (padded) ``[B, response_len]`` response window: True
        # at real response-token positions. Used to normalize policy + value
        # losses over real tokens only — a plain ``.mean()`` divides by
        # ``B * response_len`` so padded positions (0 contribution) silently
        # scale gradients by ``real_tokens / (B * response_len)``, which
        # varies per mini-batch (RIL ISS-065).
        response_window_mask = torch.zeros_like(old_log_probs, dtype=torch.bool)

        for i in range(batch_size):
            resp_len = int(response_mask[i].sum().long())
            if resp_len > 0:
                # Same real-prompt-length derivation as
                # ``_extract_response_values``: attention_mask marks all
                # real tokens, response_mask marks only response tokens, so
                # their difference is the true prompt length. Counting
                # ``(1 - response_mask[i]).sum()`` instead inflated prompt_len
                # by the trailing pad count and sliced padding-position log
                # probs into the ratio (RIL ISS-043).
                prompt_len = int(attention_mask[i].sum().long()) - resp_len
                new_response_log_probs[i, :resp_len] = token_log_probs[i, prompt_len - 1 : prompt_len - 1 + resp_len]
                response_window_mask[i, :resp_len] = True

        ratio = (new_response_log_probs - old_log_probs).exp()

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )

        # Mask advantages to response length
        response_advantages = advantages[:, :response_len]

        policy_loss_1 = -response_advantages * ratio
        policy_loss_2 = -response_advantages * clipped_ratio
        # Normalize over real response tokens only (masked mean) so padded
        # zero positions do not dilute the gradient scale (RIL ISS-065).
        policy_loss = (torch.max(policy_loss_1, policy_loss_2) * response_window_mask).sum() / (
            response_window_mask.sum().clamp(min=1)
        )

        # KL penalty
        kl = self.compute_kl_penalty(input_ids, attention_mask, response_mask)
        kl_loss = self.kl_ctl * kl

        # Value loss
        value_loss = torch.tensor(0.0, device=self.device)
        if self.value_model is not None and self.config.value_coef > 0:
            all_values = self.value_model(input_ids, attention_mask)
            pred_values = self._extract_response_values(
                all_values,
                attention_mask,
                response_mask,
                response_len,
            )
            target_returns = returns[:, :response_len]
            # Masked MSE over real response tokens only — a plain mean over
            # the [B, response_len] window includes padded positions (both
            # 0), which dilutes the value gradient identically to the policy
            # loss above (RIL ISS-065).
            squared = (pred_values - target_returns) ** 2
            value_loss = (squared * response_window_mask).sum() / (response_window_mask.sum().clamp(min=1))
            value_loss = self.config.value_coef * value_loss

        # Total loss
        loss = policy_loss + kl_loss + value_loss

        # Entropy bonus (optional) — align with shifted token positions
        if self.config.entropy_coef > 0:
            token_entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1)
            entropy = (token_entropy * shift_response_mask).sum() / shift_response_mask.sum().clamp(min=1)
            loss = loss - self.config.entropy_coef * entropy
        else:
            entropy = torch.tensor(0.0)

        # Backward pass
        self.optimizer.zero_grad()
        if self.value_optimizer is not None:
            self.value_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )
            if self.value_model is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(),
                    self.config.max_grad_norm,
                )

        self.optimizer.step()
        if self.value_optimizer is not None:
            # RIL ISS-173: the critic is trained OUTSIDE the policy's DDP
            # wrapper, so its gradients must be all-reduced explicitly before
            # stepping — otherwise rank R's critic diverges after the first
            # update and the shared policy is trained against rank-inconsistent
            # advantage/target scales (the same gradients-not-synchronized
            # class round-47 fixed for the policy).
            self._sync_value_grads()
            self.value_optimizer.step()

        # Metrics
        with torch.no_grad():
            # Masked means over real response tokens only: padded positions
            # contribute ratio=1 (approx_kl contribution 0), so a plain
            # ``.mean()`` dilutes the telemetry by real/total tokens — with
            # heavily padded mini-batches the ``target_kl`` early-stop fires
            # late or never (RIL ISS-090).
            real_tokens = response_window_mask.sum().clamp(min=1)
            approx_kl = ((ratio - 1) - ratio.log()).mul(response_window_mask).sum() / real_tokens
            ratio_mean = (ratio * response_window_mask).sum() / real_tokens

        result = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss,
            "kl": kl.item(),
            "kl_loss": kl_loss.item(),
            "entropy": entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            "approx_kl": approx_kl.item(),
            "ratio_mean": ratio_mean.item(),
        }

        # Restore train mode (RIL ISS-174): generation re-evals the policy
        # anyway, but leaving dropout modules silently off would change
        # behavior for any caller that trains after ppo_step.
        self.policy.train()
        if self.value_model is not None:
            self.value_model.train()
        if self.ref_model is not None:
            self.ref_model.train()
        return result

    def train_step(self, prompts: list[str]) -> dict[str, float]:
        """
        Perform a complete RLHF training step.

        1. Generate responses for prompts
        2. Compute rewards
        3. Store in buffer and compute advantages
        4. Perform PPO updates for multiple epochs

        Args:
            prompts: List of prompt strings.

        Returns:
            Dictionary of training metrics.
        """
        # 1. Generate responses. The fourth element reports which episodes
        # were truncated at ``response_max_len`` rather than EOS-terminated;
        # those need a V(s_L) bootstrap in GAE (RIL ISS-178).
        prompt_ids, response_ids, log_probs, truncated = self.generate_responses(prompts)

        # 2. Compute rewards
        rewards = self.compute_rewards(prompt_ids, response_ids)

        # ``normalize_rewards`` (RIL ISS-176): standardize the reward batch
        # to zero-mean/unit-variance before storing (general-preferences-style
        # reward standardization). Previously the knob was never read — the
        # raw reward scale fed GAE/returns unchanged.
        if self.config.normalize_rewards and rewards:
            stacked = torch.stack(rewards)
            normalized = (stacked - stacked.mean()) / stacked.std().clamp(min=1e-6)
            rewards = [normalized[i] for i in range(len(rewards))]

        # 3. Store in buffer
        self.buffer.clear()
        for p_ids, r_ids, lp, reward, trunc in zip(
            prompt_ids,
            response_ids,
            log_probs,
            rewards,
            truncated,
            strict=True,
        ):
            values = None
            terminal_value = None
            if self.value_model is not None:
                values = self.compute_response_values(p_ids, r_ids)
                # Truncated episodes are not terminal: collect the critic's
                # V(s_L) at the post-response state so GAE bootstraps it
                # instead of assuming a hard stop (RIL ISS-178).
                if trunc:
                    terminal_value = self.compute_terminal_value(p_ids, r_ids)
            self.buffer.add(
                prompt_ids=p_ids,
                response_ids=r_ids,
                rewards=reward,
                old_log_probs=lp,
                values=values,
                truncated=trunc,
                terminal_value=terminal_value,
            )

        # 4. Compute advantages
        self.buffer.compute_advantages()

        # 5. PPO epochs
        all_metrics = []
        # ``target_kl`` early stopping must halt the WHOLE epoch loop, not
        # just the current mini-batch pass. The old inner-only ``break`` let
        # the outer ``for _epoch`` re-apply PPO updates to the SAME rollout
        # after epoch 0 diverged, defeating the KL-blowup safeguard
        # entirely (RIL ISS-052).
        kl_breached = False
        for _epoch in range(self.config.ppo_epochs):
            if kl_breached:
                break
            for batch in self.buffer.get_batches(
                mini_batch_size=self.config.mini_batch_size,
                device=self.device,
            ):
                metrics = self.ppo_step(batch)
                all_metrics.append(metrics)

                # Early stopping based on KL
                if self.config.target_kl is not None and metrics["approx_kl"] > self.config.target_kl:
                    logger.info(f"Early stopping: KL {metrics['approx_kl']:.4f} > target {self.config.target_kl}")
                    kl_breached = True
                    break

        self.global_step += 1

        # Aggregate metrics — a degenerate-but-legal config (ppo_epochs=0, or
        # an empty buffer because mini_batch_size slicing produced nothing)
        # yields no batches; return zeroed metrics instead of IndexError-ing
        # on ``all_metrics[0]`` or dividing by zero (RIL ISS-177).
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        avg_metrics["reward_mean"] = sum(r.item() for r in rewards) / len(rewards) if rewards else 0.0
        avg_metrics["response_len_mean"] = (
            sum(len(r) for r in response_ids) / len(response_ids) if response_ids else 0.0
        )

        return avg_metrics
