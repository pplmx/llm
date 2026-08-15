"""Tests for RLHF PPO Trainer."""

import pytest
import torch

from llm.training.core.config import PPOConfig
from llm.training.rlhf.rollout_buffer import RolloutBuffer
from tests.support.devices import DEFAULT_DEVICE


class TestPPOConfig:
    """Tests for PPOConfig."""

    def test_default_config(self):
        """Test default PPO config values."""
        config = PPOConfig()

        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.1
        assert config.ppo_epochs == 4
        assert config.gae_lambda == 0.95

    def test_custom_config(self):
        """Test custom PPO config."""
        config = PPOConfig(
            clip_epsilon=0.1,
            ppo_epochs=2,
            mini_batch_size=32,
        )

        assert config.clip_epsilon == 0.1
        assert config.ppo_epochs == 2
        assert config.mini_batch_size == 32


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_add_sample(self):
        """Test adding samples to buffer."""
        buffer = RolloutBuffer()

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        assert len(buffer) == 1

    def test_compute_advantages(self):
        """Test advantage computation."""
        buffer = RolloutBuffer(normalize_advantages=False)

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        buffer.compute_advantages()

        assert len(buffer.samples[0].advantages) == 2
        assert torch.isfinite(buffer.samples[0].advantages).all()

    def test_gae_with_values(self):
        """GAE should use sparse terminal reward and bootstrap with zero."""
        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=False)
        values = torch.tensor([0.5, 0.3])

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values,
        )

        buffer.compute_advantages()

        expected = torch.tensor([0.5, 0.7])
        assert torch.allclose(buffer.samples[0].advantages, expected, atol=1e-5)
        assert torch.allclose(
            buffer.samples[0].advantages + values,
            torch.tensor([1.0, 1.0]),
            atol=1e-5,
        )

    def test_returns_are_raw_when_advantages_normalized(self):
        """Returns must stay on the raw (unnormalized) return scale.

        With the default ``normalize_advantages=True``, ``sample.advantages``
        is standardized in place for the policy ratio, but the critic's
        regression target (returns) must be ``raw_advantage + value`` — the
        true return — NOT ``normalized_advantage + value``. The next rollout's
        GAE bootstrap consumes the critic's output raw, so training the critic
        on a shifted/scaled target corrupts the advantage signal every rollout.
        """
        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=True)
        # Two samples with clearly different scales so normalization shifts mean.
        values1 = torch.tensor([0.5, 0.3])
        values2 = torch.tensor([10.0, 10.0])
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values1,
        )
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values2,
        )
        buffer.compute_advantages()

        for sample in buffer.samples:
            assert sample.returns is not None
            assert sample.returns.shape == sample.advantages.shape

        # Explicitly: normalized advantages have zero mean across the batch,
        # yet returns must still carry the true return scale (raw
        # advantage + value, uncentered) — that is what the critic regresses.
        batch = next(iter(buffer.get_batches(mini_batch_size=4, shuffle=False)))
        mean_adv = batch.advantages.mean()
        assert abs(float(mean_adv)) < 1e-5, f"advantages should be normalized (mean~0), got {mean_adv}"
        # Batch returns must equal the RAW return per sample, not the
        # centered one. First sample raw returns are [1.0, 1.0]
        # (raw_adv [0.5, 0.7] + values [0.5, 0.3]).
        raw_adv1 = torch.tensor([0.5, 0.7])
        vals1 = torch.tensor([0.5, 0.3])
        assert torch.allclose(batch.returns[0, :2], raw_adv1 + vals1, atol=1e-5)

    def test_get_batches(self):
        """Test mini-batch generation."""
        buffer = RolloutBuffer()

        # Add multiple samples
        for i in range(4):
            buffer.add(
                prompt_ids=torch.tensor([1, 2, 3]),
                response_ids=torch.tensor([4, 5, 6]),
                rewards=torch.tensor(float(i)),
                old_log_probs=torch.tensor([-0.5, -0.3, -0.2]),
            )

        buffer.compute_advantages()

        batches = list(buffer.get_batches(mini_batch_size=2))

        assert len(batches) == 2
        assert batches[0].input_ids.shape[0] == 2
        assert batches[0].rewards.shape[0] == 2

    def test_clear_buffer(self):
        """Test buffer clearing."""
        buffer = RolloutBuffer()

        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )

        assert len(buffer) == 1
        buffer.clear()
        assert len(buffer) == 0


class TestPPOTrainer:
    """Tests for PPOTrainer."""

    @pytest.fixture
    def tiny_setup(self, tiny_model):
        """Create minimal setup for PPO trainer tests."""
        from llm.training.tasks.reward_task import RewardModel

        class SimpleTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 100 for c in text[:10]]

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(i + 32) for i in ids)

            eos_token_id = None

        policy = tiny_model
        reward_model = RewardModel(tiny_model)
        tokenizer = SimpleTokenizer()
        config = PPOConfig(
            ppo_epochs=1,
            mini_batch_size=2,
            response_max_len=5,
        )

        return policy, reward_model, tokenizer, config

    def test_ppo_trainer_init(self, tiny_setup):
        """Test PPO trainer initialization."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        assert trainer.policy is policy
        assert trainer.reward_model is reward_model
        assert type(trainer.ref_model) is type(policy)
        assert trainer.ref_model is not policy
        assert trainer.value_optimizer.param_groups[0]["lr"] == 1e-5

    def test_compute_response_values(self, tiny_setup):
        """Critic should emit one value per response token."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer
        from llm.training.rlhf.value_model import ValueModel

        policy, reward_model, tokenizer, config = tiny_setup
        value_model = ValueModel(policy)

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            value_model=value_model,
            device=str(DEFAULT_DEVICE),
        )

        prompt_ids = torch.tensor([1, 2, 3], device=DEFAULT_DEVICE)
        response_ids = torch.tensor([4, 5, 6], device=DEFAULT_DEVICE)
        values = trainer.compute_response_values(prompt_ids, response_ids)

        assert values.shape == (3,)
        assert torch.isfinite(values).all()

    def test_generate_responses_old_log_probs_are_raw_policy(self, tiny_setup):
        """Regression (RIL ISS-053): ``old_log_probs`` must be log-probs of
        the RAW policy (matching what ``ppo_step`` recomputes), not the
        temperature-scaled distribution used for sampling.

        ``generate_responses`` logged ``log_softmax(logits / temperature)``
        while ``ppo_step`` recomputes ``log_softmax(raw shift_logits)``. With
        ``temperature != 1.0`` the importance ratio ``exp(new - old)`` was
        then an invalid IS ratio between two differently-scaled policies.
        Temperature affects only sampling; the stored log-prob must be the
        raw-policy value."""
        from unittest.mock import patch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.temperature = 2.0

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        prompts = ["Hello"]

        # Deterministic greedy sampling: pick argmax each step. Argmax of a
        # softmax is invariant to positive temperature scaling, so the
        # sampled tokens are identical under any temperature — whatever the
        # log-prob of those tokens is, it must be the RAW policy's value.
        # ``torch.multinomial(probs, 1)`` -> argmax(probs).
        with patch(
            "torch.multinomial", side_effect=lambda probs, num_samples=1, **kw: probs.argmax(dim=-1, keepdim=True)
        ):
            prompt_ids, response_ids, log_probs = trainer.generate_responses(prompts)

        # Sanity: greedy sampling still ran some steps.
        assert len(response_ids) == 1
        assert len(log_probs) == 1

        # Recompute the raw-policy log-probs of the generated response and
        # compare. ppo_step uses log_softmax(raw logits), so anything else
        # (e.g. a temperature-scaled value) would corrupt the ratio.
        # ``generate_responses`` runs the policy in eval() mode; replay in
        # the same mode so dropout does not perturb the logits.
        policy.eval()
        sample_logits = [None] * len(log_probs[0])
        generated = list(prompt_ids[0].tolist())
        for i in range(len(log_probs[0])):
            with torch.no_grad():
                out = policy(torch.tensor([generated], device=str(DEFAULT_DEVICE)))
            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[0, -1, :]
            expected_raw = torch.log_softmax(next_logits, dim=-1)
            token = response_ids[0][i].item()
            sample_logits[i] = expected_raw[token].item()
            generated.append(token)

        stored = log_probs[0].detach().cpu()
        expected = torch.as_tensor(sample_logits, device=stored.device)
        assert torch.allclose(stored, expected, atol=1e-5), (
            f"old_log_probs must be RAW-policy log-probs (temperature only "
            f"affects sampling), got {stored.tolist()} expected {expected.tolist()}"
        )

    def test_generate_responses(self, tiny_setup):
        """Test response generation."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        prompts = ["Hello", "Hi"]
        prompt_ids, response_ids, log_probs = trainer.generate_responses(prompts)

        assert len(prompt_ids) == 2
        assert len(response_ids) == 2
        assert len(log_probs) == 2

        # Each response should have tokens
        assert len(response_ids[0]) > 0
        assert len(log_probs[0]) == len(response_ids[0])

    def test_generate_responses_stops_at_eos(self, tiny_setup):
        """PPO rollout must stop at the tokenizer's EOS token (regression for
        RIL ISS-116). The old guard read ``hasattr(tokenizer, "eos_id")`` —
        no tokenizer exposes ``eos_id`` (the convention is ``eos_token_id``)
        — so rollouts always ran to ``response_max_len`` and folded post-EOS
        junk into the training signal. With the attribute fixed, forcing the
        first sampled token to be EOS must yield a 1-token response."""
        from unittest.mock import patch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        tokenizer.eos_token_id = 0  # EOS is id 0, in the sampled vocab range
        tokenizer.encode = lambda text: [1, 2, 3]  # fixed 3-token prompt

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        with patch("torch.multinomial", return_value=torch.tensor([0], device=DEFAULT_DEVICE)):
            _, response_ids, _ = trainer.generate_responses(["Hello"])

        assert len(response_ids[0]) == 1, f"rollout should stop at EOS (1 token), got {len(response_ids[0])}"

    def test_compute_rewards(self, tiny_setup):
        """Test reward computation."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        prompt_ids = [torch.tensor([1, 2, 3], device=DEFAULT_DEVICE)]
        response_ids = [torch.tensor([4, 5], device=DEFAULT_DEVICE)]

        rewards = trainer.compute_rewards(prompt_ids, response_ids)

        assert len(rewards) == 1
        assert rewards[0].dim() == 0  # Scalar

    def test_ppo_step_with_entropy_bonus(self, tiny_setup):
        """PPO step should work when entropy_coef > 0 (shifted mask alignment)."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.entropy_coef = 0.01

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        buffer = RolloutBuffer(normalize_advantages=False)
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
        )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=1, shuffle=False, device=str(DEFAULT_DEVICE))))

        metrics = trainer.ppo_step(batch)

        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_ppo_step_with_value_loss(self, tiny_setup):
        """PPO step should include value loss when critic is enabled."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer
        from llm.training.rlhf.value_model import ValueModel

        policy, reward_model, tokenizer, config = tiny_setup
        value_model = ValueModel(policy)

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            value_model=value_model,
            device=str(DEFAULT_DEVICE),
        )

        buffer = RolloutBuffer(normalize_advantages=False, gae_lambda=1.0, gamma=1.0)
        values = torch.tensor([0.1, 0.2])
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values,
        )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=1, shuffle=False, device=str(DEFAULT_DEVICE))))

        metrics = trainer.ppo_step(batch)

        assert metrics["value_loss"] > 0.0
        assert torch.isfinite(torch.tensor(metrics["value_loss"]))

    def test_ppo_step_losses_normalized_over_real_tokens(self):
        """Regression (RIL ISS-065): the policy loss must be normalized over
        REAL response tokens, not the padded ``[B, response_len]`` window.

        A plain ``.mean()`` divides by ``B * response_len``; padded positions
        contribute exactly 0, so gradients scale by
        ``real_tokens / (B * response_len)`` — wrong magnitude and varying
        per mini-batch. With a deterministic stub policy we can compute the
        expected masked policy loss by hand and assert it exactly."""
        import torch
        import torch.nn as nn

        from llm.training.core.config import PPOConfig
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        class StubPolicy(nn.Module):
            """Deterministic policy: logits = emb[token] @ W (no randomness,
            so the expected loss is computable analytically in the test)."""

            def __init__(self, vocab: int, hidden: int, out: int):
                super().__init__()
                self.emb = nn.Embedding(vocab, hidden)
                self.w = nn.Linear(hidden, out)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.w(self.emb(input_ids))

        class SimpleTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 100 for c in text[:10]]

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(i + 32) for i in ids)

            eos_token_id = None

        class StubReward(nn.Module):
            """Pass-through reward scoring that never fires a real reward
            forward (rewards are taken from the rollout buffer, not
            recomputed here)."""

            def __init__(self):
                super().__init__()

            def forward(self, input_ids):
                return torch.zeros(input_ids.shape[0], device=input_ids.device)

        torch.manual_seed(0)
        policy = StubPolicy(vocab=16, hidden=8, out=16).eval()
        reward_model = StubReward()
        tokenizer = SimpleTokenizer()
        config = PPOConfig(
            ppo_epochs=1,
            mini_batch_size=2,
            response_max_len=5,
            use_ref_model=False,  # stub policy is not a DecoderModel
            value_coef=0.0,  # skip the critic entirely (policy-loss only test)
        )
        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,  # type: ignore[arg-type]
            tokenizer=tokenizer,  # type: ignore[arg-type]
            config=config,
            device="cpu",
        )

        buffer = RolloutBuffer(normalize_advantages=False)
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5, 6]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3, -0.2]),
        )
        buffer.add(
            prompt_ids=torch.tensor([9, 8, 7]),
            response_ids=torch.tensor([6]),
            rewards=torch.tensor(0.5),
            old_log_probs=torch.tensor([-0.4]),
        )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=2, shuffle=False, device="cpu")))

        # Manual reference: recompute the masked policy loss over the real
        # response tokens only, then compare against the trainer's report.
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        response_mask = batch.response_mask
        old_log_probs = batch.old_log_probs
        advantages = batch.advantages
        response_len = old_log_probs.shape[1]

        with torch.no_grad():
            logits = policy(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            new_log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(new_log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

            new_response_log_probs = torch.zeros_like(old_log_probs)
            mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
            for i in range(input_ids.shape[0]):
                resp_len = int(response_mask[i].sum().long())
                if resp_len > 0:
                    prompt_len = int(attention_mask[i].sum().long()) - resp_len
                    new_response_log_probs[i, :resp_len] = token_log_probs[
                        i, prompt_len - 1 : prompt_len - 1 + resp_len
                    ]
                    mask[i, :resp_len] = True

            ratio = (new_response_log_probs - old_log_probs).exp()
            clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            resp_adv = advantages[:, :response_len]
            pl1 = -resp_adv * ratio
            pl2 = -resp_adv * clipped
            masked_loss = (torch.max(pl1, pl2) * mask).sum() / mask.sum().clamp(min=1)
            unmasked_loss = torch.max(pl1, pl2).mean()

        assert int(mask.sum()) == 4, "expected 3 + 1 real response tokens"

        metrics = trainer.ppo_step(batch)

        # The trainer's reported policy_loss must match the masked reference
        # nearly exactly (same deterministic forward).
        assert abs(metrics["policy_loss"] - float(masked_loss)) < 1e-6, (
            f"policy loss must be a masked mean over real tokens: "
            f"trainer={metrics['policy_loss']} expected={float(masked_loss)}"
        )
        # With a 2-row padded batch the masked and unmasked means differ
        # (masked == real-token mean, unmasked == padded mean).
        assert abs(float(masked_loss - unmasked_loss)) > 1e-9

    def test_ppo_step_approx_kl_normalized_over_real_tokens(self):
        """Regression (RIL ISS-090): ``approx_kl`` / ``ratio_mean`` telemetry
        must be masked means over REAL response tokens, not the padded
        ``[B, response_len]`` window.

        Padded positions hold ratio==1 (zero KL contribution), so a plain
        ``.mean()`` dilutes ``approx_kl`` by ``real/total`` — with heavily
        padded mini-batches the ``target_kl`` early-stop fires late or never.
        """
        import torch
        import torch.nn as nn

        from llm.training.core.config import PPOConfig
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        class StubPolicy(nn.Module):
            def __init__(self, vocab: int, hidden: int, out: int):
                super().__init__()
                self.emb = nn.Embedding(vocab, hidden)
                self.w = nn.Linear(hidden, out)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.w(self.emb(input_ids))

        class SimpleTokenizer:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 100 for c in text[:10]]

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(i + 32) for i in ids)

            eos_token_id = None

        class StubReward(nn.Module):
            def forward(self, input_ids):
                return torch.zeros(input_ids.shape[0], device=input_ids.device)

        torch.manual_seed(0)
        policy = StubPolicy(vocab=16, hidden=8, out=16).eval()
        config = PPOConfig(
            ppo_epochs=1,
            mini_batch_size=2,
            response_max_len=5,
            use_ref_model=False,  # stub policy is not a DecoderModel
            value_coef=0.0,  # skip the critic entirely
        )
        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=StubReward(),  # type: ignore[arg-type]
            tokenizer=SimpleTokenizer(),  # type: ignore[arg-type]
            config=config,
            device="cpu",
        )

        buffer = RolloutBuffer(normalize_advantages=False)
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5, 6]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3, -0.2]),
        )
        buffer.add(
            prompt_ids=torch.tensor([9, 8, 7]),
            response_ids=torch.tensor([6]),
            rewards=torch.tensor(0.5),
            old_log_probs=torch.tensor([-0.4]),
        )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=2, shuffle=False, device="cpu")))

        # Manual reference over real response tokens (same derivation as the
        # trainer's masked metrics).
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        response_mask = batch.response_mask
        old_log_probs = batch.old_log_probs

        with torch.no_grad():
            shift_logits = policy(input_ids)[:, :-1, :]
            new_log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(new_log_probs, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            new_response_log_probs = torch.zeros_like(old_log_probs)
            mask = torch.zeros_like(old_log_probs, dtype=torch.bool)
            for i in range(input_ids.shape[0]):
                resp_len = int(response_mask[i].sum().long())
                if resp_len > 0:
                    prompt_len = int(attention_mask[i].sum().long()) - resp_len
                    new_response_log_probs[i, :resp_len] = token_log_probs[
                        i, prompt_len - 1 : prompt_len - 1 + resp_len
                    ]
                    mask[i, :resp_len] = True

            ratio = (new_response_log_probs - old_log_probs).exp()
            kls = (ratio - 1) - ratio.log()
            ref_masked = (kls * mask).sum() / mask.sum().clamp(min=1)
            ref_unmasked = kls.mean()

        assert int(mask.sum()) == 4, "expected 3 + 1 real response tokens (padding present)"

        metrics = trainer.ppo_step(batch)

        assert abs(metrics["approx_kl"] - float(ref_masked)) < 1e-6, (
            f"approx_kl must be a masked mean over real tokens: "
            f"trainer={metrics['approx_kl']} expected={float(ref_masked)}"
        )
        assert abs(metrics["ratio_mean"] - float((ratio * mask).sum() / mask.sum())) < 1e-6, (
            "ratio_mean must be a masked mean over real tokens"
        )
        # Padding actually dilutes the plain mean away from the masked one.
        assert abs(float(ref_masked - ref_unmasked)) > 1e-9

    def test_target_kl_stops_all_ppo_epochs(self, tiny_setup):
        """Regression (RIL ISS-052): ``target_kl`` early stopping must halt
        the whole epoch loop, not just the current mini-batch loop.

        The old ``break`` sat inside the inner ``for batch in get_batches``
        loop, so after epoch 0 diverged the outer ``for _epoch in
        range(ppo_epochs)`` continued re-applying PPO updates to the SAME
        rollout — defeating the KL-blowup safeguard entirely. With
        ``target_kl`` exceeded on the first batch, ``ppo_step`` must be
        called exactly once (one epoch, one batch), not ``ppo_epochs`` times.
        """
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.ppo_epochs = 4
        config.mini_batch_size = 2  # 2 prompts -> one batch per epoch
        config.target_kl = 0.0  # approx_kl will always exceed this

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        # Stub ppo_step to always exceed the KL target so the early-stop
        # fires on the very first batch, and count how many times it runs.
        calls = {"n": 0}

        def _divergent_ppo_step(_batch):
            calls["n"] += 1
            return {
                "loss": 0.5,
                "policy_loss": 0.5,
                "value_loss": 0.0,
                "kl": 1.0,
                "kl_loss": 0.1,
                "entropy": 0.0,
                "approx_kl": 1.0,
                "ratio_mean": 1.5,
            }

        trainer.ppo_step = _divergent_ppo_step  # type: ignore[method-assign]

        trainer.train_step(["hello", "hi"])

        assert calls["n"] == 1, (
            f"target_kl early stop must break out of ALL ppo_epochs; ppo_step ran {calls['n']} times (expected 1)"
        )

    def test_ppo_step_padding_not_counted_as_prompt_tokens(self, tiny_setup):
        """Regression (RIL ISS-043): trailing padding in a padded mini-batch
        must not be counted as prompt tokens.

        ``rollout_buffer._collate_batch`` pads each sample to the longest
        sequence and only marks real response tokens in ``response_mask``.
        The old ``prompt_len = (1 - response_mask[i]).sum()`` therefore
        counted prompt tokens PLUS trailing padding as "prompt", so ratio
        slicing and value extraction read padding positions for every
        non-longest sample. Deriving ``prompt_len`` from the real sequence
        length (attention_mask sum minus response length) must align the
        extracted values with the actual response tokens.
        """
        from llm.training.rlhf.ppo_trainer import PPOTrainer
        from llm.training.rlhf.value_model import ValueModel

        policy, reward_model, tokenizer, config = tiny_setup
        value_model = ValueModel(policy)

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            value_model=value_model,
            device=str(DEFAULT_DEVICE),
        )

        # Sample 0: prompt [1], response [4, 5] → total len 3.
        # Sample 1: prompt [2, 3, 7], response [6] → total len 4.
        # Max total is 4, so sample 0's row is padded with one trailing 0.
        # That padding must NOT be counted as a prompt token when locating
        # the response region.
        buffer = RolloutBuffer(normalize_advantages=False, gae_lambda=1.0, gamma=1.0)
        buffer.add(
            prompt_ids=torch.tensor([1], dtype=torch.long),
            response_ids=torch.tensor([4, 5], dtype=torch.long),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=torch.tensor([0.1, 0.2]),
        )
        buffer.add(
            prompt_ids=torch.tensor([2, 3, 7], dtype=torch.long),
            response_ids=torch.tensor([6], dtype=torch.long),
            rewards=torch.tensor(2.0),
            old_log_probs=torch.tensor([-0.4]),
            values=torch.tensor([0.3]),
        )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=2, shuffle=False, device=str(DEFAULT_DEVICE))))

        # Rows are padded to total len 4. Each position holds a unique value
        # so we can assert exactly which real positions the extractor read.
        # Sample 0: real tokens at positions 0,1,2; padding at 3.
        #   prompt_len = 1, response len = 2 → response-token values are the
        #   values at positions [prompt_len-1, prompt_len] = [0, 1].
        # Sample 1: real tokens at positions 0,1,2,3; no padding.
        #   prompt_len = 3, response len = 1 → value at position [2].
        all_values = torch.tensor(
            [
                [100.0, 101.0, 102.0, 103.0],
                [200.0, 201.0, 202.0, 203.0],
            ],
            device=str(DEFAULT_DEVICE),
        )
        response_values = trainer._extract_response_values(all_values, batch.attention_mask, batch.response_mask, 2)

        # Sample 0: positions [0, 1] → 100, 101. The old buggy code computed
        # prompt_len = (1 - response_mask[0]).sum() = 2 (padding included),
        # so it read positions [1, 2] → 101, 102 (off by one, and wrong).
        assert torch.allclose(response_values[0, :2], torch.tensor([100.0, 101.0], device=str(DEFAULT_DEVICE)))
        # Sample 1: response len 1 → position [2] → 202.
        assert torch.allclose(response_values[1, :1], torch.tensor([202.0], device=str(DEFAULT_DEVICE)))
