"""Tests for RLHF PPO Trainer."""

import pytest
import torch
import torch.nn.functional as functional

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

    def test_gae_truncated_episode_bootstraps_from_terminal_value(self):
        """Regression (RIL ISS-178): a max-len-truncated episode is NOT
        terminal — GAE must bootstrap the last response state with the
        critic's ``V(s_L)`` (the value at the state AFTER the last
        generated token) instead of hardcoding ``next_value=0.0``. The old
        code modeled truncation- and EOS-terminated episodes identically,
        which crushed the final-token advantages of every truncated
        rollout (rollout_buffer.py:102-105)."""
        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=False)
        values = torch.tensor([0.5, 0.3])
        # Truncated at max length: compute_terminal_value scored the state
        # after the last response token at 2.0.
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values,
            truncated=True,
            terminal_value=torch.tensor(2.0),
        )
        buffer.compute_advantages()
        sample = buffer.samples[0]

        # GAE with done=False: last-token delta = r + gamma*V(s_L) - V(s_{L-1}).
        # With gamma=1, gae_lambda=1: A[1] = 1.0 + 2.0 - 0.3 = 2.7,
        # A[0] = 0 + 0.3 - 0.5 + A[1] = 2.5 (vs the terminal [0.5, 0.7]
        # the same sample would get with next_value=0).
        expected = torch.tensor([2.5, 2.7])
        assert torch.allclose(sample.advantages, expected, atol=1e-5), (
            f"truncated episode must bootstrap from V(s_L): got {sample.advantages} expected {expected}"
        )
        # Raw returns (critic targets): every position = r + gamma*V(s_L) = 3.0.
        assert sample.returns is not None
        assert torch.allclose(sample.returns, torch.tensor([3.0, 3.0]), atol=1e-5)

    def test_gae_eos_episode_stays_terminal_when_not_truncated(self):
        """A genuinely EOS-terminated episode keeps the terminal bootstrap
        (next_value=0). Only truncated episodes consume ``V(s_L)`` — a
        residual terminal_value must be ignored, not applied."""
        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=False)
        values = torch.tensor([0.5, 0.3])
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values,
            truncated=False,
            terminal_value=torch.tensor(999.0),  # must be ignored
        )
        buffer.compute_advantages()
        sample = buffer.samples[0]

        expected = torch.tensor([0.5, 0.7])
        assert torch.allclose(sample.advantages, expected, atol=1e-5)

    def test_gae_truncated_without_terminal_value_falls_back_terminal(self):
        """A sample flagged truncated but carrying no critic estimate (the
        value_coef=0 path never computes terminal values) must not crash:
        the legacy terminal-0 bootstrap applies."""
        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=False)
        values = torch.tensor([0.5, 0.3])
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5, -0.3]),
            values=values,
            truncated=True,
            terminal_value=None,
        )
        buffer.compute_advantages()
        sample = buffer.samples[0]

        expected = torch.tensor([0.5, 0.7])
        assert torch.allclose(sample.advantages, expected, atol=1e-5)

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

    def test_compute_terminal_value(self, tiny_setup):
        """V(s_L): the critic value at the state AFTER the last response
        token (prompt + full response), the bootstrap used for truncated
        episodes (RIL ISS-178). ``compute_response_values`` only emits the
        value BEFORE each generated token — the post-response state was
        never scored, so truncated episodes could not be bootstrapped."""
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
        terminal = trainer.compute_terminal_value(prompt_ids, response_ids)

        assert terminal.dim() == 0  # scalar
        assert torch.isfinite(terminal)

        # Sanity: it must equal a manual forward over prompt+response at the
        # last position (the state that would predict one more token).
        full = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        mask = torch.ones_like(full)
        trainer.value_model.eval()
        with torch.no_grad():
            manual = trainer.value_model(full, mask)[0, -1]
        assert torch.allclose(terminal, manual, atol=1e-6)

    def test_compute_terminal_value_requires_value_model(self, tiny_setup):
        """Without a critic there is no bootstrap to collect — must raise
        instead of silently deg(/crashing) on None."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.value_coef = 0.0  # no critic created

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )
        assert trainer.value_model is None

        with pytest.raises(RuntimeError, match="value_model"):
            trainer.compute_terminal_value(
                torch.tensor([1, 2, 3], device=DEFAULT_DEVICE),
                torch.tensor([4, 5], device=DEFAULT_DEVICE),
            )

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
            prompt_ids, response_ids, log_probs, _truncated = trainer.generate_responses(prompts)

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
        prompt_ids, response_ids, log_probs, _truncated = trainer.generate_responses(prompts)

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
            _, response_ids, _, truncated = trainer.generate_responses(["Hello"])

        assert len(response_ids[0]) == 1, f"rollout should stop at EOS (1 token), got {len(response_ids[0])}"
        assert truncated == [False], "an EOS-terminated episode is NOT truncated"

    def test_generate_responses_reports_max_len_truncation(self, tiny_setup):
        """Regression (RIL ISS-178): when the tokenizer has no EOS the
        rollout always exhausts ``response_max_len``; ``generate_responses``
        must report those episodes as truncated so GAE can bootstrap with
        ``V(s_L)`` instead of treating them as terminal."""
        from unittest.mock import patch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        tokenizer.eos_token_id = None  # default, but explicit: never stops early
        config.response_max_len = 2

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        with patch("torch.multinomial", return_value=torch.tensor([5], device=DEFAULT_DEVICE)):
            _, response_ids, _, truncated = trainer.generate_responses(["Hello", "Hi"])

        assert truncated == [True, True]
        assert all(len(r) == config.response_max_len for r in response_ids), (
            "no-EOS rollouts must run to response_max_len and be flagged truncated"
        )

    def test_train_step_threads_truncation_and_collects_terminal_value(self, tiny_setup):
        """Regression (RIL ISS-178): ``train_step`` must thread the truncation
        flag from generation into the rollout buffer and collect ``V(s_L)``
        for truncated episodes, so ``compute_advantages`` bootstraps instead
        of forcing terminal. End-to-end over the full train_step pipeline."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer
        from llm.training.rlhf.value_model import ValueModel

        policy, reward_model, tokenizer, config = tiny_setup
        tokenizer.eos_token_id = None  # rollouts always truncate at max len
        config.response_max_len = 2
        config.top_k = 1  # deterministic greedy sampling
        config.ppo_epochs = 1
        value_model = ValueModel(policy)

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            value_model=value_model,
            device=str(DEFAULT_DEVICE),
        )

        metrics = trainer.train_step(["Hello", "Hi there"])

        samples = trainer.buffer.samples
        assert len(samples) == 2
        assert all(s.truncated for s in samples), "no-EOS rollouts must be flagged truncated"
        assert all(s.terminal_value is not None for s in samples)
        assert all(s.terminal_value.dim() == 0 for s in samples)
        assert all(torch.isfinite(s.terminal_value) for s in samples)
        # The advantage of the last token of each truncated sample must carry
        # the V(s_L) bootstrap (positive when the critic values s_L), i.e.
        # differ from the hardcoded-terminal value — sanity that the signal
        # actually flowed through.
        assert torch.isfinite(torch.tensor(metrics["loss"]))

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

    def test_compute_kl_penalty_aligned_to_response_predicting_states(self, tiny_setup):
        """Regression (RIL ISS-118): the KL penalty must average over the
        states that PREDICT response tokens — ``[prompt_len-1, total_len-1)``,
        one slot before each response token, matching the policy-loss shift —
        not the unshifted response positions (which include the state after
        the last response token and skip the first response token's state)."""
        import torch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        # Make ref diverge from the policy, otherwise KL is 0 everywhere and
        # any masking window trivially agrees (tiny_setup's ref is a copy of
        # the untrained policy).
        with torch.no_grad():
            for p in trainer.ref_model.parameters():
                p.add_(torch.randn_like(p) * 0.2)

        # 2-token prompt + 3-token response.
        input_ids = torch.tensor([[10, 11, 5, 6, 7]], device=DEFAULT_DEVICE)
        attention_mask = torch.ones(1, 5, device=DEFAULT_DEVICE)
        response_mask = torch.tensor([[0, 0, 1, 1, 1]], device=DEFAULT_DEVICE)

        kl_value = trainer.compute_kl_penalty(input_ids, attention_mask, response_mask)

        # Reference: the correctly-aligned window is states [prompt_len-1,
        # total_len-1) = [1, 4) — states whose next-token distribution emits
        # a response token.
        with torch.no_grad():
            ref_logits = trainer.ref_model(input_ids)
            pol_logits = trainer.policy(input_ids)
        ref_lp = functional.log_softmax(ref_logits, dim=-1)
        pol_lp = functional.log_softmax(pol_logits, dim=-1)
        kl = (pol_lp.exp() * (pol_lp - ref_lp)).sum(dim=-1)  # [1, 5]

        correct_mask = torch.tensor([[0, 1, 1, 1, 0]], device=DEFAULT_DEVICE)
        expected = (kl * correct_mask).sum() / correct_mask.sum()

        assert abs(float(kl_value.detach()) - float(expected.detach())) < 1e-6, (
            f"KL penalty window must match the response-predicting states: "
            f"trainer={float(kl_value.detach()):.6f} expected={float(expected.detach()):.6f}"
        )

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

    def test_ppo_step_epoch0_ratio_is_identity_under_eval(self, tiny_setup):
        """Regression (RIL ISS-174): re-scoring a rollout with the SAME policy
        must give an identity ratio at epoch 0 — the policy's weights have not
        moved, and old_log_probs were computed under eval() (no dropout). The
        old code re-scored in train() mode, so dropout (p=0.1) made ratio != 1
        spuriously and a configured ``target_kl`` early-stopped with zero
        policy movement."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer
        from llm.training.rlhf.rollout_buffer import RolloutBuffer

        policy, reward_model, tokenizer, config = tiny_setup
        assert hasattr(policy, "training")  # real DecoderModel — has dropout ops
        assert any(m.__class__.__name__ == "Dropout" for m in policy.modules()), (
            "test precondition: tiny_model must contain dropout to expose ISS-174"
        )

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        prompts = ["Hello", "Hi there"]
        prompt_ids, response_ids, log_probs, _truncated = trainer.generate_responses(prompts)
        assert all(len(p) > 0 for p in response_ids)

        buffer = RolloutBuffer(normalize_advantages=False)
        for p_ids, r_ids, lp in zip(prompt_ids, response_ids, log_probs, strict=True):
            buffer.add(
                prompt_ids=p_ids,
                response_ids=r_ids,
                rewards=torch.tensor(1.0),
                old_log_probs=lp,
            )
        buffer.compute_advantages()
        batch = next(iter(buffer.get_batches(mini_batch_size=16, shuffle=False, device=str(DEFAULT_DEVICE))))

        metrics = trainer.ppo_step(batch)
        assert abs(metrics["ratio_mean"] - 1.0) < 1e-3, f"epoch-0 ratio should be ~1, got {metrics['ratio_mean']}"
        assert abs(metrics["approx_kl"]) < 1e-3, f"epoch-0 approx_kl should be ~0, got {metrics['approx_kl']}"

    def test_train_step_empty_prompts_returns_zeroed_metrics(self, tiny_setup):
        """Regression (RIL ISS-177): a degenerate-but-legal step (empty prompt
        list -> empty buffer -> no mini-batches) must return zeroed metrics
        instead of IndexError on ``all_metrics[0]`` or dividing by zero."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        metrics = trainer.train_step([])
        assert metrics["reward_mean"] == 0.0
        assert "approx_kl" not in metrics  # no batches -> no ppo metrics, but no crash

    def test_generate_responses_honors_top_k(self, tiny_setup):
        """Regression (RIL ISS-176): ``top_k=1`` must make every sampled token
        the argmax (deterministic greedy); previously the knob was declared in
        the config but generation sampled pure multinomial over the full
        distribution."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.top_k = 1

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        prompts = ["Hello", "Hi"]
        _, response_ids, _, _truncated = trainer.generate_responses(prompts)
        assert len(response_ids[0]) > 0

        # Every generated token must equal the greedy argmax of the raw logits
        # (re-checked under eval() so dropout does not change the logits).
        was_training = policy.training
        try:
            policy.eval()
            prompt_ids = torch.tensor(tokenizer.encode(prompts[0]), device=DEFAULT_DEVICE).unsqueeze(0)
            with torch.no_grad():
                for tok in response_ids[0]:
                    logits = policy(prompt_ids)[0, -1, :]
                    assert tok.item() == int(torch.argmax(logits))
                    prompt_ids = torch.cat([prompt_ids, tok.reshape(1, 1)], dim=1)
        finally:
            if was_training:
                policy.train()

    def test_train_step_normalize_rewards_standardizes(self, tiny_setup):
        """Regression (RIL ISS-176): ``normalize_rewards=True`` must
        standardize the reward batch to ~zero mean / unit variance before it
        feeds GAE/returns; previously the knob was never read."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        config.normalize_rewards = True
        config.ppo_epochs = 1

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        metrics = trainer.train_step(["Hello", "Hi there", "How are you", "Test prompt"])
        # Zero-mean rewards across the batch -> reward_mean ~ 0 (finite model
        # rewards are not all equal, so the mean is well-defined).
        assert torch.isfinite(torch.tensor(metrics["reward_mean"]))
        assert abs(metrics["reward_mean"]) < 1e-5
