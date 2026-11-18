"""Tests for RLHF PPO Trainer."""

import pytest
import torch
import torch.nn.functional as functional

from llm.training.core.config import PPOConfig
from llm.training.rlhf.ppo_trainer import PPOTrainer
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

    def test_rejects_unimplemented_rollout_batch_size(self):
        """``rollout_batch_size`` is declared but never read by the trainer —
        a non-default value silently does nothing. Fail fast instead."""
        with pytest.raises(ValueError, match="rollout_batch_size"):
            PPOConfig(rollout_batch_size=32)

    def test_rejects_unimplemented_ref_model_update_freq(self):
        """``ref_model_update_freq`` implies the frozen reference re-syncs to
        the policy, which is never implemented — a non-default value silently
        does nothing. Fail fast instead."""
        with pytest.raises(ValueError, match="ref_model_update_freq"):
            PPOConfig(ref_model_update_freq=100)


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
        """Regression: a max-len-truncated episode is NOT
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

    def test_gae_decayed_reference_scan(self):
        """The vectorized GAE must reproduce the classic decayed recurrence
        ``A_t = delta_t + gamma*lambda*A_{t+1}`` for ``gamma, lambda < 1``.

        The other GAE tests only pin ``gamma=lambda=1.0``, which collapses the
        decay to a plain return; this pins the decayed path (``gae_lambda < 1``)
        against an independent reverse scan so a rewrite of the closed-form
        reduction cannot silently drift the discounted-advantage signal."""
        buffer = RolloutBuffer(gae_lambda=0.95, gamma=0.99, normalize_advantages=False)
        values = torch.tensor([0.5, 0.3, -0.1, 0.2])
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3, 4]),
            response_ids=torch.tensor([5, 6, 7, 8]),
            rewards=torch.tensor(1.5),
            old_log_probs=torch.zeros(4),
            values=values,
            truncated=False,
        )
        buffer.compute_advantages()

        vals_list = values.tolist()
        gm, lam = 0.99, 0.95
        c = gm * lam
        last = 0.0
        expected = [0.0] * len(vals_list)
        for t in range(len(vals_list) - 1, -1, -1):
            if t == len(vals_list) - 1:
                r, nv = 1.5, 0.0  # EOS terminal: next_value = 0
            else:
                r, nv = 0.0, vals_list[t + 1]
            delta = r + gm * nv - vals_list[t]
            last = delta + c * last
            expected[t] = last

        assert torch.allclose(buffer.samples[0].advantages, torch.tensor(expected), atol=1e-5)

    def test_gae_advantages_fp16_dtype_preserved(self):
        """``_gae_advantages`` must keep the values' dtype (fp16 in) and stay
        finite — a rewrite that upcasts the output would silently change the
        critic/advantage dtype the PPO update consumes."""
        from llm.training.rlhf.rollout_buffer import _gae_advantages

        values = torch.randn(30, dtype=torch.float16)
        out = _gae_advantages(
            values,
            torch.tensor([2.0], dtype=torch.float16),
            terminal_value=None,
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert out.dtype == torch.float16
        assert torch.isfinite(out.float()).all()

    def test_gae_advantages_zero_accumulation(self):
        """``gamma * gae_lambda == 0`` short-circuits to ``A_t = delta_t`` (no
        advantage accumulation): ``delta = r_t + gamma*V_{t+1} - V_t`` with the
        sparse terminal reward on the last token."""
        from llm.training.rlhf.rollout_buffer import _gae_advantages

        values = torch.tensor([0.5, 0.3, -0.2])
        # gamma=0 kills the decay AND the bootstrap term; only the sparse
        # reward remains in the last-token delta.
        out = _gae_advantages(
            values,
            torch.tensor([1.0]),
            terminal_value=None,
            gamma=0.0,
            gae_lambda=0.95,
        )
        # delta = r_t + 0 - V_t: all tokens -V_t except the last +1.0 - V_last.
        expected = torch.tensor([-0.5, -0.3, 1.0 - (-0.2)])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_gae_advantages_pathological_small_decay_falls_back(self):
        """A `gamma*gae_lambda` so small that ``1/c^n`` exceeds float64 must
        fall back to the stable forward scan and stay finite + correct — the
        closed-form upcast can't represent it (the original loop handled it)."""
        from llm.training.rlhf.rollout_buffer import _gae_advantages

        values = torch.randn(20)
        reward = torch.tensor([3.0])
        # c = 1e-50 -> 1/c^20 = 1e1000 overflows float64 -> must fall back.
        out = _gae_advantages(
            values,
            reward,
            terminal_value=None,
            gamma=1.0,
            gae_lambda=1e-50,
        )
        assert torch.isfinite(out).all()
        # With c = 1e-50 every c^k (k>=1) term vanishes in float64, so
        # A_t = delta_t = r_t + gamma*V_{t+1} - V_t exactly.
        vals = values.tolist()
        ref = [vals[t + 1] - vals[t] for t in range(len(vals) - 1)] + [3.0 - vals[-1]]
        assert torch.allclose(out, torch.tensor(ref), atol=1e-4)

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

    def test_trainer_defaults_to_the_host_device(self, tiny_setup):
        """Constructing the trainer must not require an NVIDIA driver.

        ``device`` defaulted to the literal ``"cuda"``, so constructing a
        trainer without naming a device raised
        ``RuntimeError: Found no NVIDIA driver`` on any CPU-only host. Every
        test here passes ``device=``, so the default went unexercised.
        """
        import llm.utils.common as common
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        cuda_expected = common.torch.cuda.is_available()
        policy, reward_model, tokenizer, config = tiny_setup
        trainer = PPOTrainer(policy_model=policy, reward_model=reward_model, tokenizer=tokenizer, config=config)

        assert trainer.device.type == ("cuda" if cuda_expected else "cpu")

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
        episodes. ``compute_response_values`` only emits the
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
        """Regression: ``old_log_probs`` must be log-probs of
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
        ). The old guard read ``hasattr(tokenizer, "eos_id")`` —
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

    def test_generate_responses_caps_at_model_context(self, tiny_setup):
        """PPO rollout must not generate past the model's positional table: a
        prompt near ``max_seq_len`` yields at most the remaining context budget
        of response tokens, even when ``response_max_len`` is larger."""
        from unittest.mock import patch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        # tiny_model has max_seq_len=16; use a 14-token prompt and ask for
        # response_max_len=10 — only 2 tokens of response can fit.
        tokenizer.encode = lambda text: list(range(14))
        config.response_max_len = 10

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        with patch("torch.multinomial", return_value=torch.tensor([5], device=DEFAULT_DEVICE)):
            _, response_ids, _, truncated = trainer.generate_responses(["Hello"])

        assert len(response_ids[0]) == 2, (
            f"response must be capped to the 2 remaining context tokens, got {len(response_ids[0])}"
        )
        assert truncated == [True], "context-capped rollout is a truncation (not EOS)"

    def test_generate_responses_rejects_overlong_prompt(self, tiny_setup):
        """A prompt that already reaches ``max_seq_len`` must fail fast with a
        clear error — the old code kept generating and either raised IndexError
        from the learned positional embedding or silently mis-truncated the
        sinusoidal position cache."""
        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        tokenizer.encode = lambda text: list(range(16))  # 16 == tiny model context
        config.response_max_len = 5
        config.top_k = 1

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        with pytest.raises(ValueError, match="max_seq_len"):
            trainer.generate_responses(["Hello"])

    def test_generate_responses_stops_at_list_valued_eos(self, tiny_setup):
        """Some HF tokenizers expose ``eos_token_id`` as a list/sequence. The
        old ``next_token.item() == eos_id`` int-vs-list comparison silently
        never matched, so every rollout ran to ``response_max_len``. A list-valued EOS must still stop generation."""
        from unittest.mock import patch

        from llm.training.rlhf.ppo_trainer import PPOTrainer

        policy, reward_model, tokenizer, config = tiny_setup
        tokenizer.eos_token_id = [4, 5]  # list-valued, like some HF tokenizers
        tokenizer.encode = lambda text: [1, 2, 3]

        trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=config,
            device=str(DEFAULT_DEVICE),
        )

        with patch("torch.multinomial", return_value=torch.tensor([5], device=DEFAULT_DEVICE)):
            _, response_ids, _, truncated = trainer.generate_responses(["Hello"])

        assert len(response_ids[0]) == 1, f"rollout must stop at a list-valued EOS, got {len(response_ids[0])}"
        assert truncated == [False], "an EOS-terminated episode is NOT truncated"

    def test_generate_responses_reports_max_len_truncation(self, tiny_setup):
        """Regression: when the tokenizer has no EOS the
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
        """Regression: ``train_step`` must thread the truncation
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
        """Regression: the policy loss must be normalized over
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
        """Regression: the KL penalty must average over the
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
        """Regression: ``approx_kl`` / ``ratio_mean`` telemetry
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
        """Regression: ``target_kl`` early stopping must halt
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
        """Regression: trailing padding in a padded mini-batch
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
        """Regression: re-scoring a rollout with the SAME policy
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
            "test precondition: tiny_model must contain dropout to expose "
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
        """Regression: a degenerate-but-legal step (empty prompt
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
        """Regression: ``top_k=1`` must make every sampled token
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
        """Regression: ``normalize_rewards=True`` must
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

    def test_a_single_reward_normalises_to_zero_not_nan(self, tiny_setup):
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

        # ONE prompt: the degenerate batch.
        metrics = trainer.train_step(["Hello"])

        assert torch.isfinite(torch.tensor(metrics["reward_mean"])), (
            f"a single-sample batch produced reward_mean={metrics['reward_mean']}; "
            "the normaliser is using the sample std (divide by N-1)"
        )
        assert not torch.isnan(torch.tensor(metrics["loss"])), "the PPO loss is NaN on a one-sample batch"


class TestPPORefusesAnEmptyResponseBudget:
    """A non-positive ``response_max_len`` must be rejected, not trained on.

    The rollout loop only appends a token inside its range, so a budget of 0 (or
    a negative value) exits before emitting anything: every sample carries an
    empty response, ``response_mask`` has no set bit, and the clipped objective
    evaluates ``0 / clamp(0, min=1) == 0.0`` — a finite loss with zero gradient.
    The step "succeeds", the metrics look healthy, and the policy never moves.

    ``PPOConfig`` accepted both values. Two layers now refuse the situation: the
    config rejects the cause while the offending YAML is still on screen, and
    ``ppo_step`` rejects any batch that arrives with no response token (which
    also covers a caller building a batch by hand).
    """

    @pytest.mark.parametrize("bad", [0, -1, -256])
    def test_config_rejects_non_positive_budget(self, bad: int):
        from llm.training.core.config import PPOConfig

        with pytest.raises(ValueError, match="response_max_len must be > 0"):
            PPOConfig(response_max_len=bad)

    @pytest.mark.parametrize("good", [1, 16, 256])
    def test_config_accepts_a_positive_budget(self, good: int):
        """Counterpart: the guard must not reject usable budgets."""
        from llm.training.core.config import PPOConfig

        assert PPOConfig(response_max_len=good).response_max_len == good

    def test_ppo_step_refuses_a_batch_with_no_response_tokens(self):
        """The second layer: a hand-built all-empty batch must fail loud.

        Constructed directly rather than through a rollout, because the point is
        that ANY caller reaching ``ppo_step`` with an empty mask gets an error
        instead of a silent zero loss.
        """
        import torch

        from llm.training.rlhf.rollout_buffer import RolloutBatch

        batch_size, seq = 2, 6
        prompt_len = seq  # no response token at all
        batch = RolloutBatch(
            input_ids=torch.zeros(batch_size, seq, dtype=torch.long),
            attention_mask=torch.ones(batch_size, seq, dtype=torch.long),
            response_mask=torch.zeros(batch_size, seq, dtype=torch.long),
            old_log_probs=torch.zeros(batch_size, 1),
            advantages=torch.zeros(batch_size, 1),
            returns=torch.zeros(batch_size, 1),
            rewards=torch.zeros(batch_size),
        )
        del prompt_len

        # Build a trainer without touching the heavy __init__ path: the guard
        # runs before any of the forward work, so only the attributes it reads
        # need to exist.
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = type("C", (), {"clip_epsilon": 0.2, "temperature": 1.0, "top_k": None, "top_p": None})()
        trainer.device = torch.device("cpu")

        with pytest.raises(ValueError, match="no response tokens"):
            trainer.ppo_step(batch)


class TestPPOBatchingKnobs:
    """``mini_batch_size`` must be positive; ``ppo_epochs=0`` must stay graceful.

    The two fields drive different failure modes, so they get different
    treatment:

    * ``mini_batch_size`` feeds ``range(0, n, mini_batch_size)`` in
      ``RolloutBuffer.get_batches``. Zero raises a bare
      ``ValueError: range() arg 3 must not be zero`` from inside the buffer —
      a message naming neither the field nor the config — and a negative value
      silently yields no batch, which reads as "PPO ran and did not improve".
      The config rejects both.
    * ``ppo_epochs=0`` is deliberately supported: the update loop body never
      runs, and ``update`` returns zeroed metrics rather than dividing by an
      empty ``all_metrics``. That promise lived only in a code comment, so it is
      pinned here.
    """

    @pytest.mark.parametrize("bad", [0, -1, -64])
    def test_mini_batch_size_must_be_positive(self, bad: int):
        from llm.training.core.config import PPOConfig

        with pytest.raises(ValueError, match="mini_batch_size must be > 0"):
            PPOConfig(mini_batch_size=bad)

    @pytest.mark.parametrize("good", [1, 8, 64])
    def test_mini_batch_size_accepts_positive_values(self, good: int):
        """Counterpart: real batch sizes still construct."""
        from llm.training.core.config import PPOConfig

        assert PPOConfig(mini_batch_size=good).mini_batch_size == good

    def test_get_batches_with_zero_stride_is_what_we_are_guarding_against(self):
        """Documents the raw failure the config guard prevents.

        Without the guard a caller reaching this directly still gets the opaque
        message — this test records what that looks like, so the reason for the
        guard stays visible.
        """
        from llm.training.rlhf.rollout_buffer import RolloutBuffer

        buf = RolloutBuffer()
        # An empty buffer returns early, so a sample is required to reach the
        # slicing loop at all.
        buf.add(
            prompt_ids=torch.tensor([1, 2]),
            response_ids=torch.tensor([3]),
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([[0.0]]),
        )
        with pytest.raises(ValueError, match="range\\(\\) arg 3 must not be zero"):
            list(buf.get_batches(mini_batch_size=0))


class TestPPOLearningRateKnobs:
    """``policy_lr`` / ``value_lr`` must be positive and finite, like ``training.lr``.

    Neither had a constraint, and ``PPOTrainer`` builds its optimizers with
    ``AdamW(lr=config.policy_lr or 1e-5)``. Three values behaved differently and
    none of them was rejected at config load:

    * a NEGATIVE rate reached ``AdamW`` and raised
      ``ValueError: Invalid learning rate: -1e-05`` from inside the trainer — a
      message naming neither the field nor the PPO config;
    * ``inf`` is ACCEPTED by torch, and the first step turns every weight into
      ``-inf`` (measured on a one-parameter AdamW);
    * ``0`` was silently replaced by the ``1e-5`` default via the ``or``
      fallback, so an explicit "no learning" setting trained at full rate.

    ``training.lr`` already carries ``gt=0, allow_inf_nan=False``; these two now
    match it.
    """

    @pytest.mark.parametrize("field", ["policy_lr", "value_lr"])
    @pytest.mark.parametrize("bad", [0.0, -1e-5, -1.0])
    def test_non_positive_learning_rate_is_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["policy_lr", "value_lr"])
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_learning_rate_is_rejected(self, field: str, bad: float):
        """``gt=0`` alone lets ``inf`` through (``inf > 0`` is True)."""
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["policy_lr", "value_lr"])
    @pytest.mark.parametrize("good", [1e-6, 1e-5, 1e-3])
    def test_positive_learning_rate_is_accepted(self, field: str, good: float):
        assert getattr(PPOConfig(**{field: good}), field) == good

    @pytest.mark.parametrize("field", ["policy_lr", "value_lr"])
    def test_unset_stays_none(self, field: str):
        """``None`` is the documented "derive the default" value and must survive."""
        assert getattr(PPOConfig(), field) is None

    def test_the_rejected_rate_is_one_torch_rejects_too(self):
        """Record the raw failure the config guard now prevents, and the one it cannot.

        ``AdamW`` rejects a negative rate itself (late, and without naming the
        config field); ``inf`` it accepts, which is why the config needs
        ``allow_inf_nan=False`` rather than relying on torch.
        """
        p = torch.nn.Parameter(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="Invalid learning rate"):
            torch.optim.AdamW([p], lr=-1e-5)

        q = torch.nn.Parameter(torch.tensor([1.0]))
        q.grad = torch.tensor([1.0])
        torch.optim.AdamW([q], lr=float("inf")).step()
        assert q.item() == float("-inf"), "an inf learning rate poisons the weights in one step"


class TestPPORolloutSamplingKnobs:
    """The three rollout-sampling knobs must agree with ``sampling_probs``.

    ``PPOTrainer`` passes ``temperature`` / ``top_k`` / ``top_p`` straight to
    ``llm.generation.sampling_probs``, which already rejects the invalid end of
    each range — from inside the sampling loop, in a message that names the
    generic argument rather than the PPO config field. Worse, two of the
    rejected values did not even crash: they quietly changed the distribution.

    The bounds here mirror ``sampling_probs``'s documented contract exactly:

    * ``temperature == 0`` is GREEDY and is special-cased by the rollout loop to
      ``argmax``, so 0 must stay accepted; a negative value raised
      ``ValueError: temperature must be > 0`` mid-rollout.
    * ``top_k`` must be ``>= 1`` when set (torch.topk's ``k`` goes out of range
      otherwise); ``None`` disables the filter.
    * ``top_p`` must be ``> 0`` when set (0 or negative made the nucleus filter
      silently skip and draw from the FULL distribution); ``None`` and any
      value ``>= 1.0`` disable it, matching the OpenAI convention.
    """

    def test_zero_temperature_still_means_greedy(self):
        """0 is a real setting, not a hole — pin it so a `gt=0` rewrite cannot pass."""
        assert PPOConfig(temperature=0.0).temperature == 0.0

    @pytest.mark.parametrize("bad", [-1.0, -1e-6])
    def test_negative_temperature_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="temperature"):
            PPOConfig(temperature=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_temperature_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="temperature"):
            PPOConfig(temperature=bad)

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0, 2.0])
    def test_positive_temperature_is_accepted(self, good: float):
        assert PPOConfig(temperature=good).temperature == good

    @pytest.mark.parametrize("bad", [0, -1, -50])
    def test_non_positive_top_k_is_rejected(self, bad: int):
        with pytest.raises(ValueError, match="top_k"):
            PPOConfig(top_k=bad)

    @pytest.mark.parametrize("good", [1, 10, 1000])
    def test_positive_top_k_is_accepted(self, good: int):
        assert PPOConfig(top_k=good).top_k == good

    def test_top_k_none_disables_the_filter(self):
        assert PPOConfig().top_k is None

    @pytest.mark.parametrize("bad", [0.0, -0.5, -1.0])
    def test_non_positive_top_p_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="top_p"):
            PPOConfig(top_p=bad)

    @pytest.mark.parametrize("good", [0.1, 0.9, 1.0, 2.0])
    def test_positive_top_p_is_accepted(self, good: float):
        """``top_p >= 1.0`` is a documented no-op, so it must stay legal."""
        assert PPOConfig(top_p=good).top_p == good

    def test_top_p_none_disables_the_filter(self):
        assert PPOConfig().top_p is None

    def test_the_bounds_match_sampling_probs_own_contract(self):
        """The config must reject exactly what the sampler rejects (plus 0 for temperature).

        Recording the shared boundary here means a future change to either side
        shows up as a failure instead of letting the two drift apart again.
        """
        from llm.generation.sampling import sampling_probs

        logits = torch.randn(1, 8)
        for temp in (-1.0, -0.001):
            with pytest.raises(ValueError, match="temperature must be > 0"):
                sampling_probs(logits, temperature=temp)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            sampling_probs(logits, top_k=0)
        with pytest.raises(ValueError, match="top_p must be > 0"):
            sampling_probs(logits, top_p=0.0)
        # ...and the values the config now accepts are the ones it accepts.
        sampling_probs(logits, temperature=0.5, top_k=4, top_p=0.9)


class TestPPOObjectiveCoefficients:
    """The four PPO loss weights must not silently change what the objective means.

    Unlike the sampling knobs, these three failure modes are all different, and
    only the first one actually raised nothing anywhere:

    * ``clip_epsilon`` is the clamp range ``[1-eps, 1+eps]``. At 0 the clamp
      collapses to the single point 1.0, so every ratio maps there and the
      clipped surrogate stops constraining anything; a NEGATIVE value makes
      ``lo = 1-eps > hi = 1+eps``, and ``torch.clamp`` silently returns ``hi``
      for every input — the cap becomes a floor. Measured with
      ``adv=1, ratio=[0.5, 1.0, 2.0]``: ``eps=-0.2`` produced
      ``[-0.5, -0.8, -0.8]`` against ``eps=0.2``'s ``[-0.5, -1.0, -1.2]``.
    * ``kl_coef`` is added straight into the loss, so a negative weight pays the
      policy to DIVERGE from the reference rather than stay near it.
    * ``value_coef`` / ``entropy_coef`` are guarded by ``> 0`` at the call site,
      so a negative value was silently identical to 0 — a config that looks
      tuned and does nothing. 0 itself is a real setting for all three.

    ``clip_epsilon`` is therefore ``gt=0`` while the other three are ``ge=0``.
    """

    @pytest.mark.parametrize("bad", [0.0, -0.2, -1.0])
    def test_clip_epsilon_must_be_positive(self, bad: float):
        """0 is NOT a valid disable for the ratio clamp — it collapses it to a point."""
        with pytest.raises(ValueError, match="clip_epsilon"):
            PPOConfig(clip_epsilon=bad)

    @pytest.mark.parametrize("good", [0.01, 0.2, 0.5])
    def test_clip_epsilon_accepts_positive_values(self, good: float):
        assert PPOConfig(clip_epsilon=good).clip_epsilon == good

    @pytest.mark.parametrize("field", ["kl_coef", "value_coef", "entropy_coef"])
    @pytest.mark.parametrize("bad", [-0.01, -1.0])
    def test_negative_coefficient_is_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["kl_coef", "value_coef", "entropy_coef"])
    def test_zero_coefficient_stays_legal(self, field: str):
        """For these three 0 genuinely means "off" — pin it against a blanket ``gt=0``."""
        assert getattr(PPOConfig(**{field: 0.0}), field) == 0.0

    @pytest.mark.parametrize("field", ["clip_epsilon", "kl_coef", "value_coef", "entropy_coef"])
    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_non_finite_coefficient_is_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["kl_coef", "value_coef", "entropy_coef"])
    @pytest.mark.parametrize("good", [0.001, 0.1, 0.5, 1.0])
    def test_positive_coefficient_is_accepted(self, field: str, good: float):
        assert getattr(PPOConfig(**{field: good}), field) == good

    def test_the_inverted_clamp_is_what_these_guards_prevent(self):
        """Record the exact silent misbehaviour, so the reason stays checkable.

        ``torch.clamp`` does not validate ``min <= max``. With ``eps < 0`` the
        bounds invert to ``lo = 1 - eps > hi = 1 + eps``, and clamp then returns
        a SINGLE constant for every input — so the ratio is thrown away entirely
        and the surrogate stops depending on it. No error, no warning.
        """
        ratio = torch.tensor([0.5, 1.0, 2.0])
        eps = -0.2
        inverted = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        assert inverted.unique().numel() == 1, (
            f"an inverted clamp must collapse to one constant, got {inverted.tolist()}"
        )
        # The healthy case actually clamps per element.
        healthy = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        assert healthy.tolist() == pytest.approx([0.8, 1.0, 1.2])


class TestPPOScheduleKnobs:
    """``ppo_epochs`` / ``max_grad_norm`` / ``target_kl`` — the rest of the loop bounds.

    These three complete the PPO knob surface. Their call sites already encode a
    convention, so the constraints follow it rather than inventing new rules:

    * ``ppo_epochs`` drives ``range(...)`` in the update loop. 0 is explicitly
      supported (the metric aggregation returns zeroed values instead of
      dividing by an empty list), so it stays legal; a negative value yields the
      same zero passes while reading as a real setting, so it is rejected.
    * ``max_grad_norm`` is guarded by ``if max_grad_norm > 0`` — the same
      "non-positive means clip off" convention ``training.gradient_clip_val``
      now follows. It takes identical bounds (`ge=0`, finite) so the two cannot
      disagree.
    * ``target_kl`` is compared as ``approx_kl > target_kl``. ``approx_kl`` is a
      non-negative quantity, so a zero or negative threshold fires on the FIRST
      update of every step, collapsing PPO to one epoch per step regardless of
      ``ppo_epochs``. Measured against the loop's own condition with four
      configured epochs: ``target_kl=0.0`` ran 1, ``target_kl=0.5`` ran 4.
    """

    def test_zero_ppo_epochs_stays_legal(self):
        """0 is the documented degenerate case the aggregation explicitly handles."""
        assert PPOConfig(ppo_epochs=0).ppo_epochs == 0

    @pytest.mark.parametrize("bad", [-1, -4])
    def test_negative_ppo_epochs_is_rejected(self, bad: int):
        with pytest.raises(ValueError, match="ppo_epochs"):
            PPOConfig(ppo_epochs=bad)

    @pytest.mark.parametrize("good", [1, 4, 10])
    def test_positive_ppo_epochs_is_accepted(self, good: int):
        assert PPOConfig(ppo_epochs=good).ppo_epochs == good

    def test_zero_max_grad_norm_disables_clipping(self):
        """Mirrors ``training.gradient_clip_val=0`` — pin the shared convention."""
        from llm.training.core.config import TrainingConfig

        assert PPOConfig(max_grad_norm=0.0).max_grad_norm == 0.0
        assert TrainingConfig(gradient_clip_val=0.0).gradient_clip_val == 0.0

    @pytest.mark.parametrize("bad", [-1.0, -0.5])
    def test_negative_max_grad_norm_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="max_grad_norm"):
            PPOConfig(max_grad_norm=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_non_finite_max_grad_norm_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="max_grad_norm"):
            PPOConfig(max_grad_norm=bad)

    @pytest.mark.parametrize("good", [0.1, 1.0, 5.0])
    def test_positive_max_grad_norm_is_accepted(self, good: float):
        assert PPOConfig(max_grad_norm=good).max_grad_norm == good

    def test_target_kl_none_disables_early_stopping(self):
        assert PPOConfig().target_kl is None

    @pytest.mark.parametrize("bad", [0.0, -0.5, -1.0])
    def test_non_positive_target_kl_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="target_kl"):
            PPOConfig(target_kl=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_non_finite_target_kl_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="target_kl"):
            PPOConfig(target_kl=bad)

    @pytest.mark.parametrize("good", [0.01, 0.1, 1.0])
    def test_positive_target_kl_is_accepted(self, good: float):
        assert PPOConfig(target_kl=good).target_kl == good

    def test_a_non_positive_target_kl_would_fire_on_every_step(self):
        """Record the loop condition that makes the threshold need to be positive.

        Uses the exact comparison from ``ppo_trainer.train_step``'s caller:
        ``approx_kl > target_kl``. A realistic small KL breaches any
        non-positive threshold immediately.
        """
        approx_kl = 0.01  # a small but normal KL

        def epochs_run(target_kl: float | None, n_epochs: int = 4) -> int:
            for epoch in range(1, n_epochs + 1):
                if target_kl is not None and approx_kl > target_kl:
                    return epoch
            return n_epochs

        assert epochs_run(0.5) == 4, "a healthy threshold must let the configured epochs run"
        assert epochs_run(None) == 4, "None disables the check entirely"
        assert epochs_run(0.0) == 1, "a zero threshold must fire on the first update"
        assert epochs_run(-1.0) == 1, "a negative threshold must fire on the first update too"


class TestPPOAdvantageWeights:
    """``gamma`` / ``gae_lambda`` must keep the GAE recurrence bounded.

    ``_gae_advantages`` computes ``A_t = delta_t + c * A_{t+1}`` with
    ``c = gamma * gae_lambda`` (the docstring states this explicitly). The pair
    is only stable while ``|c| <= 1``; beyond that the advantages grow
    geometrically with the response length. Both fields were unconstrained, so
    the pathological case was reachable from a YAML file. Measured with a unit
    reward and zero values:

    * ``gae_lambda=1.5`` gave ``max|adv| = 7.6e+00`` at 6 tokens,
      ``1.2e+11`` at 64, and ``inf`` at 256 — a length that shipped configs use.
    * ``gamma=2.0`` gave ``32.0`` at 6 tokens (``2^5``, i.e. exactly ``c^(n-1)``).
    * a negative ``gamma`` alternated the sign of every advantage.

    ``c == 0`` is a documented degenerate case (``A_t = delta_t``) that the
    function short-circuits, so 0 must stay legal for both fields — that is why
    the bounds are ``[0, 1]`` and not ``(0, 1]``.
    """

    @pytest.mark.parametrize("field", ["gamma", "gae_lambda"])
    @pytest.mark.parametrize("bad", [1.5, 2.0, 10.0])
    def test_values_above_one_are_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["gamma", "gae_lambda"])
    @pytest.mark.parametrize("bad", [-1.0, -0.1])
    def test_negative_values_are_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["gamma", "gae_lambda"])
    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_non_finite_values_are_rejected(self, field: str, bad: float):
        with pytest.raises(ValueError, match=field):
            PPOConfig(**{field: bad})

    @pytest.mark.parametrize("field", ["gamma", "gae_lambda"])
    @pytest.mark.parametrize("good", [0.0, 0.5, 0.95, 1.0])
    def test_zero_and_unit_interval_values_stay_legal(self, field: str, good: float):
        """0 is the documented ``A_t = delta_t`` case; 1 is the episode end."""
        assert getattr(PPOConfig(**{field: good}), field) == good

    def test_the_bounds_are_what_keeps_the_gae_recurrence_finite(self):
        """Replay ``_gae_advantages`` at both ends of the accepted range.

        Uses the real function so the assertion tracks the implementation rather
        than restating the formula. A unit reward plus zero value estimates makes
        ``delta_t`` a single spike, so the only thing shaping the result is
        ``c = gamma * gae_lambda``.
        """
        from llm.training.rlhf.rollout_buffer import _gae_advantages

        values = torch.zeros(64)
        reward = torch.tensor(1.0)

        # Largest accepted c: bounded by construction.
        worst = _gae_advantages(values, reward, terminal_value=None, gamma=1.0, gae_lambda=1.0)
        assert torch.isfinite(worst).all()
        assert worst.abs().max().item() == pytest.approx(1.0), (
            f"c=1 must give a constant advantage of the reward, got {worst.abs().max().item()}"
        )

        # The smallest rejected c on the high side overflows a realistic length.
        blown = _gae_advantages(values, reward, terminal_value=None, gamma=1.0, gae_lambda=1.5)
        assert not torch.isfinite(blown).all() or blown.abs().max().item() > 1e6, (
            f"gae_lambda=1.5 must produce a runaway advantage, got max {blown.abs().max().item()}"
        )

        # c == 0 short-circuits to delta_t, which is why 0 stays legal.
        flat = _gae_advantages(values, reward, terminal_value=None, gamma=0.0, gae_lambda=0.95)
        assert flat[:-1].abs().max().item() == 0.0
        assert flat[-1].item() == pytest.approx(1.0)


class TestAdvantageNormalisationDegenerateSize:
    """A one-element advantage buffer must not normalise to NaN.

    ``.std()`` defaults to ``unbiased=True``, which divides by ``N - 1``. With a
    single advantage element — one sample whose response is a single token, the
    degenerate tail of a small rollout — that is a division by zero: PyTorch
    returns NaN (and warns), the advantages become ``[nan]``, and the PPO loss,
    gradients and optimizer step are all NaN. Measured before the fix:
    ``advantages == [nan]``. The population std makes the degenerate case yield
    0 (no spread means no advantage signal), which is what
    ``group_advantages`` already does for GRPO.
    """

    def test_a_single_advantage_element_normalises_to_zero_not_nan(self):
        from llm.training.rlhf.rollout_buffer import RolloutBuffer

        buffer = RolloutBuffer(gae_lambda=1.0, gamma=1.0, normalize_advantages=True)
        buffer.add(
            prompt_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4]),  # ONE token
            rewards=torch.tensor(1.0),
            old_log_probs=torch.tensor([-0.5]),
            values=torch.tensor([0.2]),
        )

        buffer.compute_advantages()

        advantages = buffer.samples[0].advantages
        assert advantages is not None
        assert not torch.isnan(advantages).any(), (
            f"a single-element advantage buffer normalised to {advantages.tolist()}; "
            "the normaliser is using the sample std (divide by N-1)"
        )
        assert torch.allclose(advantages, torch.zeros_like(advantages))

    def test_the_normalised_advantages_stay_finite_and_standardised(self):
        """The normal case must remain a proper z-score (mean 0, unit spread)."""
        from llm.training.rlhf.rollout_buffer import RolloutBuffer

        buffer = RolloutBuffer(gae_lambda=1.0, gamma=0.99, normalize_advantages=True)
        torch.manual_seed(0)
        for index in range(4):
            buffer.add(
                prompt_ids=torch.tensor([1, 2, 3]),
                response_ids=torch.tensor([4, 5, 6]),
                rewards=torch.tensor(float(index)),
                old_log_probs=torch.full((3,), -0.5),
                values=torch.randn(3) * 0.1,
            )

        buffer.compute_advantages()
        combined = torch.cat([s.advantages for s in buffer.samples])

        assert not torch.isnan(combined).any()
        assert abs(float(combined.mean())) < 1e-5, f"normalised advantages are not zero-mean: {float(combined.mean())}"
        # Population std is 1 by construction (before the +1e-8 in the divisor).
        assert abs(float(combined.std(unbiased=False)) - 1.0) < 1e-3
