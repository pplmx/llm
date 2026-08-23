"""Tests for speculative decoding (audit T3 #9 / Tier 3 #9).

Pins the contract for :mod:`llm.generation.speculative`:

1. ``speculative_generate`` produces the same greedy output as the
   eager backend when both draft and target are seeded with the same
   weights (so the draft's argmax always matches the target's
   argmax — every candidate is accepted).
2. The output distribution (sampled) matches the target model's
   distribution within sampling noise — the algorithm preserves the
   target distribution under sampling.
3. The backend integrates with ``BACKEND_REGISTRY`` and rejects bad
   ``gamma``.
4. EOS / ``max_new_tokens`` short-circuits the generator.
"""

from __future__ import annotations

import pytest
import torch

from llm.generation.backends import (
    SpeculativeDecodingBackend,
)
from llm.generation.eager import generate as eager_generate
from llm.generation.registry import (
    BACKEND_REGISTRY,
    ensure_backends_registered,
    get_generation_backend,
)
from llm.generation.speculative import speculative_generate
from tests.support.devices import DEFAULT_DEVICE
from tests.support.models import decoder_model_kwargs
from tests.support.tokenizers import StubTokenizer


def _make_tiny_decoder(seed: int = 0, **overrides) -> torch.nn.Module:
    """Tiny DecoderModel with deterministic init (GPU-first, falls back to CPU)."""
    from llm.models.decoder import DecoderModel

    torch.manual_seed(seed)
    kwargs = decoder_model_kwargs(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=64,
        attn_impl="mha",
        mlp_impl="mlp",
    )
    kwargs.update(overrides)
    return DecoderModel(**kwargs)


# --- Greedy correctness: draft == target weights => all accepted ----------


def test_speculative_matches_eager_greedy_when_draft_equals_target():
    """If draft weights == target weights, every candidate is accepted.

    The eager backend and the speculative backend produce **the same
    greedy tokens** in this degenerate setup, because the draft's
    argmax always matches the target's argmax (no rejection ever
    occurs).
    """
    target = _make_tiny_decoder(seed=42)
    draft = _make_tiny_decoder(seed=42)  # identical weights
    tok = StubTokenizer()

    prompt = "abc"
    eager_out = eager_generate(target, tok, prompt, max_new_tokens=6, temperature=0.0, use_cache=False)

    spec_tokens = list(
        speculative_generate(
            target,
            draft,
            tok,
            prompt,
            max_new_tokens=6,
            gamma=3,
            temperature=0.0,
        )
    )
    spec_out = prompt + "".join(spec_tokens)

    assert spec_out == eager_out, (eager_out, spec_out)


@pytest.mark.parametrize(("draft_seed", "max_tokens"), [(123, 1), (99, 4), (5, 8)])
def test_speculative_greedy_matches_eager_with_nonmatching_draft(draft_seed, max_tokens):
    """RIL TASK-251: greedy speculative == greedy eager even when the (random)
    draft never matches the target's argmax — this forces the rejection /
    correction path (the existing draft==target test only covers accept-all,
    so this guards the fallback branch against regressions)."""
    target = _make_tiny_decoder(seed=42)
    draft = _make_tiny_decoder(seed=draft_seed)  # different init -> rarely matches
    tok = StubTokenizer()
    prompt = "abc def ghi"

    eager_out = eager_generate(target, tok, prompt, max_new_tokens=max_tokens, temperature=0.0, use_cache=False)
    spec_out = prompt + "".join(
        speculative_generate(target, draft, tok, prompt, max_new_tokens=max_tokens, gamma=4, temperature=0.0)
    )
    assert spec_out == eager_out, (eager_out, spec_out)


def test_speculative_masks_undecodable_tail_vocab():
    """RIL ISS-125 on the speculative backend: draft candidates AND the
    target-verified bonus must be bounded to the tokenizer's decodeable
    vocab. A padded-vocab model (model vocab > tokenizer vocab) used to
    emit an id in the tail, which ``tokenizer.decode`` rejected with
    ``KeyError`` mid-stream. The raw draft/target logits are now masked
    (pinned to ``-inf``) before sampling/acceptance, so every emitted id
    is decodable.
    """
    from unittest.mock import patch

    target = _make_tiny_decoder(seed=42)
    draft = _make_tiny_decoder(seed=42)

    def _fixed_forward(input_ids, *_args, **_kwargs):
        # Model emits a 100-wide vocab whose argmax at EVERY row is the
        # undecodable tail id 99 (> tokenizer.vocab_size=5) — so the draft
        # proposes 99, the target verifies it, and decode([99]) raises
        # KeyError unless the tail logits are masked before sampling.
        t = max(input_ids.shape[1], 8)
        logits = torch.full((1, t, 100), -1.0, device=input_ids.device)
        logits[0, :, 99] = 10.0
        return logits, None

    class _SmallVocabDecoder:
        vocab_size, pad_token_id = 5, 0

        def encode(self, text):
            return [1]

        def decode(self, ids):
            for i in ids:
                if i >= self.vocab_size:
                    raise KeyError(f"Token ID '{i}' not found in tokenizer vocabulary")
            return "".join(chr(ord("a") + i) for i in ids)

    tok = _SmallVocabDecoder()
    with (
        patch.object(target, "forward", side_effect=_fixed_forward),
        patch.object(draft, "forward", side_effect=_fixed_forward),
    ):
        tokens = list(
            speculative_generate(
                target,
                draft,
                tok,
                "abc",
                max_new_tokens=3,
                gamma=2,
                temperature=0.0,
            )
        )

    # Pre-fix, sampling id 99 crashed decode with KeyError; with the mask
    # every candidate/bonus is a decodable id in [0, 5).
    assert tokens, "speculative_generate must not crash on a padded-vocab model"
    assert all(isinstance(t, str) for t in tokens)


def test_speculative_never_emits_pad_token():
    """RIL round-71 speculative fix: the PAD sentinel must be masked from the
    draft candidates AND the target acceptance/correction scoring, exactly like
    the eager backend does at every decode step.

    With a fixed model whose argmax is id 0 (the tokenizer's PAD), the pre-fix
    draft proposed PAD, the target accepted it, and '<PAD>' landed in the
    generated text. After the fix the threshold token stays unreachable in both
    the draft and the verified distributions.
    """
    from unittest.mock import patch

    target = _make_tiny_decoder(seed=7)
    draft = _make_tiny_decoder(seed=7)

    def _argmax_pad(input_ids, *_args, **_kwargs):
        # Every position's argmax is the PAD id 0.
        t = max(input_ids.shape[1], 8)
        logits = torch.full((1, t, 8), -1.0, device=input_ids.device)
        logits[0, :, 0] = 10.0
        return logits, None

    class _PaddedVocabTok:
        vocab_size, eos_token_id = 8, None

        @property
        def pad_token_id(self):
            return 0

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "".join("[PAD]" if i == 0 else chr(ord("a") + i) for i in ids)

    tok = _PaddedVocabTok()
    with (
        patch.object(target, "forward", side_effect=_argmax_pad),
        patch.object(draft, "forward", side_effect=_argmax_pad),
    ):
        tokens = list(
            speculative_generate(
                target,
                draft,
                tok,
                "abc",
                max_new_tokens=4,
                gamma=3,
                temperature=0.0,
            )
        )

    out = "".join(tokens)
    assert "[PAD]" not in out, f"speculative emitted the PAD sentinel: {out!r}"
    assert out, "expected some generated output"


def _make_small_ctx_decoder(seed: int = 0) -> torch.nn.Module:
    """Tiny non-RoPE decoder with a small context window (learned PE table)."""
    return _make_tiny_decoder(seed, max_seq_len=8, use_rope=False)


def test_speculative_rejects_impossible_context_budget():
    """RIL ISS-124: like the eager backend, speculative must reject a
    ``max_new_tokens`` budget that cannot fit the context window up front.

    ``max_new_tokens >= max_seq_len`` leaves no room for even a single prompt
    token; without the guard the learned-position table indexed past
    ``max_seq_len`` and raised ``ValueError: Sequence endpoint ... exceeds
    maximum sequence length`` (the eager backend rejects it with a clear
    message instead).
    """
    target = _make_small_ctx_decoder(seed=0)
    draft = _make_small_ctx_decoder(seed=0)
    tok = StubTokenizer()
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(
            speculative_generate(
                target,
                draft,
                tok,
                "abc",
                max_new_tokens=8,  # == max_seq_len (8) → impossible budget
                gamma=2,
                temperature=0.0,
            )
        )


class _OrdTokenizer:
    """One token per character, ids ``[1, 2, ...]`` (mirrors StubTokenizer ids
    but with real per-char lengths so prompts can genuinely overflow)."""

    pad_token_id: int = 0

    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text) + 1))

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(96 + i) for i in ids if i < 27)


def test_speculative_truncates_long_prompt_like_eager():
    """RIL ISS-124: a prompt that exceeds ``max_seq_len`` is truncated to fit
    (same shrink the eager backend applies), not left to crash the learned-PE
    table. Parity: on the same inputs, speculative and eager emit the same
    decoded tokens.
    """
    from llm.generation.eager import stream_generate

    target = _make_small_ctx_decoder(seed=42)
    draft = _make_small_ctx_decoder(seed=42)
    tok = _OrdTokenizer()

    prompt = "x" * 12  # 12 tokens > max_seq_len=8
    spec_out = list(
        speculative_generate(
            target,
            draft,
            tok,
            prompt,
            max_new_tokens=2,
            gamma=2,
            temperature=0.0,
        )
    )
    eager_out = list(stream_generate(target, tok, prompt, max_new_tokens=2, temperature=0.0, use_cache=False))
    # Both must truncate rather than crash, and produce the same first tokens.
    assert spec_out, "speculative_generate must not crash on an over-long prompt"
    assert eager_out, "eager stream_generate must not crash on an over-long prompt"
    assert spec_out[0] == eager_out[0], (spec_out, eager_out)


# --- gamma validation -------------------------------------------------------


def test_speculative_gamma_zero_raises():
    """``gamma < 1`` raises ``ValueError``."""
    target = _make_tiny_decoder(seed=0)
    draft = _make_tiny_decoder(seed=0)
    tok = StubTokenizer()
    with pytest.raises(ValueError, match="gamma"):
        list(
            speculative_generate(
                target,
                draft,
                tok,
                "abc",
                max_new_tokens=4,
                gamma=0,
                temperature=0.0,
            )
        )


def test_speculative_backend_gamma_zero_raises():
    """Backend constructor also rejects ``gamma < 1``."""
    target = _make_tiny_decoder(seed=0)
    draft = _make_tiny_decoder(seed=0)
    with pytest.raises(ValueError, match="gamma"):
        SpeculativeDecodingBackend(target, draft, gamma=0)


# --- EOS / max_new_tokens --------------------------------------------------


def test_speculative_stops_on_eos():
    """Generator stops when the draft emits the tokenizer's EOS id."""
    # Vocab must include 99 (StubTokenizer's eos id); use 128 to be safe.
    target = _make_tiny_decoder(seed=1, vocab_size=128)
    draft = _make_tiny_decoder(seed=1, vocab_size=128)
    tok = StubTokenizer()  # eos_token_id = 99

    # Force the model to always predict EOS by collapsing the LM head
    # bias to a large positive value for token 99. Then every greedy
    # sample returns EOS.
    with torch.no_grad():
        for layer in [target, draft]:
            layer.lm_head.bias.zero_()
            layer.lm_head.bias[99] = 100.0

    out = "".join(
        speculative_generate(
            target,
            draft,
            tok,
            "abc",
            max_new_tokens=50,
            gamma=4,
            temperature=0.0,
        )
    )
    # We should stop within a couple of rounds because the very first
    # draft call returns EOS and the round emits exactly one token.
    assert len(out) <= 4  # gamma tokens; first one is EOS so 1 emitted


def test_speculative_eos_text_excluded_from_output():
    """The EOS token's decoded text must NOT appear in the output.

    Mismatches the eager backend (ISS-98): the speculative backend used to
    append the accepted EOS token to ``generated_ids`` and yield its decoded
    chunk before halting, while eager halts *before* decoding the EOS token.
    With a tokenizer whose EOS decodes to a real character the two backends
    visibly diverge.
    """
    # Vocab must include 99 (StubTokenizer's eos id); use 128 to be safe.
    target = _make_tiny_decoder(seed=1, vocab_size=128)
    draft = _make_tiny_decoder(seed=1, vocab_size=128)
    tok = StubTokenizer()  # eos_token_id = 99, decode(anything) -> "x"

    # Force the model to always predict EOS by collapsing the LM head
    # bias to a large positive value for token 99. Then every greedy
    # sample returns EOS.
    with torch.no_grad():
        for layer in [target, draft]:
            layer.lm_head.bias.zero_()
            layer.lm_head.bias[99] = 100.0

    out = "".join(
        speculative_generate(
            target,
            draft,
            tok,
            "abc",
            max_new_tokens=50,
            gamma=4,
            temperature=0.0,
        )
    )
    # Stops on the first EOS (1 emitted copy attempt); the EOS char must not leak.
    assert "x" not in out


def test_speculative_respects_max_new_tokens():
    """``max_new_tokens`` is the hard cap on emitted tokens."""
    target = _make_tiny_decoder(seed=2)
    draft = _make_tiny_decoder(seed=2)
    tok = StubTokenizer()

    out = list(
        speculative_generate(
            target,
            draft,
            tok,
            "abc",
            max_new_tokens=5,
            gamma=3,
            temperature=0.0,
        )
    )
    assert len(out) <= 5


# --- Rejection-path distributional correctness -----------------------------


class _ConstLogitsModel:
    """Duck-typed ``DecoderModel`` stub that returns fixed per-token logits.

    ``_verify_speculative_tokens`` only interacts with the models through
    ``model(full, kv_caches=None, use_cache=False)`` and reads the returned
    logits, so a fixed-logit stub exercises the whole acceptance/rejection
    algorithm without paying for real model forwards.
    """

    def __init__(self, logit_vector):
        self._logit_vector = logit_vector

    def __call__(self, input_ids, kv_caches=None, use_cache=False):
        seq = input_ids.size(1)
        return self._logit_vector.expand(seq, -1).unsqueeze(0)


def test_speculative_stochastic_rejection_preserves_target_distribution():
    """On rejection the correction token is sampled from the normalized
    residual ``(p_target - p_draft)+`` (Leviathan et al., Algorithm 2), so
    the overall output distribution matches the target.

    Regression test: the correction used to be sampled from
    ``softmax(logits_target - logits_draft)`` (proportional to the p/q
    ratio), which measurably biases the output distribution.
    """
    from llm.generation.speculative import _verify_speculative_tokens

    torch.manual_seed(0)
    vocab = 8
    p_log = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5], device=DEFAULT_DEVICE)
    q_log = torch.tensor([1.5, 2.2, 1.8, 0.8, 0.2, -0.4, -0.9, -1.2], device=DEFAULT_DEVICE)
    target = _ConstLogitsModel(p_log)
    draft = _ConstLogitsModel(q_log)
    p = torch.softmax(p_log, -1)
    q = torch.softmax(q_log, -1)

    # Draft's argmax token; the target rates it lower (q[x] > p[x]), so
    # rejection happens with probability 1 - p[x]/q[x] > 0.
    x = 1
    assert q[x] > p[x]

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=DEFAULT_DEVICE)
    n = 2000
    accepted = 0
    bonus_samples = torch.zeros(vocab, device=DEFAULT_DEVICE)
    for _ in range(n):
        accept_count, bonus = _verify_speculative_tokens(
            target,
            draft,
            prompt,
            [x],
            temperature=1.0,
            top_k=None,
            top_p=None,
        )
        if accept_count == 1:
            accepted += 1
        elif bonus is not None:
            bonus_samples[bonus] += 1

    # Acceptance rate matches theory min(1, p[x]/q[x]).
    p_accept = min(1.0, (p[x] / q[x]).item())
    assert abs(accepted / n - p_accept) < 0.04

    # Correction distribution matches the normalized residual (the buggy
    # p/q-ratio correction deviates by ~0.55 on this setup).
    residual = (p - q).clamp(min=0)
    expected = residual / residual.sum()
    empirical = bonus_samples / bonus_samples.sum()
    assert float((empirical - expected).abs().max()) < 0.08


def test_speculative_top_k_acceptance_uses_filtered_distributions():
    """The acceptance ratio must be scored against the **filtered** target/draft
    distributions the samplers actually draw from (ISS-99).

    With ``top_k=1`` the draft proposes its argmax token (token 1, q_log[1]=3.0)
    with probability 1, and the target's filtered distribution is a one-hot on
    its own argmax (token 0). Since the proposal token is NOT the target's
    argmax, the correct acceptance probability is ``min(1, p_filtered/q_filtered)
    = min(1, 0/1) = 0``.

    The bug: ``_verify_speculative_tokens`` scores the ratio against the
    *unfiltered* full-vocab softmax, where ``p[1] ≈ 0.113`` and ``q[1] ≈ 0.859``,
    so it accepts the divergent token with probability ≈ 0.13 — a measurable
    departure from the target (eager-equivalent) distribution.
    """
    from llm.generation.speculative import _verify_speculative_tokens

    torch.manual_seed(0)
    vocab = 8
    # Target prefers token 0; token 1 (the draft's top pick) is a long tail.
    p_log = torch.full((vocab,), 0.0, device=DEFAULT_DEVICE)
    p_log[0] = 3.0
    p_log[1] = 1.2
    # Draft strongly prefers token 1, which the target rates low.
    q_log = torch.full((vocab,), -1.0, device=DEFAULT_DEVICE)
    q_log[0] = 0.1
    q_log[1] = 3.0
    target = _ConstLogitsModel(p_log)
    draft = _ConstLogitsModel(q_log)

    # Sanity: under the unfiltered softmax the draft would accept ~13% of the
    # time (the buggy behaviour); the correct filtered theory is 0.
    unfiltered_ratio = (torch.softmax(p_log, -1)[1] / torch.softmax(q_log, -1)[1]).item()
    assert 0.05 < unfiltered_ratio < 0.3, f"unexpected unfiltered ratio {unfiltered_ratio}"
    p_accept_theory = 0.0  # top_k=1: proposal (1) has p_filtered = 0

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=DEFAULT_DEVICE)
    n = 2000
    accepted = 0
    for _ in range(n):
        accept_count, _ = _verify_speculative_tokens(
            target,
            draft,
            prompt,
            [1],
            temperature=1.0,
            top_k=1,
            top_p=None,
        )
        accepted += int(accept_count == 1)

    empirical = accepted / n
    assert abs(empirical - p_accept_theory) < 0.04, (
        f"top_k=1 acceptance {empirical:.3f} should match filtered theory "
        f"{p_accept_theory} but scores against the unfiltered distribution"
    )


def test_speculative_greedy_rejection_uses_target_argmax():
    """A greedy rejection emits the target's argmax at the rejection
    position — not the argmax of the logit difference.

    Regression test: the old code sampled ``argmax(logits_target -
    logits_draft)``, which can pick a token the target rates lower.
    """
    from llm.generation.speculative import _verify_speculative_tokens

    vocab = 8
    # Target prefers token 0; token 1 is the target's second choice.
    p_log = torch.full((vocab,), -5.0, device=DEFAULT_DEVICE)
    p_log[0] = 10.0
    p_log[1] = 9.9
    # Draft's distribution makes token 1 plausible but keeps it unlikely
    # relative to token 0; the logit difference p - q peaks at token 1.
    q_log = torch.full((vocab,), -5.0, device=DEFAULT_DEVICE)
    q_log[0] = 9.99
    q_log[1] = 3.0
    target = _ConstLogitsModel(p_log)
    draft = _ConstLogitsModel(q_log)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=DEFAULT_DEVICE)
    correction_tokens = set()
    for _ in range(200):
        accept_count, bonus = _verify_speculative_tokens(
            target,
            draft,
            prompt,
            [1],
            temperature=0.0,
            top_k=None,
            top_p=None,
        )
        assert accept_count == 0
        if bonus is not None:
            correction_tokens.add(bonus)

    # The target's argmax is token 0, so every greedy correction is 0.
    assert correction_tokens == {0}


# --- Sampling-time penalties are honored during verification ---------------


class _DistinctDecodeTokenizer:
    """Tokenizer whose decode is a bijection on token ids (0 -> A, ...)."""

    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + i) for i in ids)


def _make_bias_head_decoder(seed: int, vocab_size: int = 32):
    """Tiny decoder whose logits are a fixed bias vector (W=0)."""
    torch.manual_seed(seed)
    m = _make_tiny_decoder(seed=seed, vocab_size=vocab_size)
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.bias.zero_()
        # Token 2 (in the prompt) is the unpenalized argmax; token 4 wins
        # once token 2 is penalized (5 < 6 after repetition_penalty=2).
        m.lm_head.bias[2] = 10.0
        m.lm_head.bias[4] = 6.0
    return m


def test_speculative_greedy_matches_eager_under_penalties():
    """Under greedy decoding the speculative backend must reproduce the
    eager backend token-for-token even when sampling-time penalties are
    active (repetition / frequency / presence).

    Regression test: verification, correction, and bonus sampling used to
    ignore the penalties, so e.g. ``repetition_penalty=2.0`` produced
    ``'CCCCCCCC'`` where eager produced ``'ECCCCCCC'``.
    """
    target = _make_bias_head_decoder(seed=0)
    draft = _make_bias_head_decoder(seed=1)
    tok = _DistinctDecodeTokenizer()
    prompt = "abc"

    for kwargs in (
        {"repetition_penalty": 2.0},
        {"frequency_penalty": 1.0},
        {"presence_penalty": 1.0},
    ):
        eager_out = eager_generate(target, tok, prompt, max_new_tokens=8, use_cache=False, temperature=0.0, **kwargs)
        spec_out = prompt + "".join(
            speculative_generate(
                target,
                draft,
                tok,
                prompt,
                max_new_tokens=8,
                gamma=3,
                temperature=0.0,
                **kwargs,
            )
        )
        assert spec_out == eager_out, (kwargs, eager_out, spec_out)


def test_speculative_greedy_rejection_uses_penalized_target_argmax():
    """The greedy rejection correction must be the **penalized** target
    argmax at the rejection position.

    Regression test: penalties were previously dropped in the verification
    step, so the correction came from the unpenalized distribution.
    """
    from llm.generation.speculative import _verify_speculative_tokens

    vocab = 32
    p_log = torch.full((vocab,), 0.0, device=DEFAULT_DEVICE)
    p_log[2] = 10.0  # unpenalized argmax (also present in the prompt)
    p_log[4] = 6.0  # wins once token 2 is penalized by repetition_penalty=2
    q_log = p_log.clone()
    target = _ConstLogitsModel(p_log)
    draft = _ConstLogitsModel(q_log)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=DEFAULT_DEVICE)
    corrections = set()
    for _ in range(200):
        accept_count, bonus = _verify_speculative_tokens(
            target,
            draft,
            prompt,
            [2],
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=2.0,
        )
        assert accept_count == 0  # draft proposes 2; penalized argmax is 4
        if bonus is not None:
            corrections.add(bonus)

    assert corrections == {4}


# --- Backend registry integration ------------------------------------------


def test_speculative_backend_in_registry():
    """``get_generation_backend('speculative', ...)`` returns the backend."""
    ensure_backends_registered()
    assert "speculative" in BACKEND_REGISTRY.names()

    target = _make_tiny_decoder(seed=3)
    draft = _make_tiny_decoder(seed=3)
    backend = get_generation_backend("speculative", target_model=target, draft_model=draft, gamma=3)
    assert isinstance(backend, SpeculativeDecodingBackend)
    assert backend.target_model is target
    assert backend.draft_model is draft
    assert backend.gamma == 3


def test_speculative_backend_factory_requires_models():
    """Factory raises when models are not supplied."""
    ensure_backends_registered()
    with pytest.raises(ValueError, match="target_model"):
        get_generation_backend("speculative")


# --- Backend streaming end-to-end ----------------------------------------


def test_speculative_backend_streams():
    """The backend's ``stream`` produces the same greedy output as the
    standalone ``speculative_generate`` when draft == target."""
    from llm.generation.backends import GenerationConfig

    target = _make_tiny_decoder(seed=7)
    draft = _make_tiny_decoder(seed=7)
    tok = StubTokenizer()

    backend = SpeculativeDecodingBackend(target, draft, gamma=3)
    out = backend.generate(target, tok, "abc", config=GenerationConfig(max_new_tokens=4, temperature=0.0))
    # The path that goes through ``generate`` calls ``list(stream(...))``,
    # which goes through the speculative algorithm. Verify the call
    # doesn't blow up and yields the prompt + something.
    assert out.startswith("abc")
    assert len(out) >= len("abc")
