"""Tests for the eager generation path (``llm.generation.eager``).

These tests close coverage gaps in:

- ``_mask_pad_logits``: ``pad_token_id is None`` early-return and
  out-of-bounds-skip branches.
- ``stream_generate``: prompt truncation when the prefill exceeds
  ``max_seq_len``, penalty / logit-bias application branches, and the
  stop-sequence buffer logic (suffix match, safe-prefix flush,
  final flush on loop exit).
- ``generate``: the non-streaming convenience wrapper around
  ``stream_generate`` (with and without stop sequences, with and
  without KV cache).
- ``batch_generate``: empty-prompt short-circuit, penalty / logit-bias
  application, the ``truncate_len <= 0`` branch, and stop-sequence
  truncation during the decode loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from llm.generation.eager import (
    _mask_pad_logits,
    _normalize_stop,
    batch_generate,
    generate,
    stream_generate,
)

# ---------------------------------------------------------------------------
# Test tokenizers
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """Character-level tokenizer mapping token id ``i`` to ``chr(ord('a') + i % 26)``.

    ``encode`` returns a fixed list of ``prompt_ids`` regardless of input,
    which is enough to control the prefill length in truncation tests.
    """

    pad_token_id: int = 0

    def __init__(self, prompt_ids: list[int] | None = None) -> None:
        self._prompt_ids = prompt_ids or [1]

    def encode(self, text: str) -> list[int]:
        return list(self._prompt_ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


class _NoPadTokenizer:
    """Tokenizer without a ``pad_token_id`` attribute.

    Used to exercise the ``getattr(tokenizer, "pad_token_id", None) -> None``
    path in ``_mask_pad_logits``.
    """

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


class _OutOfBoundsPadTokenizer:
    """Tokenizer whose ``pad_token_id`` exceeds the model vocabulary.

    Forces ``_mask_pad_logits`` to skip masking (the ``0 <= id < vocab``
    guard is False).
    """

    pad_token_id: int = 999

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


def _make_stop_tokenizer(sequence: list[str]):
    """Return a tokenizer whose ``decode`` emits one character from *sequence*
    per call, cycling through the list.

    The ``encode`` always returns ``[1]`` (single prompt token).
    """

    class _StopTok(_CharTokenizer):
        def __init__(self):
            super().__init__([1])
            self._idx = 0

        def decode(self, ids: list[int]) -> str:
            ch = sequence[self._idx % len(sequence)]
            self._idx += 1
            return ch

    return _StopTok


# ---------------------------------------------------------------------------
# _mask_pad_logits
# ---------------------------------------------------------------------------


def test_mask_pad_logits_none_skips():
    """``pad_token_id is None`` → early return, logits unchanged (line 20)."""
    logits = torch.zeros(10)
    original = logits.clone()
    _mask_pad_logits(logits, None)
    assert torch.equal(logits, original)


def test_mask_pad_logits_out_of_bounds_skips():
    """``pad_token_id >= vocab_size`` → no masking, logits unchanged
    (covers the ``22->exit`` branch at line 22)."""
    logits = torch.zeros(10)
    original = logits.clone()
    _mask_pad_logits(logits, pad_token_id=999)
    assert torch.equal(logits, original)


def test_mask_pad_logits_negative_pad_skips():
    """Negative ``pad_token_id`` → no masking (also covers ``22->exit``)."""
    logits = torch.zeros(10)
    original = logits.clone()
    _mask_pad_logits(logits, pad_token_id=-1)
    assert torch.equal(logits, original)


def test_mask_pad_logits_1d_masks_pad():
    """1-D logits: the pad token logit is set to ``-inf``."""
    logits = torch.zeros(10)
    _mask_pad_logits(logits, pad_token_id=3)
    assert logits[3].item() == float("-inf")


def test_mask_pad_logits_2d_masks_pad():
    """2-D logits: the pad token column is set to ``-inf`` across the batch."""
    logits = torch.zeros(4, 10)
    _mask_pad_logits(logits, pad_token_id=5)
    assert torch.all(logits[:, 5] == float("-inf"))


# ---------------------------------------------------------------------------
# _normalize_stop
# ---------------------------------------------------------------------------


def test_normalize_stop_none():
    assert _normalize_stop(None) is None


def test_normalize_stop_empty_string():
    assert _normalize_stop("") is None


def test_normalize_stop_single_string():
    assert _normalize_stop("END") == ["END"]


def test_normalize_stop_list_preserves_order():
    assert _normalize_stop(["A", "B"]) == ["A", "B"]


def test_normalize_stop_empty_list():
    assert _normalize_stop([]) is None


def test_normalize_stop_filters_empty_strings():
    assert _normalize_stop(["", "OK"]) == ["OK"]


def test_normalize_stop_all_empty():
    assert _normalize_stop(["", ""]) is None


# ---------------------------------------------------------------------------
# stream_generate — prompt truncation
# ---------------------------------------------------------------------------


def _make_recording_model(device):
    """Tiny DecoderModel that records every ``input_ids`` seen by ``forward``.

    Uses the same dimensions as the ``tiny_model`` fixture (vocab=100,
    hidden=16, layers=1, heads=2, max_seq_len=16) but subclasses
    ``DecoderModel`` so we can spy on the prefill input without mocking.
    """
    from llm.models.decoder import DecoderModel

    class _RecordingModel(DecoderModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seen_inputs: list[torch.Tensor] = []

        def forward(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self.seen_inputs.append(input_ids.detach().clone())
            return super().forward(*args, **kwargs)

    model = _RecordingModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        device=device,
    )
    model.eval()
    return model


def test_stream_generate_truncates_long_prompt(device):
    """When ``len(prompt) + max_new_tokens > max_seq_len``, the prompt is
    truncated to the last ``max_seq_len - max_new_tokens`` tokens before
    the prefill forward pass (covers lines 85-88).

    ``max_seq_len=16``; with ``max_new_tokens=4`` and a 20-token prompt,
    the prompt should be truncated to 12 tokens.
    """
    prompt_ids = list(range(1, 21))  # 20 tokens
    tokenizer = _CharTokenizer(prompt_ids)
    recording_model = _make_recording_model(device)

    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        list(
            stream_generate(
                model=recording_model,
                tokenizer=tokenizer,
                prompt="long",
                max_new_tokens=4,
                temperature=0.0,
            )
        )

    # First forward is the prefill — it should have exactly 12 tokens.
    assert len(recording_model.seen_inputs) >= 1
    assert recording_model.seen_inputs[0].shape[-1] == 12


def test_stream_generate_no_truncation_when_prompt_fits(device):
    """When ``len(prompt) + max_new_tokens <= max_seq_len``, no truncation.

    ``max_seq_len=16``, ``max_new_tokens=4``, prompt of 3 tokens → 7 <= 16.
    """
    tokenizer = _CharTokenizer([1, 2, 3])
    recording_model = _make_recording_model(device)

    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        list(
            stream_generate(
                model=recording_model,
                tokenizer=tokenizer,
                prompt="abc",
                max_new_tokens=4,
                temperature=0.0,
            )
        )

    assert len(recording_model.seen_inputs) >= 1
    assert recording_model.seen_inputs[0].shape[-1] == 3


# ---------------------------------------------------------------------------
# stream_generate — stop sequences
# ---------------------------------------------------------------------------


def test_stream_generate_stop_matches_as_suffix(tiny_model):
    """A stop sequence that becomes a suffix halts generation and excludes
    the stop string from the output (covers lines 137-145).

    Tokens emit 'a','b','c','d','E','N','D' — stop "END" matches at the
    buffer "dEND" boundary; the 'd' prefix is yielded and generation
    stops.
    """
    tok = _make_stop_tokenizer(["a", "b", "c", "d", "E", "N", "D", "x", "y"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=12,
            temperature=0.0,
            stop="END",
        )
    )
    # 'a','b','c' yielded as safe prefixes; 'd' yielded when 'END' matches.
    assert "".join(chunks) == "abcd"


def test_stream_generate_stop_first_token_is_stop(tiny_model):
    """Stop sequence equal to the very first token yields nothing and
    returns immediately (covers the ``prefix == ""`` branch skip at
    line 143-144)."""
    tok = _make_stop_tokenizer(["X", "a", "b"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=4,
            temperature=0.0,
            stop="X",
        )
    )
    assert chunks == []


def test_stream_generate_stop_never_matches_flushes_buffer(tiny_model):
    """When no stop sequence ever matches, all generated characters are
    yielded and the buffer is flushed at loop end (covers lines 171-172).

    Tokens emit 'a','b','c','d' — stop "ZZZ" never matches; buffer stays
    small ( <= max_stop_len=3), so nothing is yielded during the loop.
    The final flush at line 171-172 emits "abcd".
    """
    tok = _make_stop_tokenizer(["a", "b", "c", "d"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=4,
            temperature=0.0,
            stop="ZZZ",
        )
    )
    assert "".join(chunks) == "abcd"


def test_stream_generate_stop_buffer_exceeds_max_stop_len(tiny_model):
    """When the buffer grows beyond ``max_stop_len`` without a match,
    the safe prefix is yielded and the tail is retained (covers
    lines 148-151).

    Tokens emit 'a','b','c','d','e','f','g','h' with stop "ZZ"
    (max_stop_len=2).  The buffer exceeds 2 chars repeatedly, yielding
    safe prefixes; at loop end the remaining tail is flushed.
    """
    tok = _make_stop_tokenizer(["a", "b", "c", "d", "e", "f", "g", "h"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=8,
            temperature=0.0,
            stop="ZZ",
        )
    )
    # Every character ends up in the output — some as safe-prefix yields,
    # the rest as the final flush.
    assert "".join(chunks) == "abcdefgh"


def test_stream_generate_stop_list_first_match_wins(tiny_model):
    """Multiple stops — the first to match as a suffix wins.

    Tokens emit 'x','y','S','T','O','P' with stop=["S","STOP"].
    max_stop_len=4 (len of "STOP"), so the buffer grows to "xy"
    before "S" is appended.  "xyS" ends with "S" (the first stop in
    the list) — the prefix "xy" is yielded and generation halts.
    """
    tok = _make_stop_tokenizer(["x", "y", "S", "T", "O", "P", "a"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=10,
            temperature=0.0,
            stop=["S", "STOP"],
        )
    )
    assert "".join(chunks) == "xy"


def test_stream_generate_no_stop_yields_every_token(tiny_model):
    """With ``stop=None``, every token is yielded directly (covers the
    ``else`` branch at line 152-153)."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=3,
            temperature=0.0,
        )
    )
    assert "".join(chunks) == "abc"


# ---------------------------------------------------------------------------
# stream_generate — penalty / logit-bias branches (lines 119-126)
# ---------------------------------------------------------------------------


def test_stream_generate_repetition_penalty_branch(tiny_model):
    """``repetition_penalty != 1.0`` exercises the penalty branch (line 120)."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    with patch("llm.generation.eager.apply_repetition_penalty", wraps=lambda logits, _t, _p: logits) as spy:
        list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok(),
                prompt="p",
                max_new_tokens=3,
                temperature=0.0,
                repetition_penalty=1.5,
            )
        )
    assert spy.call_count >= 1


def test_stream_generate_frequency_penalty_branch(tiny_model):
    """``frequency_penalty != 0.0`` exercises the branch (line 122)."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    with patch("llm.generation.eager.apply_frequency_penalty", wraps=lambda logits, _t, _p: logits) as spy:
        list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok(),
                prompt="p",
                max_new_tokens=3,
                temperature=0.0,
                frequency_penalty=0.5,
            )
        )
    assert spy.call_count >= 1


def test_stream_generate_presence_penalty_branch(tiny_model):
    """``presence_penalty != 0.0`` exercises the branch (line 124)."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    with patch("llm.generation.eager.apply_presence_penalty", wraps=lambda logits, _t, _p: logits) as spy:
        list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok(),
                prompt="p",
                max_new_tokens=3,
                temperature=0.0,
                presence_penalty=0.5,
            )
        )
    assert spy.call_count >= 1


def test_stream_generate_logit_bias_branch(tiny_model):
    """``logit_bias`` truthy exercises the branch (line 126)."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    with patch("llm.generation.eager.apply_logit_bias", wraps=lambda logits, _b: logits) as spy:
        list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok(),
                prompt="p",
                max_new_tokens=3,
                temperature=0.0,
                logit_bias={1: -10.0},
            )
        )
    assert spy.call_count >= 1


# ---------------------------------------------------------------------------
# stream_generate — no-cache path
# ---------------------------------------------------------------------------


def test_stream_generate_use_cache_false(tiny_model):
    """``use_cache=False`` exercises the no-cache forward-pass branch."""
    tok = _make_stop_tokenizer(["a", "b"])
    chunks = list(
        stream_generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=2,
            temperature=0.0,
            use_cache=False,
        )
    )
    assert "".join(chunks) == "ab"


# ---------------------------------------------------------------------------
# stream_generate — pad_token_id edge cases
# ---------------------------------------------------------------------------


def test_stream_generate_pad_token_id_none(tiny_model):
    """Tokenizer without ``pad_token_id`` triggers the ``getattr -> None``
    path in ``_mask_pad_logits`` (line 20)."""
    tok = _NoPadTokenizer()
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        chunks = list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok,
                prompt="p",
                max_new_tokens=2,
                temperature=0.0,
            )
        )
    assert len(chunks) == 2


def test_stream_generate_pad_token_id_out_of_bounds(tiny_model):
    """Tokenizer with ``pad_token_id >= vocab_size`` skips masking
    (covers ``22->exit``)."""
    tok = _OutOfBoundsPadTokenizer()
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        chunks = list(
            stream_generate(
                model=tiny_model,
                tokenizer=tok,
                prompt="p",
                max_new_tokens=2,
                temperature=0.0,
            )
        )
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# generate (non-streaming wrapper)
# ---------------------------------------------------------------------------


def test_generate_returns_prompt_plus_generated(tiny_model):
    """``generate`` concatenates the prompt with the streamed tokens."""
    tok = _make_stop_tokenizer(["a", "b", "c"])
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        result = generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="START",
            max_new_tokens=3,
            temperature=0.0,
        )
    assert result == "STARTabc"


def test_generate_with_stop_excludes_stop_string(tiny_model):
    """``generate`` respects stop sequences and excludes the stop string."""
    tok = _make_stop_tokenizer(["a", "b", "X", "a", "b"])
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        result = generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=5,
            temperature=0.0,
            stop="X",
        )
    # "ab" before "X", then "X" matches → stop, exclude "X".
    assert result == "pab"


def test_generate_with_use_cache_false(tiny_model):
    """``generate`` works with ``use_cache=False``."""
    tok = _make_stop_tokenizer(["a", "b"])
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        result = generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="x",
            max_new_tokens=2,
            temperature=0.0,
            use_cache=False,
        )
    assert result == "xab"


def test_generate_with_repetition_penalty(tiny_model):
    """``generate`` forwards ``repetition_penalty`` to ``stream_generate``."""
    tok = _make_stop_tokenizer(["a", "b"])
    with (
        patch("llm.generation.eager.apply_repetition_penalty", wraps=lambda logits, _t, _p: logits),
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        result = generate(
            model=tiny_model,
            tokenizer=tok(),
            prompt="p",
            max_new_tokens=2,
            temperature=0.0,
            repetition_penalty=2.0,
        )
    assert result == "pab"


# ---------------------------------------------------------------------------
# batch_generate — empty prompts and truncation branch
# ---------------------------------------------------------------------------


def test_batch_generate_empty_prompts_returns_empty():
    """``batch_generate([])`` short-circuits to ``[]`` (line 258)."""

    class _StubTok:
        pad_token_id = 0

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "x"

    result = batch_generate(
        model=MagicMock(),
        tokenizer=_StubTok(),
        prompts=[],
        max_new_tokens=5,
    )
    assert result == []


def test_batch_generate_truncate_len_zero_skips_truncation_branch():
    """When ``max_new_tokens >= max_seq_len``, ``truncate_len <= 0`` and the
    truncation block is skipped entirely (covers the ``275->280``
    branch).

    We use a mock model because ``max_new_tokens >= max_seq_len`` means
    even a 1-token prompt + ``max_new_tokens`` decode steps would exceed
    the positional-encoding limit.  The mock lets us verify the code path
    without the model actually running.

    ``max_seq_len=16``, ``max_new_tokens=16`` → ``truncate_len = 0``.
    The prefill forward should receive the full untruncated 5-token prompt.
    """
    mock_model = MagicMock()
    mock_model.max_seq_len = 16
    mock_logits = torch.randn(1, 1, 100)
    mock_model.return_value = (mock_logits, [])
    # ``next(model.parameters())`` must return a real tensor so
    # ``.device`` works for tensor creation.  ``next()`` requires an
    # iterator, so we wrap the list.
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    tok = _CharTokenizer([1, 2, 3, 4, 5])  # 5-token prompt

    with (
        patch("llm.generation.eager.create_decoder_kv_caches", return_value=[]),
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        result = batch_generate(
            model=mock_model,
            tokenizer=tok,
            prompts=["p"],
            max_new_tokens=16,
            temperature=0.0,
        )

    # The prefill forward (call 0) should have received the full 5-token prompt
    # — no truncation happened because truncate_len <= 0.
    prefill_input = mock_model.call_args_list[0].args[0]
    assert prefill_input.shape[-1] == 5
    assert len(result) == 1


# ---------------------------------------------------------------------------
# batch_generate — penalty / logit-bias branches (lines 308-315)
# ---------------------------------------------------------------------------


def test_batch_generate_repetition_penalty_branch(tiny_model):
    """``repetition_penalty != 1.0`` exercises the batch penalty branch
    (line 308)."""
    tok = _CharTokenizer([1])
    with (
        patch("llm.generation.eager.apply_repetition_penalty", wraps=lambda logits, _t, _p: logits) as spy,
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        batch_generate(
            model=tiny_model,
            tokenizer=tok,
            prompts=["p"],
            max_new_tokens=3,
            temperature=0.0,
            repetition_penalty=2.0,
        )
    assert spy.call_count >= 1


def test_batch_generate_frequency_penalty_branch(tiny_model):
    """``frequency_penalty != 0.0`` exercises the batch branch (line 310)."""
    tok = _CharTokenizer([1])
    with (
        patch("llm.generation.eager.apply_frequency_penalty", wraps=lambda logits, _t, _p: logits) as spy,
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        batch_generate(
            model=tiny_model,
            tokenizer=tok,
            prompts=["p"],
            max_new_tokens=3,
            temperature=0.0,
            frequency_penalty=0.5,
        )
    assert spy.call_count >= 1


def test_batch_generate_presence_penalty_branch(tiny_model):
    """``presence_penalty != 0.0`` exercises the batch branch (line 312)."""
    tok = _CharTokenizer([1])
    with (
        patch("llm.generation.eager.apply_presence_penalty", wraps=lambda logits, _t, _p: logits) as spy,
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        batch_generate(
            model=tiny_model,
            tokenizer=tok,
            prompts=["p"],
            max_new_tokens=3,
            temperature=0.0,
            presence_penalty=0.5,
        )
    assert spy.call_count >= 1


def test_batch_generate_logit_bias_branch(tiny_model):
    """``logit_bias`` truthy exercises the batch branch (line 314)."""
    tok = _CharTokenizer([1])
    with (
        patch("llm.generation.eager.apply_logit_bias", wraps=lambda logits, _b: logits) as spy,
        patch(
            "llm.generation.eager.sample_next_token",
            side_effect=lambda logits, **kw: 1,
        ),
    ):
        batch_generate(
            model=tiny_model,
            tokenizer=tok,
            prompts=["p"],
            max_new_tokens=3,
            temperature=0.0,
            logit_bias={1: -10.0},
        )
    assert spy.call_count >= 1


# ---------------------------------------------------------------------------
# batch_generate — stop sequences
# ---------------------------------------------------------------------------


def test_batch_generate_with_stop_excludes_stop_string(tiny_model):
    """``batch_generate`` truncates output at the stop suffix (OpenAI
    semantics).

    Uses a deterministic model (``temperature=0``) with a tokenizer whose
    ``decode`` maps token ids to chars. We patch ``sample_next_token`` to
    control output.
    """

    class _StopTok:
        pad_token_id: int = 0

        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, ids: list[int]) -> str:
            char_map = {1: "Q", 4: "D", 5: "E"}
            return "".join(char_map.get(i, "Z") for i in ids)

    call_count = [0]

    def fake_sample(logits, **kwargs):  # noqa: ARG001
        call_count[0] += 1
        if call_count[0] == 1:
            return 4  # 'D'
        return 5  # 'E'

    with patch("llm.generation.eager.sample_next_token", side_effect=fake_sample):
        result = batch_generate(
            model=tiny_model,
            tokenizer=_StopTok(),
            prompts=["q"],
            max_new_tokens=5,
            temperature=0.0,
            stop="DE",
        )

    # Stop "DE" matches as suffix of generated_part "DE" → truncated to "".
    assert result == ["Q"]


def test_batch_generate_stop_never_matches_runs_full(tiny_model):
    """When stop is never emitted, ``batch_generate`` runs all
    ``max_new_tokens``."""

    class _SimpleTok:
        pad_token_id: int = 0

        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, ids: list[int]) -> str:
            char_map = {1: "A", 4: "B"}
            return "".join(char_map.get(i, "Z") for i in ids)

    def fake_sample(logits, **kwargs):  # noqa: ARG001
        return 4  # Always 'B'

    with patch("llm.generation.eager.sample_next_token", side_effect=fake_sample):
        result = batch_generate(
            model=tiny_model,
            tokenizer=_SimpleTok(),
            prompts=["a"],
            max_new_tokens=3,
            temperature=0.0,
            stop="ZZZ",
        )

    assert result == ["ABBB"]


def test_batch_generate_stop_list_multiple(tiny_model):
    """``batch_generate`` with a list of stops — the first suffix match
    wins."""

    class _MultiStopTok:
        pad_token_id: int = 0

        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, ids: list[int]) -> str:
            char_map = {1: "P", 4: "X", 5: "Y"}
            return "".join(char_map.get(i, "Z") for i in ids)

    call_count = [0]

    def fake_sample(logits, **kwargs):  # noqa: ARG001
        call_count[0] += 1
        if call_count[0] == 1:
            return 4  # 'X'
        return 5  # 'Y'

    with patch("llm.generation.eager.sample_next_token", side_effect=fake_sample):
        result = batch_generate(
            model=tiny_model,
            tokenizer=_MultiStopTok(),
            prompts=["q"],
            max_new_tokens=5,
            temperature=0.0,
            stop=["X", "XY"],
        )

    # First generated token 'X' matches stop "X" → generated_part = "X"
    # → truncated to "".  Result: "P" + "" = "P".
    assert result == ["P"]


def test_batch_generate_no_stop_returns_full(tiny_model):
    """``batch_generate`` without stop sequences returns prompt + generated
    tokens for every prompt."""
    tok = _CharTokenizer([1])
    with patch(
        "llm.generation.eager.sample_next_token",
        side_effect=lambda logits, **kw: 1,
    ):
        result = batch_generate(
            model=tiny_model,
            tokenizer=tok,
            prompts=["a", "b"],
            max_new_tokens=3,
            temperature=0.0,
        )

    assert len(result) == 2
    # Each prompt encodes to [1] → 'b' (chr(ord('a')+1)); 3 generated
    # tokens of id 1 → 'bbb'.
    assert result == ["bbbb", "bbbb"]
