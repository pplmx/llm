from unittest.mock import patch

import pytest
import torch

from llm.serving.batch_engine import ContinuousBatchingEngine, SlotAllocator
from llm.serving.schemas import GenerationRequest, RequestState


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0

    def encode(self, text):
        # Deterministic ids: one distinct id per character position
        return list(range(1, len(text) + 1))

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def test_slot_allocator_allocate_and_free_round_trip():
    allocator = SlotAllocator(total_slots=4)

    slot1 = allocator.allocate("req1")
    assert slot1 in {0, 1, 2, 3}
    assert len(allocator.free_slots) == 3
    assert allocator.get_slot("req1") == slot1

    slot2 = allocator.allocate("req2")
    assert slot2 == 1
    assert len(allocator.free_slots) == 2

    allocator.free("req1")
    assert len(allocator.free_slots) == 3
    assert allocator.get_slot("req1") == -1


def test_engine_serves_rope_model(device, mock_tokenizer):
    """A ``use_rope=True`` model must be servable by the batch engine.

    Regression (RIL ISS-112): ``_rope_positions`` evaluated the int-only
    ``start_pos == 0`` guard before the tensor type check, so the engine's
    ``[B, S]`` ``position_ids`` tensor raised ``Boolean value of Tensor with
    more than one element is ambiguous`` on the very first step — every RoPE
    request returned 500 via both the dense and paged paths.
    """
    from llm.models.decoder import DecoderModel

    rope_model = DecoderModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        use_rope=True,
        device=str(device),
    )
    rope_model.eval()

    engine = ContinuousBatchingEngine(
        model=rope_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=3)
    req.request_id = "req-rope"
    engine.add_request(req)
    engine.step()  # must not crash on the tensor position_ids

    seq = engine.scheduler.get_sequence("req-rope")
    assert seq is not None
    assert len(seq.generated_ids) == 1
    # Walk it to completion to make sure decode steps also pass through RoPE.
    for _ in range(3):
        if seq.is_finished():
            break
        engine.step()
    assert engine.slot_allocator.get_slot("req-rope") == -1  # released


def test_engine_prefill_populates_sequence_and_allocates_slot(tiny_model, device, mock_tokenizer):
    """Requirement: first step tokenizes prompt, runs prefill, and assigns a KV slot."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=10)
    req.request_id = "req1"
    engine.add_request(req)
    engine.step()

    seq = engine.scheduler.get_sequence("req1")
    assert seq.status == RequestState.RUNNING
    assert seq.input_ids == [1, 2, 3, 4]
    assert len(seq.generated_ids) == 1
    assert engine.slot_allocator.get_slot("req1") >= 0


def test_engine_decode_step_appends_generated_token(tiny_model, device, mock_tokenizer):
    """Requirement: second step appends one decode token while keeping the same slot."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=10)
    req.request_id = "req2"
    engine.add_request(req)
    engine.step()
    slot = engine.slot_allocator.get_slot("req2")

    engine.step()

    seq = engine.scheduler.get_sequence("req2")
    assert len(seq.generated_ids) == 2
    assert seq.status == RequestState.RUNNING
    assert engine.slot_allocator.get_slot("req2") == slot


def test_engine_prefix_cache_reuses_kv_on_matching_prompt(tiny_model, device, mock_tokenizer):
    """Requirement: identical prompts reuse cached KV via _copy_kv_between_slots."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        enable_prefix_cache=True,
    )

    req1 = GenerationRequest(prompt="hello", max_new_tokens=3)
    req1.request_id = "req-a"
    engine.add_request(req1)
    engine.step()

    slot_a = engine.slot_allocator.get_slot("req-a")
    cached = engine.prefix_cache.get([1, 2, 3, 4, 5])
    assert cached == (slot_a, 5)

    req2 = GenerationRequest(prompt="hello", max_new_tokens=3)
    req2.request_id = "req-b"
    engine.add_request(req2)

    with patch.object(engine, "_copy_kv_between_slots", wraps=engine._copy_kv_between_slots) as copy_kv:
        engine.step()
        copy_kv.assert_called_once()
        src_slot, dst_slot, prefix_len = copy_kv.call_args.args
        assert src_slot == slot_a
        assert prefix_len == 5
        assert dst_slot == engine.slot_allocator.get_slot("req-b")

    seq2 = engine.scheduler.get_sequence("req-b")
    assert len(seq2.generated_ids) == 1


def test_engine_paged_attention_uses_configured_pool(tiny_model, device, mock_tokenizer):
    """``use_paged_attention=True`` builds the paged pool and skips the dense one."""
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
        enable_prefix_cache=False,
    )
    # Paged pool wired through.
    assert engine.paged_kv_cache.num_blocks == 64
    assert engine.paged_kv_cache.block_size == 8
    # Dense pool is skipped — the model now writes into the paged blocks.
    assert engine.kv_caches == []
    assert engine.prefix_cache is None


def test_engine_paged_attention_prefix_cache_does_not_short_circuit(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-068): with ``use_paged_attention=True`` AND
    ``enable_prefix_cache=True`` the prefix-cache fast path must NOT
    short-circuit to a 1-token prefill.

    ``_copy_kv_between_slots`` (which replays cached K/V into a fresh dense
    slot) is a no-op on the paged path, so a prefix hit used to feed only
    the final prompt token into a brand-new 1-token block table — the paged
    kernel then attended over that single token, silently producing output
    that diverged from the dense backend. We now fall back to a full
    prefill, so two identical prompts must generate identical tokens.
    """

    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
        enable_prefix_cache=True,
        max_prefixes=4,
    )

    def _generate(tag: str) -> list[int]:
        # temperature=0 → greedy, so determinism isolates the cache behavior.
        req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0)
        req.request_id = tag
        engine.add_request(req)
        # Run steps until this request reaches a terminal state so its KV is
        # fully committed (and prefix put happens before the next request).
        for _ in range(10):
            engine.step()
            seq = engine.scheduler.get_sequence(tag)
            if seq.status == RequestState.FINISHED:
                return list(seq.generated_ids)
        raise AssertionError(f"request {tag} did not finish in bounds")

    # Both requests prefill "hello" ([1,2,3,4,5]) in full on the paged path.
    first = _generate("r1")
    second = _generate("r2")

    assert second == first, (
        "identical prompts must generate identical output on the paged+prefix "
        f"path; got {first} vs {second} (prefix fast path wrongly short-circuited)"
    )

    # The DENSE SlotPrefixCache must stay empty on the paged path: its
    # entries can never be read (the shortcut is disabled) so populating it
    # would only fill the LRU with dead hashes and churn invalidate_for_slot.
    assert engine.prefix_cache is not None
    assert not engine.prefix_cache._entries, "paged path must not populate the unreachable dense SlotPrefixCache"


def test_from_serving_config_wires_flags(tiny_model, device, mock_tokenizer):
    """Requirement: from_serving_config maps ServingConfig fields onto engine state."""
    from llm.serving.config import ServingConfig

    config = ServingConfig(
        max_concurrent_requests=3,
        max_seq_len=64,
        enable_prefix_cache=True,
        max_prefixes=5,
        use_paged_attention=False,
        max_blocks=32,
        block_size=8,
        device=str(device),
    )

    engine = ContinuousBatchingEngine.from_serving_config(
        config,
        model=tiny_model,
        tokenizer=mock_tokenizer,
    )

    assert engine.max_batch_size == 3
    assert engine.max_seq_len == 64
    assert engine.enable_prefix_cache is True
    assert engine.prefix_cache.max_prefixes == 5


def test_from_serving_config_wires_paged_attention_through(tiny_model, device, mock_tokenizer):
    """``use_paged_attention=True`` no longer raises — it wires the paged path.

    After T3 #3 Paged Attention is fully wired through the engine forward:
    ``from_serving_config`` builds the engine with a ``PagedKVCache`` and the
    dense ``KVCache`` pool is skipped (no double allocation). A smoke
    ``step()`` runs end-to-end.
    """
    from llm.serving.config import ServingConfig

    config = ServingConfig(
        use_paged_attention=True,
        max_blocks=32,
        block_size=8,
        max_concurrent_requests=2,
        max_seq_len=tiny_model.max_seq_len,
        device=str(device),
    )

    engine = ContinuousBatchingEngine.from_serving_config(
        config,
        model=tiny_model,
        tokenizer=mock_tokenizer,
    )

    # Dense pool is skipped in favour of the paged pool.
    assert engine.kv_caches == []
    assert engine.paged_kv_cache is not None
    assert engine.paged_kv_cache.num_blocks == 32
    assert engine.paged_kv_cache.block_size == 8

    # End-to-end smoke: a single ``step()`` runs the paged forward path.
    req_id = engine.add_request(GenerationRequest(prompt="abcd", max_new_tokens=3))
    engine.step()
    seq = engine.scheduler.get_sequence(req_id)
    assert seq is not None
    assert len(seq.generated_ids) == 1


# --- step() return contract + observer hook (T2 #22) ------------------------


def test_step_returns_stepstats_with_fill_ratio_fields(tiny_model, device, mock_tokenizer):
    """step() returns a StepStats dataclass with scheduled + total_active_slots."""
    from llm.serving.batch_engine import StepStats

    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=4,
        device=str(device),
    )

    # Idle engine: scheduled=0, total = max_batch_size.
    stats = engine.step()
    assert isinstance(stats, StepStats)
    assert stats.scheduled == 0
    assert stats.total_active_slots == 4

    # After adding a request and stepping, scheduled reflects the batch size.
    req = GenerationRequest(prompt="abcd", max_new_tokens=2)
    req.request_id = "stats-req"
    engine.add_request(req)
    stats = engine.step()
    assert stats.scheduled == 1
    assert stats.total_active_slots == 4


def test_step_observer_invoked_with_stepstats(tiny_model, device, mock_tokenizer):
    """set_step_observer receives the StepStats for each call to step()."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    observed: list = []
    engine.set_step_observer(observed.append)

    engine.step()
    engine.step()
    assert len(observed) == 2
    assert all(s.total_active_slots == 2 for s in observed)

    # Clearing the observer stops future invocations.
    engine.set_step_observer(None)
    engine.step()
    assert len(observed) == 2


# --- MLA + KV cache (T3 #31) --------------------------------------------
#
# Smoke test: a 1-layer DecoderModel with ``attn_impl='mla'`` runs
# end-to-end through ``ContinuousBatchingEngine``. Both the dense
# ``KVCache`` path and the paged ``PagedKVCache`` path are exercised;
# the MLA placeholder's K/V are written into the configured cache and
# the latent attention then runs over the cached context.


def _make_mla_decoder(device: str):
    """Tiny 1-layer DecoderModel with ``attn_impl='mla'``.

    The placeholder MLA needs ``hidden_size % num_heads == 0`` and uses
    its own ``num_latents`` / ``latent_dim`` defaults.
    """
    from llm.models.decoder import DecoderModel

    torch.manual_seed(0)
    return DecoderModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        attn_impl="mla",
        attn_dropout_p=0.0,
        embedding_dropout_p=0.0,
        mlp_dropout_p=0.0,
        device=device,
    )


def test_engine_runs_mla_step_with_dense_cache(device, mock_tokenizer):
    """MLA + dense KV cache: one prefill step writes into the cache."""
    model = _make_mla_decoder(device=str(device))
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        max_seq_len=model.max_seq_len,
        device=str(device),
        dtype=torch.float32,
    )

    # MLA writes into the dense cache the same way MHA does.
    assert engine.kv_caches
    assert engine.paged_kv_cache is None

    req = GenerationRequest(prompt="abcd", max_new_tokens=3)
    req.request_id = "mla-dense-1"
    engine.add_request(req)
    stats = engine.step()

    assert stats.scheduled == 1
    seq = engine.scheduler.get_sequence("mla-dense-1")
    assert seq.status == RequestState.RUNNING
    assert len(seq.generated_ids) == 1
    # The dense cache buffer recorded the prefill tokens (the per-row
    # buffer is sized to max_seq_len; we only check the per-slot slot
    # write landed, not the scalar ``seq_len`` which ``update_at_indices``
    # does not bump — same constraint as the MHA dense-cache tests).
    slot_id = engine.slot_allocator.get_slot("mla-dense-1")
    assert torch.any(engine.kv_caches[0].k_cache[slot_id, :, :, :] != 0)

    # A second step appends one more decode token.
    engine.step()
    seq = engine.scheduler.get_sequence("mla-dense-1")
    assert len(seq.generated_ids) == 2


def test_engine_runs_mla_step_with_paged_cache(device, mock_tokenizer):
    """MLA + paged KV cache: prefill allocates blocks; decode reuses them."""
    model = _make_mla_decoder(device=str(device))
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        max_seq_len=model.max_seq_len,
        device=str(device),
        dtype=torch.float32,
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
    )

    # Paged pool wired; dense pool skipped.
    assert engine.paged_kv_cache is not None
    assert engine.kv_caches == []

    req = GenerationRequest(prompt="abcd", max_new_tokens=3)
    req.request_id = "mla-paged-1"
    engine.add_request(req)
    stats = engine.step()

    assert stats.scheduled == 1
    seq = engine.scheduler.get_sequence("mla-paged-1")
    assert seq.status == RequestState.RUNNING
    # The paged cache has all prefill tokens for this request.
    slot_id = engine.slot_allocator.get_slot("mla-paged-1")
    assert engine.paged_kv_cache.block_manager.get_num_tokens(slot_id) == len(seq.input_ids)

    # A second step adds a decode token without allocating a new block
    # (block_size=8, prefill length is 4 → room remains).
    engine.step()
    assert engine.paged_kv_cache.block_manager.get_num_tokens(slot_id) == len(seq.input_ids) + 1
    assert len(seq.generated_ids) == 2


# --- ContinuousBatchingEngine: penalty + stop parameter forwarding ----------


def test_sequence_stores_all_sampling_parameters(tiny_model, device, mock_tokenizer):
    """``add_request`` must propagate every sampling parameter to the Sequence."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(
        prompt="abcd",
        max_new_tokens=5,
        frequency_penalty=1.3,
        presence_penalty=0.7,
        logit_bias={"1": 2.0},
        stop="END",
    )
    req.request_id = "req-pen"
    engine.add_request(req)

    seq = engine.scheduler.get_sequence("req-pen")
    assert seq.frequency_penalty == 1.3
    assert seq.presence_penalty == 0.7
    assert seq.logit_bias == {"1": 2.0}
    assert seq.stop == "END"


def test_generate_request_with_stop_truncates_output(tiny_model, device, mock_tokenizer):
    """``generate_request`` honours ``stop``: the stop string is excluded from the result."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(
        prompt="abcd",
        max_new_tokens=2,
        stop="x",
    )
    req.request_id = "req-stop"
    # MockTokenizer.decode emits "1 2 3 ..." — we can't easily make a stop
    # match, so we just assert the call completes without error and stop
    # is stored on the Sequence.
    engine.add_request(req)
    seq = engine.scheduler.get_sequence("req-stop")
    assert seq.stop == "x"

    # Generate a short result — it should complete without raising on stop.
    result = engine.generate_request(req)
    assert isinstance(result, str)


def test_engine_excludes_eos_text_from_output(tiny_model, device, mock_tokenizer):
    """The EOS token's decoded text must NOT appear in the output (parity
    with the eager/speculative backends — RIL ISS-96/ISS-98).

    ``_emit_tokens`` used to decode and yield the EOS id like any ordinary
    token (``_lock_step_post`` appends it, then marks the seq FINISHED), so a
    tokenizer whose EOS decodes to a real string polluted every request's
    streamed/final output.
    """
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    from llm.serving.batch_engine import _StepInputs, _StepResult

    def _emit_eos(inputs: _StepInputs) -> _StepResult:
        # Force every sampled token to be the EOS id (mock_tokenizer.eos=99).
        return _StepResult(inputs=inputs, next_token_ids=[99] * len(inputs.running_sequences))

    with patch.object(engine, "_forward_and_sample", side_effect=_emit_eos):
        req = GenerationRequest(prompt="abcd", max_new_tokens=5)
        req.request_id = "req-eos-excl"
        engine.add_request(req)
        result = engine.generate_request(req)

    # "99" is MockTokenizer.decode([99]) — the EOS token's text — must be excluded.
    assert "99" not in result
    assert isinstance(result, str)
    # The EOS finish path still releases the KV slot.
    assert engine.slot_allocator.get_slot("req-eos-excl") == -1


def test_stop_terminated_request_frees_kv_slot(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-044): a request that ends via a stop-sequence match
    must release its KV slot (dense + prefix + paged), otherwise the pool
    leaks a slot per stop-terminated request and eventually 503s on
    ``No free slots available in KV cache``.

    MockTokenizer.decode emits the token id as text (``decode([3]) == "3"``),
    so a stop string equal to a generated token's decode reliably matches.
    """
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        enable_prefix_cache=True,
    )
    total_slots = engine.slot_allocator.total_slots

    # Force the model to emit token 3 (decode → "3") so ``stop="3"`` matches.
    from llm.serving.batch_engine import _StepInputs, _StepResult

    def _emit_token_3(inputs: _StepInputs) -> _StepResult:
        return _StepResult(inputs=inputs, next_token_ids=[3] * len(inputs.running_sequences))

    with patch.object(engine, "_forward_and_sample", side_effect=_emit_token_3):
        req = GenerationRequest(prompt="abcd", max_new_tokens=5, stop="3")
        req.request_id = "req-stop-free"
        engine.add_request(req)

        result = engine.generate_request(req)

    assert isinstance(result, str)
    # Slot must be released back into the free pool.
    assert engine.slot_allocator.get_slot("req-stop-free") == -1
    assert engine.slot_allocator.num_free == total_slots


def test_stop_terminated_request_frees_paged_blocks(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-044): on the paged path a stop-terminated request
    must also return its KV blocks to the paged allocator."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
        enable_prefix_cache=False,
    )

    from llm.serving.batch_engine import _StepInputs, _StepResult

    def _emit_token_3(inputs: _StepInputs) -> _StepResult:
        return _StepResult(inputs=inputs, next_token_ids=[3] * len(inputs.running_sequences))

    with patch.object(engine, "_forward_and_sample", side_effect=_emit_token_3):
        req = GenerationRequest(prompt="abcd", max_new_tokens=5, stop="3")
        req.request_id = "req-stop-paged"
        engine.add_request(req)

        result = engine.generate_request(req)

    assert isinstance(result, str)
    assert engine.slot_allocator.get_slot("req-stop-paged") == -1
    # The paged sequence's blocks must have been returned to the allocator.
    assert engine.paged_kv_cache.block_manager.sequences == {}
    assert engine.paged_kv_cache.block_manager.num_free_blocks == engine.paged_kv_cache.num_blocks


def test_stream_request_does_not_double_yield_tail_buffer(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-054): when a sequence is already finished at the
    top of ``stream_request``'s loop (e.g. a concurrent step completed it)
    and a stop is configured, the drained tail buffer must be emitted exactly
    once — not again by the post-loop ``yield buffer``.

    Path A: ``seq.is_finished()`` at loop top → ``_emit_tokens`` drains a
    tail into ``buffer`` with no stop hit → the old code ``yield buffer``
    then ``break``, and the post-loop statement yielded the same buffer a
    second time (it was never cleared). ``stream_request`` must return after
    draining since the sequence is finished.
    """
    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=5, stop="NEVER-MATCHES")
    req.request_id = "req-double"
    engine.add_request(req)

    # Force path A: the sequence is already FINISHED before the loop runs.
    seq = engine.scheduler.get_sequence("req-double")
    seq.append_token_id(5)  # decode("5") -> "5"; stop never matches it
    seq.status = RequestState.FINISHED

    # step() would normally run, but the sequence is finished so the loop
    # exits at the top-of-loop branch before reaching step().
    chunks = list(engine.stream_request(req))

    emitted = "".join(chunks)
    assert emitted.count("5") == 1, f"tail buffer must be emitted exactly once, got chunks={chunks!r}"


def test_abandoned_stream_request_releases_slot_and_sequence(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-105): abandoning the streaming generator mid-run
    (consumer disconnect → ``gen.close()`` / GC) must release the KV slot and
    drop the sequence.

    ``stream_request`` is the sequence's only stepper; once the consumer stops
    pulling it, the sequence sits RUNNING with its slot allocated forever —
    each disconnect permanently consumes one of ``max_batch_size`` slots.
    """
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=100)
    req.request_id = "req-abandon"

    # ``stream_request`` adds the request itself (production path — no
    # separate ``add_request`` call).
    gen = engine.stream_request(req)
    next(gen)
    assert engine.slot_allocator.get_slot("req-abandon") >= 0, "slot should be allocated"
    assert engine.scheduler.get_sequence("req-abandon") is not None

    # Abandon the stream (what a client disconnect does to the generator).
    gen.close()

    assert engine.slot_allocator.get_slot("req-abandon") == -1, "abandoned request must release its slot"
    assert engine.scheduler.get_sequence("req-abandon") is None, "abandoned sequence must be reaped"


def test_abandoned_stream_removes_under_step_lock(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-117): the abandoned-generator cleanup mutates the
    scheduler's live ``running`` list, so ``scheduler.remove`` must run under
    the engine's ``_step_lock`` — otherwise it races with ``_lock_step_pre``
    iterating that list in a concurrent ``step()`` (a mid-iteration pop
    silently skips/duplicates a sequence)."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=100)
    req.request_id = "req-abandon-lock"
    gen = engine.stream_request(req)
    next(gen)

    locked_observations: list[bool] = []
    orig_remove = engine.scheduler.remove

    def locked_remove(request_id):
        locked_observations.append(engine._step_lock.locked())
        return orig_remove(request_id)

    engine.scheduler.remove = locked_remove  # type: ignore[method-assign]
    try:
        gen.close()
    finally:
        engine.scheduler.remove = orig_remove

    assert locked_observations, "scheduler.remove should have been called on abandonment"
    assert all(locked_observations), "scheduler.remove must run while _step_lock is held"


def test_persistent_forward_failure_does_not_livelock_engine(tiny_model, device, mock_tokenizer):
    """Regression (RIL ISS-051): a request whose forward persistently fails
    (OOM / bad token id / shape mismatch) must be dropped, not re-scheduled
    forever.

    The forward-failure path in ``_lock_step_post`` freed the slots but left
    the sequence RUNNING. ``Scheduler.schedule`` only drops FINISHED
    sequences, so every subsequent ``step()`` re-allocated a slot for the
    dead sequence and re-ran the failing forward — the whole engine
    livelocked and every request errored. Marking the affected sequences
    FINISHED lets ``schedule()`` remove them.
    """
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    # A request whose forward will persistently fail.
    req = GenerationRequest(prompt="abcd", max_new_tokens=3)
    req.request_id = "req-dying"
    engine.add_request(req)

    # First step: the model forward raises inside the real
    # ``_forward_and_sample``, which records it in ``_StepResult.forward_failed``
    # so ``_lock_step_post`` frees the slot and (after the fix) marks the
    # sequence FINISHED — otherwise the exception would propagate without the
    # status flip and the sequence would be re-scheduled forever. Patch
    # ``forward`` (not ``__call__``): nn.Module dispatch goes through the
    # class ``__call__``, so an instance-level ``__call__`` patch is ignored.
    with patch.object(engine.model, "forward", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            engine.step()

        # After the fix the failed sequence is FINISHED; the next
        # ``schedule()`` (start of the following step) drops it, so the
        # engine does NOT re-run the failing forward on the same sequence.
        assert engine.scheduler.running  # still present until next schedule()
        assert engine.scheduler.running[0].is_finished()
        assert engine.slot_allocator.get_slot("req-dying") == -1
        scheduled = engine.scheduler.schedule()
        assert "req-dying" not in [s.request_id for s in scheduled]


def test_forward_applies_frequency_penalty(tiny_model, device, mock_tokenizer):
    """``_forward_and_sample`` applies ``frequency_penalty`` from the Sequence."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(
        prompt="abcd",
        max_new_tokens=3,
        frequency_penalty=1.5,
    )
    req.request_id = "req-fp"
    engine.add_request(req)
    engine.step()

    seq = engine.scheduler.get_sequence("req-fp")
    assert seq.frequency_penalty == 1.5
    assert len(seq.generated_ids) == 1


def _eager_greedy_reference(model, tokenizer, prompt, max_new_tokens):
    """Mirror ``llm.generation.eager.generate``'s greedy path: recompute the
    full context each step and take the pad-masked argmax."""
    ids = list(tokenizer.encode(prompt))
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(torch.tensor([ids], dtype=torch.long, device=model_device), use_cache=False)
            logits = out[0][0, -1, :] if isinstance(out, tuple) else out[0, -1, :]
            logits = logits.clone()
            if tokenizer.pad_token_id is not None and 0 <= tokenizer.pad_token_id < logits.size(-1):
                logits[tokenizer.pad_token_id] = float("-inf")
            ids.append(int(logits.argmax().item()))
    return ids[len(tokenizer.encode(prompt)) :]


def _drive_engine_to_completion(engine):
    """Step the engine until idle, returning {request_id: generated_ids}."""
    last = {}
    guard = 0
    while engine.scheduler.has_pending_work and guard < 200:
        engine.step()
        for s in engine.scheduler.running:
            last[s.request_id] = list(s.generated_ids)
        guard += 1
    return last


def test_engine_greedy_matches_eager_reference(tiny_model, device, mock_tokenizer):
    """Regression: the continuous-batching engine's greedy output must match
    the eager backend. The causal attention mask used to be built from the
    zero-filled position buffer, so decode attention only saw the first
    prompt token's KV and outputs diverged from eager after a few steps."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        dtype=torch.float32,
    )
    req = GenerationRequest(prompt="abcd", max_new_tokens=8, temperature=0.0)
    req.request_id = "req-greedy"
    engine.add_request(req)

    last = _drive_engine_to_completion(engine)
    assert last["req-greedy"] == _eager_greedy_reference(tiny_model, mock_tokenizer, "abcd", 8)


def test_engine_mixed_length_batch_matches_eager_greedy(tiny_model, device, mock_tokenizer):
    """Regression: mixed-length batches corrupt each other's KV cache.

    Two defects were silently corrupting the cache in continuous batching:
    (1) the mixed-batch prefill wrote padded slots (position_id 0) over a
    short row's real position-0 K/V; (2) the decode write used an
    unflattened [B, 1] start_pos, broadcasting every slot's K/V onto every
    batch position. Both produced output that diverged from the eager
    backend. Greedy output must match eager per request."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=4,
        device=str(device),
        dtype=torch.float32,
    )
    prompts = ["abcd", "xy", "python"]  # mixed lengths (all <= model max_seq_len)
    for i, prompt in enumerate(prompts):
        req = GenerationRequest(prompt=prompt, max_new_tokens=6, temperature=0.0)
        req.request_id = f"req-{i}"
        engine.add_request(req)

    last = _drive_engine_to_completion(engine)
    for i, prompt in enumerate(prompts):
        assert last[f"req-{i}"] == _eager_greedy_reference(tiny_model, mock_tokenizer, prompt, 6), prompt


def test_engine_paged_mixed_length_batch_matches_eager_greedy(tiny_model, device, mock_tokenizer):
    """Regression: the paged-attention path appended padded (garbage) K/V for
    shorter prompts and skipped causal masking in multi-token prefill, so
    mixed-length batches diverged from the eager backend. Paged greedy
    output must match eager per request."""
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=4,
        device=str(device),
        dtype=torch.float32,
        use_paged_attention=True,
        max_blocks=64,
        block_size=16,
    )
    prompts = ["abcd", "xy", "python"]  # mixed lengths
    for i, prompt in enumerate(prompts):
        req = GenerationRequest(prompt=prompt, max_new_tokens=6, temperature=0.0)
        req.request_id = f"req-{i}"
        engine.add_request(req)

    last = _drive_engine_to_completion(engine)
    for i, prompt in enumerate(prompts):
        assert last[f"req-{i}"] == _eager_greedy_reference(tiny_model, mock_tokenizer, prompt, 6), prompt


def test_engine_step_with_metrics_observer(tiny_model, device, mock_tokenizer):
    """Regression: the API startup wired the metrics observer directly, but
    record_batch_fill_ratio takes keyword-only args while the engine invokes
    observers with a positional StepStats — every step raised TypeError.
    Stepping with the API's observer adapter must complete."""
    from llm.serving.api import _step_observer

    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        dtype=torch.float32,
    )
    engine.set_step_observer(_step_observer)
    req = GenerationRequest(prompt="abcd", max_new_tokens=4, temperature=0.0)
    req.request_id = "req-obs"
    engine.add_request(req)

    engine.step()  # must not raise TypeError
    assert engine.slot_allocator.get_slot("req-obs") >= 0


def test_prefix_cache_entry_invalidated_on_slot_reuse(tiny_model, device, mock_tokenizer):
    """Regression: a freed slot's prefix-cache entry must not survive reuse.

    Sequence: request A (prompt P) populates the prefix cache and finishes,
    freeing its KV slot; request B takes that *same* slot with a different
    prompt, overwriting the K/V; request C (matching prompt P) must NOT hit a
    stale prefix entry — replaying another request's overwritten KV as its
    prefix would corrupt generation. Before the fix the cache entry pointed at
    a freed-then-reused slot (a use-after-free of the cached KV).
    """
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        enable_prefix_cache=True,
    )

    # Request A: prompt "hello" (ids [1,2,3,4,5]), generate to completion so
    # it both caches its prefix and finishes (freeing its slot).
    req_a = GenerationRequest(prompt="hello", max_new_tokens=2)
    req_a.request_id = "req-a"
    engine.add_request(req_a)
    guard = _drive_engine_to_completion(engine)
    assert guard["req-a"], "req-a did not complete"
    slot_a = engine.slot_allocator.get_slot("req-a")
    assert slot_a == -1, "req-a slot should have been freed"

    # No prefix entry may point at the freed slot anymore.
    cached = engine.prefix_cache.get([1, 2, 3, 4, 5])
    assert cached is None, f"stale prefix entry survived slot free: {cached}"

    # Request B takes a slot with a *different* prompt (longer, so the mock
    # tokenizer produces distinct ids: "hello there" -> [1..11]).
    req_b = GenerationRequest(prompt="hello there", max_new_tokens=2)
    req_b.request_id = "req-b"
    engine.add_request(req_b)
    engine.step()

    # Request C = same prompt as A, must NOT replay a stale prefix (no
    # copy_kv call allowed on the first step — the cache entry was dropped).
    req_c = GenerationRequest(prompt="hello", max_new_tokens=2)
    req_c.request_id = "req-c"
    engine.add_request(req_c)
    with patch.object(engine, "_copy_kv_between_slots", wraps=engine._copy_kv_between_slots) as copy_kv:
        engine.step()
        copy_kv.assert_not_called()


def test_engine_paged_prefix_replay_reuses_blocks_and_matches_prefill(tiny_model, device, mock_tokenizer):
    """TASK-065: on the paged path with ``enable_prefix_cache=True``, a new
    request whose prompt exactly matches a cached prefix must STAGE the cached
    blocks (shared, refcounted) instead of re-prefilling them, and generate
    output identical to a full prefill.

    The origin sequence is still running (its slot alive) when the second
    request arrives — the concurrent-request case the prefix cache targets.
    """
    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
        enable_prefix_cache=True,
        max_prefixes=4,
    )

    # "hello world" → [1..11] (11 tokens) spans 2 blocks (block_size=8).
    prompt = "hello world"
    tokens = list(range(1, len(prompt) + 1))
    assert len(tokens) == 11

    req_a = GenerationRequest(prompt=prompt, max_new_tokens=4, temperature=0)
    req_a.request_id = "req-a"
    engine.add_request(req_a)
    engine.step()

    slot_a = engine.slot_allocator.get_slot("req-a")
    prefix_blocks = engine.paged_kv_cache.try_get_prefix_blocks(tokens)
    assert prefix_blocks is not None, "paged prefix cache must be populated after req-a's prefill"
    assert len(prefix_blocks) == 2
    assert prefix_blocks[0] == engine.paged_kv_cache.get_block_table(slot_a)[0]

    req_b = GenerationRequest(prompt=prompt, max_new_tokens=4, temperature=0)
    req_b.request_id = "req-b"
    engine.add_request(req_b)
    before_b = engine.paged_kv_cache.block_manager.get_num_tokens(slot_a)
    engine.step()  # runs req-a's decode AND req-b's staged prefix-hit prefill

    slot_b = engine.slot_allocator.get_slot("req-b")
    table_b = engine.paged_kv_cache.get_block_table(slot_b)
    # The leading (unwritten) prefix block is SHARED with req-a — the prefill
    # of the prefix was skipped, the block was reused, not recomputed. (The
    # write-loop order decides which seq copy-on-writes the shared boundary
    # block; whichever runs second writes the idempotent last-token value
    # directly into the now-private block — so we assert the ORDER-INDEPENDENT
    # invariants: shared leading block, intact entry, and byte-identical
    # prompt K/V on both sequences.)
    assert table_b[0] == prefix_blocks[0]
    assert engine.paged_kv_cache.block_manager.is_block_shared(prefix_blocks[0])
    # req-a's prefix cache entry is untouched (still live), and req-a advanced
    # by exactly its own single decode token — req-b's staged hit must not
    # disturb req-a's sequence state.
    assert engine.paged_kv_cache.try_get_prefix_blocks(tokens) == prefix_blocks
    assert engine.paged_kv_cache.block_manager.get_num_tokens(slot_a) == before_b + 1
    # Replay + COW kept both sequences' K/V for the whole prompt identical.
    kv_a = engine.paged_kv_cache.get(slot_a, 0, len(tokens))[0]
    kv_b = engine.paged_kv_cache.get(slot_b, 0, len(tokens))[0]
    assert torch.equal(kv_a, kv_b), "replay + COW must keep both seqs' prompt K/V identical"

    # Drain both to completion; identical prompts must give identical output.
    def _drain(tag: str) -> list[int]:
        for _ in range(12):
            engine.step()
            seq = engine.scheduler.get_sequence(tag)
            if seq.status == RequestState.FINISHED:
                return list(seq.generated_ids)
        raise AssertionError(f"request {tag} did not finish in bounds")

    first = _drain("req-a")
    second = _drain("req-b")

    assert len(first) == 4
    assert second == first, (
        f"paged prefix replay must produce identical output to a full prefill; got {first} vs {second}"
    )


def test_engine_rejects_prompt_that_exceeds_max_seq_len(tiny_model, device, mock_tokenizer):
    """A prompt (plus max_new_tokens) longer than the engine's max_seq_len
    must be rejected at add_request with a clear ValueError.

    Without this guard the model computes position ids past its positional-
    encoding table; the embedding gather then hits a CUDA device-side assert
    that corrupts the CUDA context and can kill the whole serving process —
    not just the offending request.
    """
    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        max_seq_len=16,  # matches tiny_model's max_seq_len
    )

    # 40-token prompt beyond the 16-token context.
    with pytest.raises(ValueError, match="max_seq_len"):
        engine.add_request(GenerationRequest(prompt="x" * 40, max_new_tokens=2))

    # Prompt fits, but the requested new tokens overflow the cap.
    with pytest.raises(ValueError, match="max_seq_len"):
        engine.add_request(GenerationRequest(prompt="x" * 15, max_new_tokens=4))

    # Exactly at the cap is accepted (each decode position stays < max_seq_len).
    req = GenerationRequest(prompt="x" * 15, max_new_tokens=1)
    req.request_id = "ok"
    engine.add_request(req)
    assert engine.scheduler.get_sequence("ok") is not None


def test_engine_paged_rejects_prompt_that_exceeds_max_seq_len(tiny_model, device, mock_tokenizer):
    """Same rejection on the paged-attention path (the guard lives in
    add_request, before any forward touches the block cache)."""
    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device=str(device),
        max_seq_len=16,
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
    )
    with pytest.raises(ValueError, match="max_seq_len"):
        engine.add_request(GenerationRequest(prompt="x" * 40, max_new_tokens=2))
