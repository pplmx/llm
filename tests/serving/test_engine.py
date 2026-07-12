from unittest.mock import patch

import pytest

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


def test_engine_prefill_populates_sequence_and_allocates_slot(tiny_model, mock_tokenizer):
    """Requirement: first step tokenizes prompt, runs prefill, and assigns a KV slot."""
    tiny_model.to("cpu")
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device="cpu",
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


def test_engine_decode_step_appends_generated_token(tiny_model, mock_tokenizer):
    """Requirement: second step appends one decode token while keeping the same slot."""
    tiny_model.to("cpu")
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device="cpu",
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


def test_engine_prefix_cache_reuses_kv_on_matching_prompt(tiny_model, mock_tokenizer):
    """Requirement: identical prompts reuse cached KV via _copy_kv_between_slots."""
    tiny_model.to("cpu")
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device="cpu",
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


def test_engine_paged_attention_sidecar_uses_configured_pool(tiny_model, mock_tokenizer):
    """Requirement: use_paged_attention attaches a PagedKVCache sidecar with configured size."""
    tiny_model.to("cpu")
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device="cpu",
        use_paged_attention=True,
        max_blocks=64,
        block_size=8,
        enable_prefix_cache=False,
    )
    assert engine.paged_kv_cache.num_blocks == 64
    assert engine.paged_kv_cache.block_size == 8
    assert engine.prefix_cache is None


def test_from_serving_config_wires_flags(tiny_model, mock_tokenizer):
    """Requirement: from_serving_config maps ServingConfig fields onto engine state."""
    from llm.serving.config import ServingConfig

    tiny_model.to("cpu")
    config = ServingConfig(
        max_concurrent_requests=3,
        max_seq_len=64,
        enable_prefix_cache=True,
        max_prefixes=5,
        use_paged_attention=False,
        max_blocks=32,
        block_size=8,
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


def test_from_serving_config_raises_on_paged_attention():
    """Paged Attention is sidecar-only; from_serving_config must refuse.

    Users who flip ``use_paged_attention=True`` get no benefit today and may
    not realize it. ``from_serving_config`` raises ``NotImplementedError`` so
    the failure surfaces at startup rather than at the first forward pass.
    Direct construction (e.g., from tests) still works.

    The check fires before any model/tokenizer access, so we pass sentinels.
    """
    from llm.serving.config import ServingConfig

    config = ServingConfig(
        use_paged_attention=True,
        max_blocks=32,
        block_size=8,
    )

    with pytest.raises(NotImplementedError, match="Paged Attention is not yet supported"):
        ContinuousBatchingEngine.from_serving_config(
            config,
            model=None,  # sentinel — check fires before engine is built
            tokenizer=None,
        )


# --- step() return contract + observer hook (T2 #22) ------------------------


def test_step_returns_stepstats_with_fill_ratio_fields(tiny_model, mock_tokenizer):
    """step() returns a StepStats dataclass with scheduled + total_active_slots."""
    from llm.serving.batch_engine import StepStats

    tiny_model.to("cpu")
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=4,
        device="cpu",
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


def test_step_observer_invoked_with_stepstats(tiny_model, mock_tokenizer):
    """set_step_observer receives the StepStats for each call to step()."""
    tiny_model.to("cpu")
    tiny_model.eval()

    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=mock_tokenizer,
        max_batch_size=2,
        device="cpu",
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
