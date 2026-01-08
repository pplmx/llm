import pytest

from llm.serving.engine import ContinuousBatchingEngine, SlotAllocator
from llm.serving.schemas import GenerationRequest, RequestState


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0

    def encode(self, text):
        # Return dummy ids based on length of text
        return [1] * len(text)

    def decode(self, ids):
        return " ".join(map(str, ids))


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def test_slot_allocator():
    allocator = SlotAllocator(total_slots=4)
    assert len(allocator.free_slots) == 4

    slot1 = allocator.allocate("req1")
    assert slot1 in range(4)
    assert len(allocator.free_slots) == 3
    assert allocator.get_slot("req1") == slot1

    slot2 = allocator.allocate("req2")
    assert slot2 != slot1
    assert len(allocator.free_slots) == 2

    allocator.free("req1")
    assert len(allocator.free_slots) == 3
    assert allocator.get_slot("req1") == -1


def test_engine_initialization(mock_tokenizer):
    engine = ContinuousBatchingEngine(model_path="dummy", tokenizer=mock_tokenizer, max_batch_size=4, device="cpu")
    assert engine.max_batch_size == 4
    assert engine.scheduler.max_batch_size == 4
    assert len(engine.slot_allocator.free_slots) == 4


def test_engine_step_prefill(tiny_model, mock_tokenizer):
    # Setup Engine with Tiny Model
    engine = ContinuousBatchingEngine(model_path="dummy", tokenizer=mock_tokenizer, max_batch_size=2, device="cpu")
    # Ensure tiny_model is on CPU for this test
    tiny_model.to("cpu")
    tiny_model.eval()

    engine.load_model(tiny_model)

    # Add Request
    req = GenerationRequest(prompt="test", max_new_tokens=10)
    req.request_id = "req1"

    engine.add_request(req)

    # Step 1: Prefill
    engine.step()

    # Verify Sequence State
    seq = engine.scheduler.get_sequence("req1")
    assert seq.status == RequestState.RUNNING
    # Assuming "test" -> 4 len prompt (mock tokenizer)
    assert len(seq.input_ids) == 4
    # Prefill should generate 1 new token
    assert len(seq.generated_ids) == 1

    # Verify Slot Allocation
    assert engine.slot_allocator.get_slot("req1") != -1

    # Verify we can access the sequence again
    assert seq.request_id == "req1"


def test_engine_single_step_decode(tiny_model, mock_tokenizer):
    # Setup Engine with Tiny Model
    engine = ContinuousBatchingEngine(model_path="dummy", tokenizer=mock_tokenizer, max_batch_size=2, device="cpu")
    tiny_model.to("cpu")
    tiny_model.eval()
    engine.load_model(tiny_model)

    # Add Request
    req = GenerationRequest(prompt="test", max_new_tokens=10)
    req.request_id = "req2"
    engine.add_request(req)

    # Step 1: Prefill
    engine.step()

    seq = engine.scheduler.get_sequence("req2")
    assert len(seq.generated_ids) == 1

    # Step 2: Decode
    engine.step()

    # Verify Sequence State updated
    assert len(seq.generated_ids) == 2
    assert seq.status == RequestState.RUNNING

    # Verify we are still in the same slot
    assert engine.slot_allocator.get_slot("req2") != -1
