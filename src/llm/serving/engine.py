import uuid

import torch

from llm.core.kv_cache import KVCache
from llm.models.decoder import DecoderModel
from llm.serving.scheduler import Scheduler
from llm.serving.schemas import GenerationRequest, RequestState, Sequence


class SlotAllocator:
    """Manages allocation of KV cache slots."""

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self.free_slots: set[int] = set(range(total_slots))
        self.seq_to_slot: dict[str, int] = {}  # request_id -> slot_id

    def allocate(self, request_id: str) -> int:
        if request_id in self.seq_to_slot:
            return self.seq_to_slot[request_id]
        if not self.free_slots:
            raise RuntimeError("No free slots available in KV cache.")
        slot = self.free_slots.pop()
        self.seq_to_slot[request_id] = slot
        return slot

    def free(self, request_id: str):
        if request_id in self.seq_to_slot:
            slot = self.seq_to_slot.pop(request_id)
            self.free_slots.add(slot)

    def get_slot(self, request_id: str) -> int:
        return self.seq_to_slot.get(request_id, -1)


class ContinuousBatchingEngine:
    """
    Inference engine supporting continuous batching (iteration-level scheduling).

    This is the primary serving engine. It manages request states, schedules
    sequences at an iteration level, and orchestrates the forward pass.
    """

    def __init__(
        self,
        model: DecoderModel,
        tokenizer: object,
        device: str | torch.device = "cuda",
        max_batch_size: int = 16,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the engine with an already-loaded model and tokenizer.

        Args:
            model: The loaded DecoderModel instance.
            tokenizer: The tokenizer instance (must have encode/decode methods).
            device: Target device ("cuda", "cpu", or torch.device).
            max_batch_size: Maximum number of concurrent sequences.
            max_seq_len: Maximum sequence length for KV cache.
            dtype: Data type for model and cache.
        """
        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        else:
            self.device = device

        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # Model setup
        self.model = model
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # Scheduler and Slot Allocator
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.slot_allocator = SlotAllocator(total_slots=max_batch_size)

        # Initialize KV Cache Pool
        self.kv_caches = KVCache.from_model_config(
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            num_layers=len(self.model.transformer_blocks),
            num_kv_heads=self.model.transformer_blocks[0].self_attn.num_kv_heads,
            head_dim=self.model.transformer_blocks[0].self_attn.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def add_request(self, request: GenerationRequest) -> str:
        """Add a request to the engine."""
        encoded = self.tokenizer.encode(request.prompt)
        if isinstance(encoded, list):
            input_ids = encoded
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded.tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
        else:
            input_ids = list(encoded)

        req_id = request.request_id or uuid.uuid4().hex

        seq = Sequence(request_id=req_id, prompt=request.prompt, input_ids=input_ids, status=RequestState.WAITING)
        self.scheduler.add_sequence(seq)
        return req_id

    @torch.no_grad()
    def step(self):
        """Run one inference step."""
        running_sequences = self.scheduler.schedule()
        if not running_sequences:
            return

        batch_size = len(running_sequences)

        batch_input_ids_list = []
        batch_position_ids_list = []
        batch_slots_list = []
        seq_input_lengths = []

        for seq in running_sequences:
            slot = self.slot_allocator.allocate(seq.request_id)
            batch_slots_list.append(slot)

            if len(seq.generated_ids) == 0:
                # Prefill Phase
                ids = seq.input_ids
                pos_ids = list(range(len(ids)))
            else:
                # Decode Phase
                ids = [seq.generated_ids[-1]]
                pos_val = seq.total_len - 1
                pos_ids = [pos_val]

            batch_input_ids_list.append(ids)
            batch_position_ids_list.append(pos_ids)
            seq_input_lengths.append(len(ids))

        max_len = max(seq_input_lengths)

        padded_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        batch_indices = torch.tensor(batch_slots_list, dtype=torch.long, device=self.device)

        pad_id = 0
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id

        padded_input_ids.fill_(pad_id)

        q_len = max_len
        k_len = self.max_seq_len

        col_indices = torch.arange(k_len, device=self.device).reshape(1, 1, 1, -1)
        q_pos = padded_position_ids.unsqueeze(1).unsqueeze(-1)
        run_attn_mask = col_indices > q_pos

        for i, length in enumerate(seq_input_lengths):
            input_row = torch.tensor(batch_input_ids_list[i], dtype=torch.long, device=self.device)
            pos_row = torch.tensor(batch_position_ids_list[i], dtype=torch.long, device=self.device)

            padded_input_ids[i, :length] = input_row
            padded_position_ids[i, :length] = pos_row

            if length < q_len:
                run_attn_mask[i, :, length:, :] = True

        logits, _ = self.model(
            input_ids=padded_input_ids,
            position_ids=padded_position_ids,
            kv_caches=self.kv_caches,
            use_cache=True,
            batch_indices=batch_indices,
            attn_mask=run_attn_mask,
        )

        next_token_ids = []
        for i, length in enumerate(seq_input_lengths):
            seq_logits = logits[i, length - 1, :]
            next_token_eval = torch.argmax(seq_logits, dim=-1).item()
            next_token_ids.append(next_token_eval)

        for i, seq in enumerate(running_sequences):
            token_id = next_token_ids[i]
            seq.append_token_id(token_id)

            if (
                hasattr(self.tokenizer, "eos_token_id")
                and token_id == self.tokenizer.eos_token_id
                or len(seq.generated_ids) >= 50
            ):
                seq.status = RequestState.FINISHED
                self.slot_allocator.free(seq.request_id)

    def unload_model(self):
        """Unload model."""
        self.model = None
        self.kv_caches = []
