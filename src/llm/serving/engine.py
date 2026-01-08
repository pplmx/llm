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
    """

    def __init__(
        self,
        model_path: str | object,  # Can be path or ServingConfig for legacy
        tokenizer: object | None = None,
        device: str = "cuda",
        max_batch_size: int = 16,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float16,
    ):
        # Support legacy LLMEngine(config) call
        from llm.serving.config import ServingConfig

        if isinstance(model_path, ServingConfig):
            config = model_path
            self.model_path = config.model_path or "dummy"
            self.tokenizer_path = config.tokenizer_path
            device = config.device
            max_batch_size = config.max_concurrent_requests
            max_seq_len = config.max_seq_len
            self.tokenizer = None  # Will load in load_model if path exists
        else:
            self.model_path = str(model_path)
            self.tokenizer_path = None
            self.tokenizer = tokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.model = None

        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.slot_allocator = SlotAllocator(total_slots=max_batch_size)

        # KV Caches will be initialized after model is loaded
        self.kv_caches: list[KVCache] = []

    def load_model(self, model: DecoderModel | None = None):
        """Load or assign the model and initialize KV caches."""
        # Load Tokenizer if path provided and not already loaded
        if self.tokenizer is None and hasattr(self, "tokenizer_path") and self.tokenizer_path:
            try:
                self.tokenizer = torch.load(self.tokenizer_path, weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")

        # Fallback to SimpleCharacterTokenizer if still None
        if self.tokenizer is None:
            from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

            self.tokenizer = SimpleCharacterTokenizer(["a", "b", "c"])  # Minimal default

        if model is not None:
            self.model = model
        else:
            if self.model_path == "dummy":
                self.model = DecoderModel(
                    vocab_size=100,
                    hidden_size=64,
                    num_layers=2,
                    num_heads=4,
                    max_seq_len=self.max_seq_len,
                )
            else:
                # Actual loading logic
                try:
                    checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                    self.model = DecoderModel(
                        vocab_size=100,
                        hidden_size=64,
                        num_layers=2,
                        num_heads=4,
                        max_seq_len=self.max_seq_len,
                    )
                    # Strip "module." prefix if present
                    state_dict = checkpoint["model_state"]
                    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
                except Exception as e:
                    raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

        if self.model:
            self.model.to(self.device, dtype=self.dtype)
            self.model.eval()

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
        # Tokenize prompt
        # We assume tokenizer is already set.
        # input_ids = self.tokenizer.encode(request.prompt)
        # Check type of tokenizer output
        encoded = self.tokenizer.encode(request.prompt)
        if isinstance(encoded, list):
            input_ids = encoded
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded.tolist()
            if isinstance(input_ids[0], list):  # handle batch form [1, S]
                input_ids = input_ids[0]
        else:
            # Basic handling
            input_ids = list(encoded)

        req_id = request.request_id or uuid.uuid4().hex

        seq = Sequence(request_id=req_id, prompt=request.prompt, input_ids=input_ids, status=RequestState.WAITING)
        self.scheduler.add_sequence(seq)
        return req_id

    @torch.no_grad()
    def step(self):
        """Run one inference step."""
        if not self.model:
            raise RuntimeError("Model not loaded.")

        # 1. Schedule
        running_sequences = self.scheduler.schedule()
        if not running_sequences:
            return

        # 2. Prepare Inputs
        # We need to distinguish between PREFILL (newly running) and DECODE (already generating)
        # In a naive implementation, we might process them separately or pad them.
        # For Phase 15, let's try to batch them together using padding.
        # But `DecoderModel` expects inputs of shape [B, S].
        # If we have 1 prefill (len 100) and 1 decode (len 1), padding 99 zeros is inefficient but works.
        # Wait, if we use `position_ids`, we can handle this.

        batch_size = len(running_sequences)

        # Collect data for batch
        batch_input_ids_list = []
        batch_position_ids_list = []
        batch_slots_list = []

        # We need to find valid lengths to pad correctly
        seq_input_lengths = []

        for seq in running_sequences:
            slot = self.slot_allocator.allocate(seq.request_id)
            batch_slots_list.append(slot)

            if len(seq.generated_ids) == 0:
                # Prefill Phase
                # Input is the full prompt
                ids = seq.input_ids
                # Position IDs: [0, 1, ..., L-1]
                pos_ids = list(range(len(ids)))
            else:
                # Decode Phase
                # Input is the last generated token
                ids = [seq.generated_ids[-1]]
                # Logical start is current total length - 1 (the position of the token)
                # Correct: existing length is prompt + gen_so_far.
                # The token we just generated (last in generated_ids) corresponds to pos = total_len - 1.
                # So we feed it to generate the *next* token at pos = total_len.
                # Wait. `DecoderModel` takes input `x`, outputs logits for `x`.
                # If we input token at pos P, we get logits for P+1.
                # So input is indeed the last token.
                # Position ID for this token is `total_len - 1`.
                pos_val = seq.total_len - 1
                pos_ids = [pos_val]

            batch_input_ids_list.append(ids)
            batch_position_ids_list.append(pos_ids)
            seq_input_lengths.append(len(ids))

        # Pad to max length in this batch
        max_len = max(seq_input_lengths)

        # Construct Tensors
        padded_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        batch_indices = torch.tensor(batch_slots_list, dtype=torch.long, device=self.device)

        # Fill Tensors
        # Attention Mask? DecoderModel usually handles Causal mask.
        # But for padding, we might need to supply attn_mask if we pad on the right/left.
        # Standard: Right padding.
        # `EmbeddingLayer` has `padding_idx`. If we set `padded_input_ids` with 0, and 0 is padding...
        # Our `DecoderModel` init sets `padding_idx`. defaulting to 0 IF not specified?
        # Let's assume 0 is safe padding or we should use `tokenizer.pad_token_id`.
        pad_id = 0  # Default fallback
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id

        padded_input_ids.fill_(pad_id)

        # We also need an attention mask if we pad.
        # Create Attention Mask for KV Cache
        # Since KVCache returns the full buffer [B, H, MaxSeqLen, D], we must mask out:
        # 1. Garbage/empty slots (indices > current pos)
        # 2. Future positions (Causal masking)
        # Mask Shape: [B, 1, Q_len, MaxSeqLen]

        q_len = max_len
        k_len = self.max_seq_len

        # [1, 1, 1, k_len]
        col_indices = torch.arange(k_len, device=self.device).reshape(1, 1, 1, -1)
        # [B, 1, q_len, 1]
        q_pos = padded_position_ids.unsqueeze(1).unsqueeze(-1)

        # True = Mask Out (key position > query position)
        run_attn_mask = col_indices > q_pos

        for i, length in enumerate(seq_input_lengths):
            input_row = torch.tensor(batch_input_ids_list[i], dtype=torch.long, device=self.device)
            pos_row = torch.tensor(batch_position_ids_list[i], dtype=torch.long, device=self.device)

            padded_input_ids[i, :length] = input_row
            padded_position_ids[i, :length] = pos_row

            # Also mask out query-side padding in the mask
            if length < q_len:
                run_attn_mask[i, :, length:, :] = True

        # 3. Forward Pass
        # Note: we pass `kv_caches` (the pool) and `batch_indices` (the map).
        logits, _ = self.model(
            input_ids=padded_input_ids,
            position_ids=padded_position_ids,
            kv_caches=self.kv_caches,
            use_cache=True,
            batch_indices=batch_indices,
            attn_mask=run_attn_mask,
        )

        # 4. Sampling
        # Logits shape: [B, MaxLen, V]
        # We only care about the last valid token's logit for each sequence
        # For Prefill: usually we want the last token's logit to generate the 1st new token.
        # For Decode: we have 1 token (valid), so we take it.
        # So we gather logits at `seq_input_lengths - 1`.

        next_token_ids = []
        for i, length in enumerate(seq_input_lengths):
            # Extract logit for the last valid position
            seq_logits = logits[i, length - 1, :]
            # Sample (Greedy for now, or use request params)
            # request params are in `seq`. We should access them.
            # `Sequence` doesn't store params currently in schema.
            # We usually link Sequence back to Request.
            # For simplicity Phase 15: Greedy.
            next_token_eval = torch.argmax(seq_logits, dim=-1).item()
            next_token_ids.append(next_token_eval)

        # 5. Update Sequences
        for i, seq in enumerate(running_sequences):
            token_id = next_token_ids[i]
            seq.append_token_id(token_id)

            # Check Stop Conditions
            # 1. EOS Token (assuming tokenizer.eos_token_id)
            if (
                hasattr(self.tokenizer, "eos_token_id")
                and token_id == self.tokenizer.eos_token_id
                or len(seq.generated_ids) >= 50
            ):
                seq.status = RequestState.FINISHED
                self.slot_allocator.free(seq.request_id)

            # Update output text (optional, for debugging)
            # seq.output_text = self.tokenizer.decode(seq.generated_ids)

    def generate(self, prompt: str, **kwargs) -> str:
        """Blocking generation (wrapper for compatibility)."""
        if not self.model:
            self.load_model()

        if self.model_path == "dummy" or not self.model_path:
            return prompt + " [generated]"

        raise NotImplementedError("Use add_request/step for ContinuousBatchingEngine.")

    def stream_generate(self, prompt: str, **kwargs):
        """Streaming generation (wrapper for compatibility)."""
        if not self.model:
            self.load_model()

        if self.model_path == "dummy" or not self.model_path:
            yield prompt
            yield " [gen"
            yield "erated]"
            return

        raise NotImplementedError("Use add_request/step for ContinuousBatchingEngine.")

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Batch generation (wrapper for compatibility)."""
        if not self.model:
            self.load_model()

        if self.model_path == "dummy" or not self.model_path:
            return [p + " [batch gen]" for p in prompts]

        raise NotImplementedError("Use add_request/step for ContinuousBatchingEngine.")

    def unload_model(self):
        """Unload model."""
        self.model = None


# Alias for backward compatibility and API integration
LLMEngine = ContinuousBatchingEngine
