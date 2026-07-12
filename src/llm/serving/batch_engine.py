from __future__ import annotations

import hashlib
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llm.core.kv_cache import KVCache
from llm.generation.sampling import apply_repetition_penalty, sample_next_token
from llm.models.decoder import DecoderModel
from llm.serving.scheduler import Scheduler
from llm.serving.schemas import GenerationRequest, RequestState, Sequence


@dataclass(frozen=True)
class StepStats:
    """Per-step stats returned by :meth:`ContinuousBatchingEngine.step`.

    ``scheduled`` is the number of sequences that ran a forward pass in
    the step (a.k.a. effective batch size). ``total_active_slots`` is the
    engine's full slot pool — used as the denominator for the
    ``llm_batch_fill_ratio`` Prometheus gauge.
    """

    scheduled: int
    total_active_slots: int


class SlotPrefixCache:
    """Maps token prefixes to KV cache slots for reuse across requests."""

    def __init__(self, max_prefixes: int = 10, min_prefix_len: int = 4) -> None:
        self.max_prefixes = max_prefixes
        self.min_prefix_len = min_prefix_len
        self._entries: OrderedDict[str, tuple[int, int]] = OrderedDict()

    @staticmethod
    def hash_tokens(tokens: list[int]) -> str:
        return hashlib.sha256(bytes(tokens)).hexdigest()

    def get(self, tokens: list[int]) -> tuple[int, int] | None:
        if len(tokens) < self.min_prefix_len:
            return None
        return self._entries.get(self.hash_tokens(tokens))

    def put(self, tokens: list[int], slot: int, prefix_len: int) -> None:
        if len(tokens) < self.min_prefix_len:
            return
        key = self.hash_tokens(tokens)
        if len(self._entries) >= self.max_prefixes and key not in self._entries:
            self._entries.popitem(last=False)
        self._entries[key] = (slot, prefix_len)
        self._entries.move_to_end(key)


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
        *,
        enable_prefix_cache: bool = False,
        max_prefixes: int = 10,
        use_paged_attention: bool = False,
        max_blocks: int = 256,
        block_size: int = 16,
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

        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = SlotPrefixCache(max_prefixes=max_prefixes) if enable_prefix_cache else None
        self.paged_kv_cache = None
        if use_paged_attention:
            from llm.core.paged_attention.paged_kv_cache import PagedKVCache

            self.paged_kv_cache = PagedKVCache(
                num_layers=len(self.model.transformer_blocks),
                num_kv_heads=self.model.transformer_blocks[0].self_attn.num_kv_heads,
                head_dim=self.model.transformer_blocks[0].self_attn.head_dim,
                num_blocks=max_blocks,
                block_size=block_size,
                device=str(self.device),
                dtype=self.dtype,
                enable_prefix_cache=enable_prefix_cache,
                max_prefixes=max_prefixes,
            )

        # Concurrency control. ``step()`` mutates Python bookkeeping
        # (``self._seq_len``, ``self.free_slots``, ``self.kv_caches``, prefix
        # cache) that is not thread-safe. FastAPI's ``run_in_threadpool`` calls
        # ``service.generate`` from multiple worker threads, so we serialize.
        # PyTorch CUDA ops have their own internal serialization; this lock
        # only guards the Python-side state machine. Future async refactors
        # should release this lock during the inner model forward.
        self._step_lock = threading.Lock()
        # Optional callback invoked once per ``step()`` with the resulting
        # :class:`StepStats`. The serving tier uses it to publish
        # ``llm_batch_fill_ratio``. Called under ``self._step_lock`` so the
        # callback sees consistent post-step state.
        self._on_step: Callable[[StepStats], None] | None = None

    @classmethod
    def from_serving_config(cls, config, model: DecoderModel, tokenizer: object) -> ContinuousBatchingEngine:
        """Build an engine from ServingConfig flags.

        Raises ``NotImplementedError`` when ``config.use_paged_attention`` is True.
        Paged Attention is currently sidecar-only — the model forward path still
        uses the standard ``KVCache`` pool, so enabling it provides no benefit.
        See ``docs/adr/004-paged-attention-serving.md`` for the implementation plan.
        """
        if getattr(config, "use_paged_attention", False):
            raise NotImplementedError(
                "Paged Attention is not yet supported in the continuous batching engine "
                "forward path (sidecar-only). See "
                "docs/adr/004-paged-attention-serving.md for the current state. "
                "Set ServingConfig.use_paged_attention=False."
            )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=config.device,
            max_batch_size=config.max_concurrent_requests,
            max_seq_len=config.max_seq_len,
            enable_prefix_cache=config.enable_prefix_cache,
            max_prefixes=config.max_prefixes,
            use_paged_attention=config.use_paged_attention,
            max_blocks=config.max_blocks,
            block_size=config.block_size,
        )

    def _copy_kv_between_slots(self, src_slot: int, dst_slot: int, length: int) -> None:
        for cache in self.kv_caches:
            cache.k_cache[dst_slot, :, :length, :] = cache.k_cache[src_slot, :, :length, :].clone()
            cache.v_cache[dst_slot, :, :length, :] = cache.v_cache[src_slot, :, :length, :].clone()

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

        seq = Sequence(
            request_id=req_id,
            prompt=request.prompt,
            input_ids=input_ids,
            status=RequestState.WAITING,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        self.scheduler.add_sequence(seq)
        return req_id

    def stream_request(
        self,
        request: GenerationRequest,
    ):
        """Run a request to completion, yielding decoded text chunks."""
        req_id = self.add_request(request)
        emitted = 0
        while True:
            seq = self.scheduler.get_sequence(req_id)
            if seq is None:
                break
            if seq.is_finished():
                for token_id in seq.generated_ids[emitted:]:
                    yield self.tokenizer.decode([token_id])
                break
            self.step()
            seq = self.scheduler.get_sequence(req_id)
            if seq is None:
                break
            for token_id in seq.generated_ids[emitted:]:
                yield self.tokenizer.decode([token_id])
            emitted = len(seq.generated_ids)
            if seq.is_finished():
                break

    def generate_request(self, request: GenerationRequest) -> str:
        """Run a request to completion and return prompt + generated text."""
        chunks = list(self.stream_request(request))
        return request.prompt + "".join(chunks)

    def batch_generate_requests(self, requests: list[GenerationRequest]) -> list[str]:
        """Run multiple requests sequentially through the batching engine."""
        return [self.generate_request(request) for request in requests]

    @torch.no_grad()
    def step(self) -> StepStats:
        """Run one inference step.

        Holds ``self._step_lock`` for the duration of the call to serialize
        concurrent worker threads. ``run_in_threadpool`` offloads each request
        to a separate thread; without the lock, mutations to ``scheduler``,
        ``slot_allocator``, ``kv_caches``, and ``prefix_cache`` would race.

        Returns:
            :class:`StepStats` describing the step. ``scheduled`` is the
            effective batch size; ``total_active_slots`` is the engine's
            full slot pool (denominator for ``llm_batch_fill_ratio``).

        The engine optionally invokes ``self._on_step(stats)`` after the
        forward pass, under the step lock, so observers see consistent
        post-step state. The serving tier wires this to publish metrics.
        """
        with self._step_lock:
            stats = self._step_locked()
            if self._on_step is not None:
                self._on_step(stats)
            return stats

    def _step_locked(self) -> StepStats:
        """Body of :meth:`step`. Caller MUST hold ``self._step_lock``.

        Returns:
            :class:`StepStats` summarising this step. Always returned
            (never ``None``) so callers can compute fill-ratio even when
            the scheduler had no work — a fully idle engine reports
            ``scheduled=0`` rather than crashing the metric pipeline.
        """
        running_sequences = self.scheduler.schedule()
        if not running_sequences:
            return StepStats(scheduled=0, total_active_slots=self.slot_allocator.total_slots)

        batch_size = len(running_sequences)

        batch_input_ids_list = []
        batch_position_ids_list = []
        batch_slots_list = []
        seq_input_lengths = []
        prefix_full_hits: list[bool] = []

        for seq in running_sequences:
            slot = self.slot_allocator.allocate(seq.request_id)
            batch_slots_list.append(slot)
            prefix_full_hit = False

            if len(seq.generated_ids) == 0:
                cached = self.prefix_cache.get(seq.input_ids) if self.prefix_cache else None
                if cached is not None and cached[1] == len(seq.input_ids):
                    src_slot, prefix_len = cached
                    if src_slot != slot:
                        self._copy_kv_between_slots(src_slot, slot, prefix_len)
                    ids = [seq.input_ids[-1]]
                    pos_ids = [prefix_len - 1]
                    prefix_full_hit = True
                else:
                    ids = seq.input_ids
                    pos_ids = list(range(len(ids)))
            else:
                ids = [seq.generated_ids[-1]]
                pos_val = seq.total_len - 1
                pos_ids = [pos_val]

            batch_input_ids_list.append(ids)
            batch_position_ids_list.append(pos_ids)
            seq_input_lengths.append(len(ids))
            prefix_full_hits.append(prefix_full_hit)

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
            seq = running_sequences[i]
            seq_logits = logits[i, length - 1, :]
            context_ids = seq.input_ids + seq.generated_ids
            if seq.repetition_penalty != 1.0:
                seq_logits = apply_repetition_penalty(seq_logits, context_ids, seq.repetition_penalty)
            next_token_ids.append(
                sample_next_token(
                    seq_logits,
                    temperature=seq.temperature,
                    top_k=seq.top_k,
                    top_p=seq.top_p,
                )
            )

        for i, seq in enumerate(running_sequences):
            token_id = next_token_ids[i]
            seq.append_token_id(token_id)

            if self.prefix_cache and len(seq.generated_ids) == 1 and not prefix_full_hits[i]:
                self.prefix_cache.put(seq.input_ids, batch_slots_list[i], len(seq.input_ids))

            if (
                (hasattr(self.tokenizer, "eos_token_id") and token_id == self.tokenizer.eos_token_id)
                or len(seq.generated_ids) >= seq.max_new_tokens
                or seq.total_len >= self.max_seq_len
            ):
                seq.status = RequestState.FINISHED
                self.slot_allocator.free(seq.request_id)

        return StepStats(
            scheduled=batch_size,
            total_active_slots=self.slot_allocator.total_slots,
        )

    def set_step_observer(self, callback: Callable[[StepStats], None] | None) -> None:
        """Install or clear a per-step observer (used for metric publishing).

        The callback runs at the end of every :meth:`step` call, under
        ``self._step_lock``, with the :class:`StepStats` for that step.
        Pass ``None`` to remove a previously installed observer.
        """
        self._on_step = callback

    def unload_model(self):
        """Unload model."""
        self.model = None
        self.kv_caches = []
