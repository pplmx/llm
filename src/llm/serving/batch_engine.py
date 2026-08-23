from __future__ import annotations

import array
import asyncio
import hashlib
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from llm.core.kv_cache import KVCache
from llm.generation.sampling import (
    apply_frequency_penalty,
    apply_logit_bias,
    apply_presence_penalty,
    apply_repetition_penalty,
    mask_undecodable_logits,
    sample_next_token,
)
from llm.models.decoder import DecoderModel
from llm.serving.scheduler import Scheduler
from llm.serving.schemas import GenerationRequest, RequestState, Sequence
from llm.tokenization.tokenizer import BaseTokenizer


@runtime_checkable
class _CacheSizedAttention(Protocol):
    """Minimal structural type for attention backends that expose the
    KV-cache sizing attributes (MHA / MLA / FlashAttention all do)."""

    num_kv_heads: int
    head_dim: int


if TYPE_CHECKING:
    pass


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


@dataclass
class _StepInputs:
    """Inputs handed from the pre-compute (lock-protected) to the forward.

    Holds the dense batch tensors plus references to the running
    sequences — no Python-side state in the slot allocator or
    scheduler. The forward phase is free to mutate these tensors and
    produce a :class:`_StepResult` without holding the lock.
    """

    running_sequences: list[Sequence]
    batch_slots_list: list[int]
    seq_input_lengths: list[int]
    prefix_full_hits: list[bool]
    padded_input_ids: torch.Tensor
    padded_position_ids: torch.Tensor
    batch_indices: torch.Tensor
    run_attn_mask: torch.Tensor
    batch_size: int


@dataclass
class _StepResult:
    """Outputs handed from the forward (lock-free) to the post-compute.

    The post-compute mutates sequence status and frees slots, so it
    MUST re-acquire the lock before touching :class:`_StepResult`.
    """

    inputs: _StepInputs
    next_token_ids: list[int] = field(default_factory=list)
    forward_failed: BaseException | None = None


class SlotPrefixCache:
    """Maps token prefixes to KV cache slots for reuse across requests."""

    def __init__(self, max_prefixes: int = 10, min_prefix_len: int = 4) -> None:
        self.max_prefixes = max_prefixes
        self.min_prefix_len = min_prefix_len
        self._entries: OrderedDict[str, tuple[int, int]] = OrderedDict()

    @staticmethod
    def hash_tokens(tokens: list[int]) -> str:
        # Token ids from real tokenizers (BPE, SentencePiece) routinely
        # exceed 255 (vocabularies of 50 k-128 k are the norm).  ``bytes()``
        # rejects values outside ``[0, 256)`` and would crash the prefix
        # cache on the first request with a large vocabulary.  Use
        # ``array.array('i', ...)`` to pack each id as a 4-byte C int —
        # deterministic, collision-free for distinct lists, and works
        # across the full ``int`` range.
        return hashlib.sha256(array.array("i", tokens).tobytes()).hexdigest()

    def get(self, tokens: list[int]) -> tuple[int, int] | None:
        if len(tokens) < self.min_prefix_len:
            return None
        key = self.hash_tokens(tokens)
        if key in self._entries:
            # Promote to most-recently-used so LRU eviction evicts the
            # correct entry.
            self._entries.move_to_end(key)
            return self._entries[key]
        return None

    def put(self, tokens: list[int], slot: int, prefix_len: int) -> None:
        if len(tokens) < self.min_prefix_len:
            return
        key = self.hash_tokens(tokens)
        if len(self._entries) >= self.max_prefixes and key not in self._entries:
            self._entries.popitem(last=False)
        self._entries[key] = (slot, prefix_len)
        self._entries.move_to_end(key)

    def invalidate_for_slot(self, slot: int) -> None:
        """Drop every prefix entry that points at ``slot``.

        When a sequence finishes, its KV slot returns to the free pool and a
        later request may be allocated the same slot, overwriting the cached
        K/V.  If the stale prefix entry were left in place, a later request
        with the *same* prompt would hit the cache and replay another
        request's (now-overwritten or in-flight) K/V as its own prefix —
        a use-after-free of the cached KV.  Entries are removed rather than
        re-pointed so the most-recent-usage ordering is unaffected for the
        remaining prefixes.
        """
        if not self._entries:
            return
        stale = [key for key, (cached_slot, _len) in self._entries.items() if cached_slot == slot]
        for key in stale:
            del self._entries[key]


class SlotAllocator:
    """Manages allocation of KV cache slots."""

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self.free_slots: set[int] = set(range(total_slots))
        self.seq_to_slot: dict[str, int] = {}  # request_id -> slot_id

    @property
    def num_free(self) -> int:
        """Number of available slots."""
        return len(self.free_slots)

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
        tokenizer: BaseTokenizer,
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

        # Initialize KV Cache Pool. The dense ``KVCache`` pool is only
        # built when paged attention is disabled — when enabled, the
        # block-allocator pool below replaces it for the model forward
        # path (and building both would waste memory).
        self.kv_caches: list[KVCache] = []
        if not use_paged_attention:
            first_attn = self.model.transformer_blocks[0].self_attn
            if not isinstance(first_attn, _CacheSizedAttention):
                raise TypeError(f"attention backend {type(first_attn).__name__} must expose num_kv_heads/head_dim")
            self.kv_caches = KVCache.from_model_config(
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                num_layers=len(self.model.transformer_blocks),
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = SlotPrefixCache(max_prefixes=max_prefixes) if enable_prefix_cache else None
        self.paged_kv_cache = None
        if use_paged_attention:
            from llm.core.paged_attention.paged_kv_cache import PagedKVCache

            paged_attn = self.model.transformer_blocks[0].self_attn
            if not isinstance(paged_attn, _CacheSizedAttention):
                raise TypeError(f"attention backend {type(paged_attn).__name__} must expose num_kv_heads/head_dim")
            self.paged_kv_cache = PagedKVCache(
                num_layers=len(self.model.transformer_blocks),
                num_kv_heads=paged_attn.num_kv_heads,
                head_dim=paged_attn.head_dim,
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
    def from_serving_config(cls, config, model: DecoderModel, tokenizer: BaseTokenizer) -> ContinuousBatchingEngine:
        """Build an engine from ServingConfig flags.

        Paged Attention is fully wired through the continuous batching
        forward path (``docs/adr/004-paged-attention-serving.md`` was
        flipped to "Accepted" with this slice). When
        ``config.use_paged_attention=True`` the engine builds a
        :class:`PagedKVCache`, passes it to the model forward, and
        frees per-request blocks on sequence completion.
        """
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
        if self.paged_kv_cache is not None:
            # Prefix replay on the paged path does not copy K/V between dense
            # slots — ``_lock_step_pre`` instead stages the cached blocks into
            # the new sequence via ``PagedKVCache.stage_prefix`` (shared, no
            # data copy). Nothing to do here.
            return
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

        # Reject requests that would outgrow the context window BEFORE they
        # are scheduled. The model computes a position id for every prompt and
        # generated token; past the capacity the positional-encoding table is
        # indexed out of bounds, which surfaces on CUDA as a device-side
        # assert that corrupts the CUDA context and can crash the whole
        # serving process, not just this request.
        #
        # The hard ceiling is the MODEL's positional capacity, not the engine's
        # KV-window budget — the engine window may be configured (or resized)
        # larger than the checkpoint's ``max_seq_len``, and a request that fits
        # the engine budget but exceeds the model capacity used to crash in the
        # positional-encoding range check (RIL round-73 serving deep-dive:
        # ``max_seq_len=128`` engine serving a ``max_seq_len=8`` model).
        budget = min(getattr(self.model, "max_seq_len", self.max_seq_len), self.max_seq_len)
        if len(input_ids) + request.max_new_tokens > budget:
            raise ValueError(
                f"Prompt has {len(input_ids)} tokens and max_new_tokens="
                f"{request.max_new_tokens}, but the model's context window is "
                f"{budget}; the request would exceed the context window and "
                "crash the forward pass"
            )

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
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=request.logit_bias,
            stop=request.stop,
        )

        # Reject a *different* request that reuses an active request_id (RIL
        # ISS-123/F3). The slot allocator keys KV slots by request_id, so two
        # distinct requests claiming the same id would both write into the
        # same slot — cross-request contamination plus a freed-slot reuse when
        # the first finishes. We must NOT reject unconditionally: the engine's
        # own ``generate_request`` -> ``stream_request`` re-adds the SAME
        # logical request (double-add contract), and the streaming reap loop
        # explicitly removes every copy. Only a request whose content differs
        # from the active holder's is a genuine collision.
        if request.request_id is not None:
            added = self.scheduler.add_sequence_if_not_conflicting(
                seq,
                matches=lambda existing: (
                    existing.prompt == seq.prompt
                    and existing.max_new_tokens == seq.max_new_tokens
                    and existing.temperature == seq.temperature
                    and existing.top_k == seq.top_k
                    and existing.top_p == seq.top_p
                    and existing.repetition_penalty == seq.repetition_penalty
                    and existing.frequency_penalty == seq.frequency_penalty
                    and existing.presence_penalty == seq.presence_penalty
                    and existing.logit_bias == seq.logit_bias
                    and existing.stop == seq.stop
                ),
            )
            if not added:
                raise ValueError(
                    f"request_id '{req_id}' is already in use by a different active "
                    "request; duplicate request ids would share one KV slot"
                )
        else:
            self.scheduler.add_sequence(seq)
        return req_id

    def _emit_tokens(
        self,
        sequence: Sequence,
        new_token_ids: list[int],
        stops: list[str] | None,
        max_stop_len: int,
        buffer: str,
    ) -> tuple[list[str], bool, str]:
        """Decode ``new_token_ids`` into text chunks, honouring stop sequences.

        Shared by both the finished-sequence drain and the post-step token
        emission paths in :meth:`stream_request`. ``buffer`` carries over
        any un-yielded tail from the previous call (the last
        ``<max_stop_len>`` characters, held for cross-token-boundary suffix
        checking). Returns a triple:

        - ``chunks``: list of text chunks to yield to the caller.
        - ``stop_hit``: True when a stop sequence matched as a suffix — the
          caller should terminate the streaming generator.
        - ``buffer``: remaining un-yielded buffer text for the next call.

        When ``stops`` is None / empty, each decoded chunk is returned
        directly and the buffer is returned unchanged (it stays empty).
        """
        chunks: list[str] = []
        stop_hit = False

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        for token_id in new_token_ids:
            # The EOS token ends generation; never emit its decoded text
            # (parity with the eager/speculative backends — RIL ISS-96/98).
            # ``_lock_step_post`` already appended it and marked the sequence
            # FINISHED, so the drain here simply stops short of the EOS.
            if eos_id is not None and token_id == eos_id:
                break
            text_chunk = self.tokenizer.decode([token_id])
            if stops and text_chunk:
                buffer += text_chunk
                for s in stops:
                    if buffer.endswith(s):
                        prefix = buffer[: len(buffer) - len(s)]
                        if prefix:
                            chunks.append(prefix)
                        sequence.status = RequestState.FINISHED
                        stop_hit = True
                        return chunks, stop_hit, ""
                # No stop match — yield the safe prefix (everything beyond
                # the last max_stop_len characters) and keep the tail.
                if len(buffer) > max_stop_len:
                    safe_len = len(buffer) - max_stop_len
                    chunks.append(buffer[:safe_len])
                    buffer = buffer[safe_len:]
            else:
                chunks.append(text_chunk)

        return chunks, stop_hit, buffer

    def stream_request(
        self,
        request: GenerationRequest,
    ):
        """Run a request to completion, yielding decoded text chunks.

        Honours ``stop`` sequences: when the accumulated generated text
        (post-prompt) ends with any stop string, generation halts and the
        stop string itself is excluded from the yielded output (OpenAI
        semantics).
        """
        req_id = self.add_request(request)
        from llm.generation.eager import _normalize_stop

        stops = _normalize_stop(request.stop)
        max_stop_len = max((len(s) for s in stops), default=0) if stops else 0
        buffer = ""
        emitted = 0
        try:
            while True:
                seq = self.scheduler.get_sequence(req_id)
                if seq is None:
                    break
                if seq.is_finished():
                    chunks, stop_hit, buffer = self._emit_tokens(
                        seq, seq.generated_ids[emitted:], stops, max_stop_len, buffer
                    )
                    yield from chunks
                    if stop_hit:
                        # A stop-sequence match finished the request/sequence
                        # *outside* a step, so ``_lock_step_post`` never ran its
                        # free path — release the slot here or it leaks (ISS-044).
                        with self._step_lock:
                            self._release_request_slot_by_id(req_id)
                        return
                    if stops and buffer:
                        # The sequence is already finished, so after draining the
                        # tail buffer there is nothing left to stream. Return
                        # (not ``break``): the post-loop ``yield buffer`` would
                        # emit the SAME tail a second time (RIL ISS-054).
                        yield buffer
                    return
                self.step()
                # Reuse the ``seq`` reference captured BEFORE the step (from
                # the ``get_sequence`` at the top of this iteration). The
                # post-step re-fetch was a race: a concurrent generator's
                # ``step()`` -> ``Scheduler.schedule()`` can evict a just-
                # finished sequence from ``running`` between our ``step()``
                # releasing ``_step_lock`` and this ``get_sequence``
                # re-acquiring the scheduler lock, so the re-fetch returned
                # ``None`` and the final step's token(s) were silently
                # dropped (RIL F1 — concurrent stream truncation). The
                # ``Sequence`` object itself is never destroyed until the
                # ``finally`` reap below, and ``step()`` appended this
                # iteration's tokens to ``seq.generated_ids`` in place, so
                # the already-held reference drains them correctly.
                chunks, stop_hit, buffer = self._emit_tokens(
                    seq, seq.generated_ids[emitted:], stops, max_stop_len, buffer
                )
                yield from chunks
                if stop_hit:
                    # Same leak vector as above.
                    with self._step_lock:
                        self._release_request_slot_by_id(req_id)
                    return
                emitted = len(seq.generated_ids)
                if seq.is_finished():
                    break
            if stops and buffer:
                yield buffer
        finally:
            # Cleanup for ANY exit path — including an *abandoned* generator.
            # ``stream_request`` is the sequence's only stepper; when the
            # consumer disconnects (gen.close() / GC) it can never advance the
            # sequence again, so a RUNNING sequence with its slot allocated
            # would otherwise leak one ``max_batch_size`` slot per disconnect
            # (RIL ISS-105). Reap the sequence and release its slot here;
            # ``_release_request_slot_by_id`` is a no-op if it was already
            # freed (normal completion, stop-halt, forward failure). Loop
            # because a caller may have added the request twice (a stray
            # ``add_request`` before streaming) — every copy must be reaped.
            #
            # ``scheduler.remove`` mutates the LIVE ``running``/``waiting``
            # lists, so it MUST run under ``_step_lock`` like every other
            # engine mutation — otherwise it races with ``_lock_step_pre``
            # iterating the ``running`` list a concurrent ``step()`` returned,
            # and the mid-iteration pop silently skips/duplicates a sequence
            # for that step (RIL ISS-117, a regression the ISS-105 fix
            # originally left un-serialized).
            with self._step_lock:
                while self.scheduler.remove(req_id) is not None:
                    self._release_request_slot_by_id(req_id)

    def generate_request(self, request: GenerationRequest) -> str:
        """Run a request to completion and return prompt + generated text."""
        chunks = list(self.stream_request(request))
        return request.prompt + "".join(chunks)

    def batch_generate_requests(self, requests: list[GenerationRequest]) -> list[str]:
        """Run multiple requests sequentially through the batching engine."""
        return [self.generate_request(request) for request in requests]

    @torch.no_grad()
    def step(self) -> StepStats:
        """Run one inference step (sync wrapper).

        The whole step — pre-compute, model forward, post-compute — holds
        ``self._step_lock`` so two concurrent ``step()`` calls can never both
        run the forward against the same slots and append two tokens to the
        same sequence from one logical step (a real corruption when the
        ``batched`` backend serves concurrent HTTP requests from FastAPI's
        threadpool). Holding the lock across the forward does NOT serialize
        request enqueueing: :meth:`add_request` never takes the step lock, so
        new requests still arrive in parallel. The forward is the only code
        that mutates the KV caches / slot bookkeeping the post step reads, so
        this is exactly what the lock needs to guard.

        Returns:
            :class:`StepStats` describing the step. ``scheduled`` is the
            effective batch size; ``total_active_slots`` is the engine's
            full slot pool (denominator for ``llm_batch_fill_ratio``).
        """
        with self._step_lock:
            inputs = self._lock_step_pre()
            if inputs is None:
                stats = StepStats(scheduled=0, total_active_slots=self.slot_allocator.total_slots)
            else:
                result = self._forward_and_sample(inputs)
                stats = self._lock_step_post(result)
            if self._on_step is not None:
                self._on_step(stats)
        return stats

    async def step_async(self) -> StepStats:
        """Run one inference step, yielding to the event loop during the forward.

        Identical contract to :meth:`step`, but the model forward runs
        in a worker thread via :func:`asyncio.to_thread`. The lock is
        only held for the bookkeeping portions (pre + post). This lets
        the FastAPI event loop keep processing I/O (other requests,
        health checks, /metrics scrapes) while a forward pass runs.

        .. warning::
           On the PAGED path (``paged_kv_cache`` set) the forward mutates the
           block manager (allocation / extension / copy-on-write of shared
           prefix blocks — RIL TASK-065) which is NOT thread-safe. Two
           overlapping ``step_async`` calls would interleave those mutations
           and corrupt the block table. Production serving uses the
           synchronous :meth:`step` (which holds the lock across the whole
           forward) via ``run_in_threadpool``; callers that enable paged
           attention + prefix caching must use :meth:`step`, not
           ``step_async``, until the cache is externally synchronized.

        Returns:
            :class:`StepStats` (same fields as :meth:`step`).
        """
        with self._step_lock:
            inputs = self._lock_step_pre()
        if inputs is None:
            stats = StepStats(scheduled=0, total_active_slots=self.slot_allocator.total_slots)
        else:
            result = await asyncio.to_thread(self._forward_and_sample, inputs)
            with self._step_lock:
                stats = self._lock_step_post(result)
        if self._on_step is not None:
            with self._step_lock:
                self._on_step(stats)
        return stats

    def _lock_step_pre(self) -> _StepInputs | None:
        """Acquire work from the scheduler and build the dense batch.

        Caller MUST hold ``self._step_lock``. Returns ``None`` when
        there is no work to do (idle engine).
        """
        running_sequences = self.scheduler.schedule()
        if not running_sequences:
            return None

        batch_size = len(running_sequences)

        batch_input_ids_list: list[list[int]] = []
        batch_position_ids_list: list[list[int]] = []
        batch_slots_list: list[int] = []
        seq_input_lengths: list[int] = []
        prefix_full_hits: list[bool] = []

        for seq in running_sequences:
            slot = self.slot_allocator.allocate(seq.request_id)
            batch_slots_list.append(slot)
            prefix_full_hit = False

            if len(seq.generated_ids) == 0:
                # Dense path prefix replay: ``_copy_kv_between_slots`` copies
                # the cached K/V into a fresh slot and only the final prompt
                # token runs through the model.
                use_prefix_shortcut = self.prefix_cache is not None and self.paged_kv_cache is None
                cached = self.prefix_cache.get(seq.input_ids) if use_prefix_shortcut else None
                if cached is not None and cached[1] == len(seq.input_ids):
                    src_slot, prefix_len = cached
                    if src_slot != slot:
                        self._copy_kv_between_slots(src_slot, slot, prefix_len)
                    ids = [seq.input_ids[-1]]
                    pos_ids = [prefix_len - 1]
                    prefix_full_hit = True
                elif (
                    self.paged_kv_cache is not None
                    and self.paged_kv_cache.enable_prefix_cache
                    and len(seq.input_ids) > 0
                ):
                    # Paged path prefix replay (RIL TASK-065). An exact
                    # full-prompt match stages the cached blocks into the new
                    # sequence SHARED (refcounted — ``stage_prefix`` forks
                    # them) and runs only the final prompt token, mirroring
                    # the dense shortcut: the model re-writes that token's K/V
                    # at position N-1, which is idempotent with the cached
                    # value, and ``PagedKVCache.update`` copy-on-writes if the
                    # boundary block is still shared so the cache owner's K/V
                    # is never corrupted.
                    prefix_blocks = self.paged_kv_cache.try_get_prefix_blocks(seq.input_ids)
                    if prefix_blocks is not None:
                        self.paged_kv_cache.stage_prefix(slot, prefix_blocks, len(seq.input_ids) - 1)
                        ids = [seq.input_ids[-1]]
                        pos_ids = [len(seq.input_ids) - 1]
                        prefix_full_hit = True
                    else:
                        ids = seq.input_ids
                        pos_ids = list(range(len(ids)))
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
        # The causal attention mask must be built from the *real* position
        # ids, which are only known after the per-row fill below. Building
        # it from the freshly-allocated (all-zero) ``padded_position_ids``
        # collapsed every query row to q_pos=0, so decode attention could
        # only ever see the first prompt token's KV and outputs diverged
        # from the eager backend after a few decode steps. All-True
        # (masked) by default; the visible causal region for each query
        # row is set after the fill.
        run_attn_mask = torch.ones((batch_size, 1, q_len, k_len), dtype=torch.bool, device=self.device)

        # Sparse/streaming scheme (RIL TASK-246): when the served model carries an
        # ``attn_sparse`` scheme, fold its sink+window pattern into the mask so the
        # batched/paged serving path actually constrains keys instead of silently
        # running dense (the decoder's own auto-mask is bypassed here because the
        # engine always supplies an explicit ``attn_mask``). The pattern is
        # absolute-position based over the full key window; per query row it is
        # indexed by the real position id below and OR-ed with the causal mask.
        sparse_mask_out = None
        _spar = getattr(self.model, "attn_sparse", None) if self.model is not None else None
        if _spar:
            from llm.core.attn.sparse import build_sparse_attention_mask

            _params = dict(_spar)
            _kind = _params.pop("kind")
            _params.pop("causal", None)  # causality comes from the causal mask
            _allow = build_sparse_attention_mask(_kind, k_len, causal=False, **_params)
            sparse_mask_out = (~_allow).bool()  # True = mask out (SDPA convention)

        for i, length in enumerate(seq_input_lengths):
            input_row = torch.tensor(batch_input_ids_list[i], dtype=torch.long, device=self.device)
            pos_row = torch.tensor(batch_position_ids_list[i], dtype=torch.long, device=self.device)

            padded_input_ids[i, :length] = input_row
            padded_position_ids[i, :length] = pos_row

            # True = mask out columns > position[s] (the sdpa wrapper's
            # convention), so each query position s sees keys 0..position[s].
            q_pos_row = padded_position_ids[i, :length].reshape(1, 1, length, 1)
            run_attn_mask[i, :, :length, :] = col_indices > q_pos_row
            if sparse_mask_out is not None:
                # Fold the sparse pattern for each query row's real absolute
                # position (sink + window), on top of the causal mask.
                pos = padded_position_ids[i, :length].long()
                run_attn_mask[i, :, :length, :] = run_attn_mask[i, :, :length, :] | sparse_mask_out[pos]

        return _StepInputs(
            running_sequences=running_sequences,
            batch_slots_list=batch_slots_list,
            seq_input_lengths=seq_input_lengths,
            prefix_full_hits=prefix_full_hits,
            padded_input_ids=padded_input_ids,
            padded_position_ids=padded_position_ids,
            batch_indices=batch_indices,
            run_attn_mask=run_attn_mask,
            batch_size=batch_size,
        )

    def _forward_and_sample(self, inputs: _StepInputs) -> _StepResult:
        """Run the model forward and sampling WITHOUT holding the lock.

        This is the expensive path: ~ ms of GPU/CPU work depending on
        batch size and model size. The lock is released for the entire
        duration so other threads can pre-/post-compute in parallel.

        On forward failure we record the exception in the result so
        the caller can free slots + clean up state under the lock
        (so the engine stays consistent even when a forward raises).
        """
        try:
            if self.model is None:
                raise RuntimeError("engine model was unloaded")
            logits, _ = self.model(
                input_ids=inputs.padded_input_ids,
                position_ids=inputs.padded_position_ids,
                kv_caches=self.kv_caches if self.paged_kv_cache is None else None,
                paged_kv_cache=self.paged_kv_cache,
                use_cache=True,
                batch_indices=inputs.batch_indices,
                attn_mask=inputs.run_attn_mask,
            )

            next_token_ids: list[int] = []
            for i, length in enumerate(inputs.seq_input_lengths):
                seq = inputs.running_sequences[i]
                seq_logits = logits[i, length - 1, :]
                # The pad token must never be emitted (the eager backend
                # masks it the same way); without this the engine can
                # sample pad and diverge from eager.
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is not None and 0 <= pad_id < seq_logits.size(-1):
                    seq_logits = seq_logits.clone()
                    seq_logits[pad_id] = float("-inf")
                # Sample only tokenizer-decodable ids: a padded-vocab or
                # BPE/HF model served with a smaller-vocab tokenizer would
                # otherwise sample a tail id and crash in ``_emit_tokens`` at
                # ``tokenizer.decode([id])`` (KeyError), while eager and
                # speculative mask it on every step (RIL ISS-125).
                mask_undecodable_logits(seq_logits, getattr(self.tokenizer, "vocab_size", None))
                context_ids = seq.input_ids + seq.generated_ids
                if seq.repetition_penalty != 1.0:
                    seq_logits = apply_repetition_penalty(seq_logits, context_ids, seq.repetition_penalty)
                if seq.frequency_penalty != 0.0:
                    seq_logits = apply_frequency_penalty(seq_logits, context_ids, seq.frequency_penalty)
                if seq.presence_penalty != 0.0:
                    seq_logits = apply_presence_penalty(seq_logits, context_ids, seq.presence_penalty)
                if seq.logit_bias:
                    seq_logits = apply_logit_bias(seq_logits, seq.logit_bias)
                next_token_ids.append(
                    sample_next_token(
                        seq_logits,
                        temperature=seq.temperature,
                        top_k=seq.top_k,
                        top_p=seq.top_p,
                    )
                )
        except BaseException as exc:  # noqa: BLE001 - propagate via result
            return _StepResult(inputs=inputs, forward_failed=exc)

        return _StepResult(inputs=inputs, next_token_ids=next_token_ids)

    def _release_request_slots(self, request_id: str, slot: int) -> None:
        """Return a request's KV slot, prefix-cache entry and paged blocks.

        Shared by the finished-in-step path (:meth:`_lock_step_post`), the
        stop-sequence termination path (:meth:`stream_request`) and the
        forward-failure path. Omitting this anywhere leaves the slot
        permanently leaked (RIL ISS-044): the scheduler filters FINISHED
        sequences out of ``running`` without ever freeing their slot, so a
        slot leaked per stop-terminated request eventually exhausts the pool
        and every ``allocate()`` raises ``No free slots available in KV
        cache``.
        """
        self.slot_allocator.free(request_id)
        if self.prefix_cache is not None:
            # The slot is back in the free pool; a later request may reuse it
            # and overwrite the cached K/V. A leftover prefix entry would
            # replay stale/in-flight K/V on a prompt match.
            self.prefix_cache.invalidate_for_slot(slot)
        if self.paged_kv_cache is not None:
            # ``seq_id`` == slot id in the paged path.
            self.paged_kv_cache.free(slot)

    def _release_request_slot_by_id(self, request_id: str) -> None:
        """Release a request's slot, resolving the slot id from the allocator.

        Caller MUST hold ``self._step_lock`` (the same contract as
        :meth:`_lock_step_post`) — this mutates the shared allocator /
        prefix cache / paged pool. No-op when the request no longer holds a
        slot (e.g. it already finished inside a step and was released there).
        """
        slot = self.slot_allocator.get_slot(request_id)
        if slot < 0:
            return
        self._release_request_slots(request_id, slot)

    def _lock_step_post(self, result: _StepResult) -> StepStats:
        """Append sampled tokens, free slots, mark finished sequences.

        Caller MUST hold ``self._step_lock``. If the forward failed,
        we free the slots we allocated in pre but don't append any
        token — the sequences are left in their previous state.
        """
        inputs = result.inputs
        if result.forward_failed is not None:
            # Free the slots we allocated in pre, and mark the sequences
            # FINISHED so the next ``schedule()`` drops them. Without the
            # status flip a persistently-failing sequence (OOM, bad token
            # id, shape mismatch) stays RUNNING and is re-scheduled every
            # step — re-allocated, re-forwarded, re-failed — livelocking
            # the whole engine past the first error (RIL ISS-051).
            for i, seq in enumerate(inputs.running_sequences):
                seq.status = RequestState.FINISHED
                self._release_request_slots(seq.request_id, inputs.batch_slots_list[i])
            raise result.forward_failed

        for i, seq in enumerate(inputs.running_sequences):
            token_id = result.next_token_ids[i]
            seq.append_token_id(token_id)

            # Dense path maintains the dense SlotPrefixCache; the paged path
            # maintains its own block cache via PagedKVCache.add_prefix.
            if (
                self.prefix_cache
                and self.paged_kv_cache is None
                and len(seq.generated_ids) == 1
                and not inputs.prefix_full_hits[i]
            ):
                self.prefix_cache.put(seq.input_ids, inputs.batch_slots_list[i], len(seq.input_ids))

            if (
                self.paged_kv_cache is not None
                and self.paged_kv_cache.enable_prefix_cache
                and len(seq.generated_ids) == 1
                and not inputs.prefix_full_hits[i]
            ):
                # Register the just-prefilled prompt's blocks for future
                # replay. At this instant the block table holds exactly the
                # prompt's blocks (decode has not extended it yet), so the
                # cached prefix is the full prompt. Skipped for a prefix-hit
                # sequence (it shared the owner's blocks — re-registering the
                # same hash to this seq would make its free() wrongly evict
                # the owner's entry).
                slot = inputs.batch_slots_list[i]
                self.paged_kv_cache.add_prefix(slot, seq.input_ids, self.paged_kv_cache.get_block_table(slot))

            if (
                (hasattr(self.tokenizer, "eos_token_id") and token_id == self.tokenizer.eos_token_id)
                or len(seq.generated_ids) >= seq.max_new_tokens
                or seq.total_len >= self.max_seq_len
            ):
                seq.status = RequestState.FINISHED
                self._release_request_slots(seq.request_id, inputs.batch_slots_list[i])

        return StepStats(
            scheduled=inputs.batch_size,
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
        """Release model, KV caches and all scheduler state.

        Clears model weights, both legacy and paged KV caches, prefix
        cache, slot allocator mappings, and scheduler queues so that
        GPU memory is freed and the engine is reusable after this call.
        """
        self.model = None
        self.kv_caches = []
        self.paged_kv_cache = None
        self._on_step = None

        self.scheduler.clear()
        self.slot_allocator.seq_to_slot.clear()
        self.slot_allocator.free_slots = set(range(self.slot_allocator.total_slots))

        if self.prefix_cache is not None:
            self.prefix_cache._entries.clear()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
