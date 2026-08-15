from collections import deque
from threading import Lock

from llm.serving.schemas import RequestState, Sequence


class Scheduler:
    """Simple FCFS scheduler for continuous batching with backpressure.

    Limits the ``waiting`` queue to ``max_waiting`` to prevent unbounded
    memory growth under load. When the queue is full, :meth:`add_sequence`
    raises ``RuntimeError`` so callers can apply backpressure (HTTP 503).

    Thread safety: every method that reads or mutates ``waiting`` /
    ``running`` takes the scheduler's own :attr:`_lock`. The engine's
    ``step()`` path calls :meth:`schedule` under the engine's
    ``_step_lock``, while streaming/backpressure paths call
    :meth:`add_sequence`, :meth:`get_sequence` and :attr:`has_pending_work`
    WITHOUT the engine lock — without a scheduler-internal lock those
    concurrent ``popleft`` / iteration calls crashed with ``RuntimeError:
    deque mutated during iteration`` under concurrent streaming (RIL
    ISS-069). Lock ordering stays acyclic: ``schedule`` (engine lock →
    scheduler lock) never nests the engine lock inside the scheduler lock.
    """

    DEFAULT_MAX_WAITING = 1024

    def __init__(self, max_batch_size: int = 16, *, max_waiting: int | None = None):
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting if max_waiting is not None else self.DEFAULT_MAX_WAITING
        self._lock = Lock()

    @property
    def has_pending_work(self) -> bool:
        with self._lock:
            return len(self.waiting) > 0 or len(self.running) > 0

    def add_sequence(self, seq: Sequence):
        """Add a new sequence to the waiting queue.

        Raises:
            RuntimeError: If the waiting queue is at capacity.
        """
        with self._lock:
            if len(self.waiting) >= self.max_waiting:
                raise RuntimeError(
                    f"Waiting queue full ({len(self.waiting)}/{self.max_waiting}); retry later or increase max_waiting."
                )
            self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """
        Schedule sequences for the next inference step.
        Promotes waiting sequences to running if there is capacity.
        """
        with self._lock:
            # Clean up finished sequences (engine should ideally handle this, or we handle it here pre-schedule)
            # But if engine updates state to FINISHED, we can filter them out.
            # However, we usually want to return 'Finished' status to user once before removing.
            # Let's assume engine calls `free_completed` explicitly or we filter here.
            # Better: Filter out finished ones from `running` at the start.
            self.running = [s for s in self.running if not s.is_finished()]

            # Fill available slots
            while self.waiting and len(self.running) < self.max_batch_size:
                seq = self.waiting.popleft()
                seq.status = RequestState.RUNNING
                self.running.append(seq)

            return self.running

    def get_sequence(self, request_id: str) -> Sequence | None:
        """Find a sequence by its request_id."""
        with self._lock:
            for s in self.running:
                if s.request_id == request_id:
                    return s
            for s in self.waiting:
                if s.request_id == request_id:
                    return s
            return None

    def remove(self, request_id: str) -> Sequence | None:
        """Drop a sequence from the waiting queue / running list.

        Used by the streaming generator's cleanup path when the consumer
        abandons mid-generation (RIL ISS-105): the abandoned generator is the
        sequence's only stepper and can never advance it again, so leaving it
        RUNNING permanently consumes a KV slot (``schedule`` only filters
        FINISHED). Idempotent: returns the removed sequence or ``None``.
        """
        with self._lock:
            for i, s in enumerate(self.running):
                if s.request_id == request_id:
                    return self.running.pop(i)
            for i, s in enumerate(self.waiting):
                if s.request_id == request_id:
                    del self.waiting[i]
                    return s
            return None

    def clear(self) -> None:
        """Empty both queues (engine teardown)."""
        with self._lock:
            self.waiting.clear()
            self.running.clear()
