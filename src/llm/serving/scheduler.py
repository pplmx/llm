from collections import deque

from llm.serving.schemas import RequestState, Sequence


class Scheduler:
    """Simple FCFS scheduler for continuous batching with backpressure.

    Limits the ``waiting`` queue to ``max_waiting`` to prevent unbounded
    memory growth under load. When the queue is full, :meth:`add_sequence`
    raises ``RuntimeError`` so callers can apply backpressure (HTTP 503).
    """

    DEFAULT_MAX_WAITING = 1024

    def __init__(self, max_batch_size: int = 16, *, max_waiting: int | None = None):
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting if max_waiting is not None else self.DEFAULT_MAX_WAITING

    @property
    def has_pending_work(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def add_sequence(self, seq: Sequence):
        """Add a new sequence to the waiting queue.

        Raises:
            RuntimeError: If the waiting queue is at capacity.
        """
        if len(self.waiting) >= self.max_waiting:
            raise RuntimeError(
                f"Waiting queue full ({len(self.waiting)}/{self.max_waiting}); "
                "retry later or increase max_waiting."
            )
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """
        Schedule sequences for the next inference step.
        Promotes waiting sequences to running if there is capacity.
        """
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
        for s in self.running:
            if s.request_id == request_id:
                return s
        for s in self.waiting:
            if s.request_id == request_id:
                return s
        return None
