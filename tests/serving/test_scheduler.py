"""
Tests for Scheduler - request scheduling for continuous batching.
"""

from llm.serving.scheduler import Scheduler
from llm.serving.schemas import RequestState, Sequence


def make_sequence(request_id: str, prompt: str = "test prompt") -> Sequence:
    """Helper to create a Sequence."""
    return Sequence(
        request_id=request_id,
        prompt=prompt,
        input_ids=[1, 2, 3],
    )


class TestScheduler:
    """Tests for Scheduler class."""

    def test_init_default(self):
        """Test scheduler initialization with defaults."""
        scheduler = Scheduler()
        assert scheduler.max_batch_size == 16
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0

    def test_init_custom(self):
        """Test scheduler initialization with custom max_batch_size."""
        scheduler = Scheduler(max_batch_size=8)
        assert scheduler.max_batch_size == 8

    def test_has_pending_work_empty(self):
        """Test has_pending_work returns False when empty."""
        scheduler = Scheduler()
        assert not scheduler.has_pending_work

    def test_has_pending_work_waiting(self):
        """Test has_pending_work returns True when waiting queue has sequences."""
        scheduler = Scheduler()
        scheduler.add_sequence(make_sequence("req1"))
        assert scheduler.has_pending_work

    def test_has_pending_work_running(self):
        """Test has_pending_work returns True when running queue has sequences."""
        scheduler = Scheduler()
        seq = make_sequence("req1")
        scheduler.add_sequence(seq)
        scheduler.schedule()  # Move to running
        assert scheduler.has_pending_work

    def test_add_sequence(self):
        """Test adding a sequence to waiting queue."""
        scheduler = Scheduler()
        scheduler.add_sequence(make_sequence("req1"))

        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0].request_id == "req1"

    def test_schedule_moves_to_running(self):
        """Test that schedule moves waiting sequences to running."""
        scheduler = Scheduler(max_batch_size=4)
        scheduler.add_sequence(make_sequence("req1"))

        running = scheduler.schedule()

        assert len(running) == 1
        assert running[0].request_id == "req1"
        assert running[0].status == RequestState.RUNNING

    def test_schedule_respects_max_batch_size(self):
        """Test that schedule respects max_batch_size limit."""
        scheduler = Scheduler(max_batch_size=2)

        for i in range(5):
            scheduler.add_sequence(make_sequence(f"req{i}"))

        running = scheduler.schedule()

        assert len(running) == 2
        assert len(scheduler.waiting) == 3

    def test_schedule_multiple_calls(self):
        """Test that multiple schedule calls drain waiting queue."""
        scheduler = Scheduler(max_batch_size=2)

        for i in range(3):
            scheduler.add_sequence(make_sequence(f"req{i}"))

        # First batch - schedule 2
        running1 = scheduler.schedule()
        assert len(running1) == 2
        assert len(scheduler.waiting) == 1

        # Complete first batch to free slots
        for seq in running1:
            seq.status = RequestState.FINISHED

        # Second batch - schedule remaining 1
        running2 = scheduler.schedule()
        assert len(running2) == 1  # Only the waiting one gets scheduled
        assert len(scheduler.waiting) == 0

        # Third batch - nothing left
        running3 = scheduler.schedule()
        assert len(running3) == 1  # Still the same one running

    def test_get_sequence_in_running(self):
        """Test finding a sequence in running queue."""
        scheduler = Scheduler()
        seq = make_sequence("req1")
        scheduler.add_sequence(seq)
        scheduler.schedule()

        found = scheduler.get_sequence("req1")
        assert found.request_id == "req1"

    def test_get_sequence_in_waiting(self):
        """Test finding a sequence in waiting queue."""
        scheduler = Scheduler(max_batch_size=1)
        scheduler.add_sequence(make_sequence("req1"))
        # Don't schedule, keep in waiting

        found = scheduler.get_sequence("req1")
        assert found.request_id == "req1"

    def test_get_sequence_not_found(self):
        """Test that get_sequence returns None for unknown request."""
        scheduler = Scheduler()

        found = scheduler.get_sequence("nonexistent")
        assert found is None

    def test_schedule_removes_finished(self):
        """Test that schedule removes finished sequences from running."""
        scheduler = Scheduler()

        seq1 = make_sequence("req1")
        seq2 = make_sequence("req2")

        scheduler.add_sequence(seq1)
        scheduler.add_sequence(seq2)
        scheduler.schedule()

        # Mark seq1 as finished
        seq1.status = RequestState.FINISHED

        # Schedule again - should remove finished
        running = scheduler.schedule()

        assert len(running) == 1
        assert running[0].request_id == "req2"

    def test_fifo_order(self):
        """Test that sequences are scheduled in FIFO order."""
        scheduler = Scheduler(max_batch_size=10)

        for i in range(5):
            scheduler.add_sequence(make_sequence(f"req{i}"))

        running = scheduler.schedule()

        # Should maintain order
        for i, seq in enumerate(running):
            assert seq.request_id == f"req{i}"


class TestSchedulerConcurrency:
    """Concurrent access to ``waiting`` / ``running`` (RIL ISS-069).

    ``Scheduler.get_sequence`` runs on the streaming/caller thread while
    ``Scheduler.schedule`` (from an engine ``step()``) mutates the same
    deques. Before the scheduler owned a lock, a ``get_sequence``
    iteration racing a ``schedule`` ``popleft`` raised ``RuntimeError:
    deque mutated during iteration`` under load.  These tests hammer the
    two methods from many threads and assert no exception escapes and the
    invariants hold.
    """

    def test_get_sequence_holds_scheduler_lock(self):
        """The read path must be serialised against ``schedule`` / churn.

        Deterministic proof the lock is wired into ``get_sequence`` (not just
        present on the object): with the scheduler's lock held by the test
        thread, a concurrent ``get_sequence`` in another thread must block
        until the lock is released.
        """
        import threading

        scheduler = Scheduler()
        scheduler.add_sequence(make_sequence("req1"))
        scheduler.schedule()

        entered = threading.Event()
        proceed = threading.Event()
        result: list[Sequence | None] = [None]

        def reader():
            entered.set()
            # This must block while the test thread holds scheduler._lock.
            result[0] = scheduler.get_sequence("req1")
            proceed.set()

        # Test thread owns the scheduler lock.
        with scheduler._lock:
            t = threading.Thread(target=reader)
            t.start()
            # Give the reader a chance to reach the lock acquire.
            entered.wait(1)
            # It must NOT have proceeded while the lock is held.
            assert not proceed.wait(0.2), "get_sequence ran despite scheduler lock held"
        # Lock released → reader proceeds.
        assert proceed.wait(1), "get_sequence deadlocked after lock release"
        t.join()
        assert result[0] is not None, "get_sequence returned nothing after lock release"
        assert result[0].request_id == "req1"

    def test_concurrent_get_sequence_and_schedule_no_crash(self):
        import threading

        scheduler = Scheduler(max_batch_size=4)

        errors: list[BaseException] = []

        def seeder():
            try:
                for i in range(200):
                    scheduler.add_sequence(make_sequence(f"req{i % 8}"))
                    scheduler.schedule()
            except BaseException as exc:  # noqa: BLE001 - report from thread
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    scheduler.get_sequence("req3")
                    _ = scheduler.has_pending_work
            except BaseException as exc:  # noqa: BLE001 - report from thread
                errors.append(exc)

        threads = [threading.Thread(target=seeder) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent scheduler access raised: {errors[0]!r}"

    def test_concurrent_add_and_get_sequence_no_crash(self):
        import threading

        scheduler = Scheduler(max_batch_size=2, max_waiting=64)
        errors: list[BaseException] = []

        def churner():
            try:
                # Add → schedule (popleft) churn races append and pop
                # mutations against the readers' iteration.  Periodic
                # ``clear()`` (teardown path) keeps the queue from hitting
                # the max_waiting backpressure cap — otherwise the expected
                # ``Waiting queue full`` RuntimeError would drown out the
                # data-race signal we're actually probing for.
                for i in range(1000):
                    scheduler.add_sequence(make_sequence(f"c{i}"))
                    scheduler.schedule()
                    if i % 32 == 0:
                        scheduler.clear()
            except BaseException as exc:  # noqa: BLE001 - report from thread
                errors.append(exc)

        def getter():
            try:
                for _ in range(1000):
                    scheduler.get_sequence("missing")
            except BaseException as exc:  # noqa: BLE001 - report from thread
                errors.append(exc)

        threads = [threading.Thread(target=churner) for _ in range(3)] + [
            threading.Thread(target=getter) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent add/schedule/get raised: {errors[0]!r}"
