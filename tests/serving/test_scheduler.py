"""
Tests for Scheduler - request scheduling for continuous batching.
"""

from unittest.mock import MagicMock

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
    """Concurrent access to ``waiting`` / ``running``.

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


class TestSchedulerRepr:
    """The repr must expose queue depth and be safe under concurrency."""

    def test_repr_shows_queue_depths(self):
        """``waiting`` / ``running`` lengths are the core serving-health signal.

        They are what distinguishes "requests piling up" from "queue healthy but
        each step is slow", and the default ``<Scheduler object at 0x...>``
        hides both.
        """
        scheduler = Scheduler(max_batch_size=4, max_waiting=2)
        empty = repr(scheduler)
        assert "object at 0x" not in empty, "Scheduler still uses the default object repr"
        assert "max_batch_size=4" in empty
        assert "waiting=0" in empty
        assert "running=0" in empty

        scheduler.add_sequence(MagicMock())
        assert "waiting=1" in repr(scheduler)

        scheduler.add_sequence(MagicMock())
        full = repr(scheduler)
        assert "waiting=2" in full
        assert scheduler.is_full(), "repr and is_full disagree about a full queue"

    def test_repr_is_safe_while_another_thread_mutates_the_queues(self):
        """``repr`` must take the scheduler lock, like every other reader.

        The deques are mutated by ``step()`` and the streaming paths without the
        engine lock; an unlocked ``len()`` can observe a torn state or raise
        ``RuntimeError: deque mutated during iteration``.
        """
        import threading

        scheduler = Scheduler(max_batch_size=8, max_waiting=64)
        errors: list[BaseException] = []
        stop = threading.Event()

        def churn() -> None:
            try:
                while not stop.is_set():
                    scheduler.add_sequence(MagicMock())
                    scheduler.clear()
            except BaseException as exc:  # noqa: BLE001 - report from the thread
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(2000):
                    text = repr(scheduler)
                    assert text.startswith("Scheduler(")
            except BaseException as exc:  # noqa: BLE001 - report from the thread
                errors.append(exc)

        threads = [threading.Thread(target=churn) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        readers_done = threading.Event()

        def _wait_readers() -> None:
            for thread in threads[2:]:
                thread.join()
            readers_done.set()

        threading.Thread(target=_wait_readers).start()
        readers_done.wait(timeout=30)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors, f"concurrent repr raised: {errors[0]!r}"


class TestSchedulerSlotReuse:
    """A finished sequence must hand its slot to a waiting one.

    This is the point of continuous batching: the batch size is a limit on
    CONCURRENT sequences, not on total work, so a completed request has to free
    capacity within the same step. The existing
    ``test_schedule_removes_finished`` checks the removal but not that the freed
    slot is actually refilled.
    """

    def test_finished_sequence_slot_is_refilled_from_waiting(self):
        scheduler = Scheduler(max_batch_size=2)
        for i in range(4):
            scheduler.add_sequence(make_sequence(f"req{i}"))

        first = scheduler.schedule()
        assert [s.request_id for s in first] == ["req0", "req1"]
        assert len(scheduler.waiting) == 2

        first[0].status = RequestState.FINISHED
        second = scheduler.schedule()

        ids = [s.request_id for s in second]
        assert "req0" not in ids, "the finished sequence must leave the running set"
        assert ids == ["req1", "req2"], f"the freed slot was not refilled: {ids}"
        assert len(scheduler.waiting) == 1

    def test_batch_stays_at_capacity_while_work_remains(self):
        """Under sustained load the batch must stay FULL, not drain.

        A scheduler that only removed finished sequences (without refilling)
        would shrink the batch over time and waste the engine's whole point.
        """
        scheduler = Scheduler(max_batch_size=3)
        for i in range(9):
            scheduler.add_sequence(make_sequence(f"req{i}"))

        for round_index in range(3):
            running = scheduler.schedule()
            assert len(running) == 3, f"round {round_index} ran {len(running)} sequences, expected 3"
            for sequence in running:
                sequence.status = RequestState.FINISHED

        assert not scheduler.waiting, "all nine should have been served"


class TestConflictingRequestIdGuard:
    """Two live sequences must never share a `request_id`.

    `SlotAllocator` keys KV slots by `request_id`, so two distinct requests whose
    client-supplied ids collide would write into the SAME slot: their K/V and
    generated tokens contaminate each other, and a `free()` by either returns the
    other's still-live slot to the pool. `add_sequence_if_not_conflicting` exists
    to prevent exactly that, and had no test at all.
    """

    def test_a_conflicting_duplicate_is_rejected(self):
        scheduler = Scheduler(max_batch_size=4, max_waiting=8)
        first = make_sequence("dup", "first prompt")
        assert scheduler.add_sequence_if_not_conflicting(first) is True

        conflicting = make_sequence("dup", "a completely different prompt")
        assert scheduler.add_sequence_if_not_conflicting(conflicting) is False, (
            "a different request reusing a live request_id was accepted — both would share one KV slot"
        )
        assert len(scheduler.waiting) == 1, "the rejected sequence must not have been enqueued"

    def test_conflict_is_detected_against_running_sequences_too(self):
        """The guard must scan `running`, not only `waiting`.

        A request that already started decoding is the more dangerous case: its
        slot holds real K/V that a second request would overwrite.
        """
        scheduler = Scheduler(max_batch_size=4, max_waiting=8)
        scheduler.add_sequence_if_not_conflicting(make_sequence("dup"))
        scheduler.schedule()

        conflicting = make_sequence("dup", "different")
        assert scheduler.add_sequence_if_not_conflicting(conflicting) is False
        assert len(scheduler.waiting) == 0

    def test_the_same_logical_request_may_be_re_added_via_matches(self):
        """The engine's internal double-add contract must keep working.

        ``generate_request`` re-adds the same logical request so the reap loop can
        remove every copy; the ``matches`` callback is what distinguishes that from
        a genuine collision.
        """
        scheduler = Scheduler(max_batch_size=4, max_waiting=8)
        original = make_sequence("dup", "same prompt")
        assert scheduler.add_sequence_if_not_conflicting(original) is True

        re_added = make_sequence("dup", "same prompt")
        allowed = scheduler.add_sequence_if_not_conflicting(
            re_added,
            matches=lambda other: other.prompt == re_added.prompt and other.input_ids == re_added.input_ids,
        )
        assert allowed is True, "the same logical request was wrongly rejected as a conflict"
        assert len(scheduler.waiting) == 2

    def test_matches_does_not_whitelist_a_different_request(self):
        """A `matches` callback that returns False must still reject."""
        scheduler = Scheduler(max_batch_size=4, max_waiting=8)
        scheduler.add_sequence_if_not_conflicting(make_sequence("dup", "one"))

        other = make_sequence("dup", "two")
        assert scheduler.add_sequence_if_not_conflicting(other, matches=lambda _existing: False) is False
        assert len(scheduler.waiting) == 1

    def test_a_finished_holder_does_not_block_reuse(self):
        """Once the original finishes, its id is free again."""
        scheduler = Scheduler(max_batch_size=4, max_waiting=8)
        finished = make_sequence("dup")
        scheduler.add_sequence_if_not_conflicting(finished)
        scheduler.schedule()
        finished.status = RequestState.FINISHED

        reused = make_sequence("dup", "new request, reused id")
        assert scheduler.add_sequence_if_not_conflicting(reused) is True, (
            "a FINISHED sequence still blocked id reuse — ids would be permanently consumed"
        )


class TestSlotReuseAfterCompletion:
    """A finished sequence must free its batch slot for a waiting one.

    `test_schedule_respects_max_batch_size` only counts how many sequences one
    `schedule()` call returns. The behaviour that makes continuous batching worth
    anything is the NEXT step: when a running sequence finishes, its slot must go to
    the head of the waiting queue instead of leaving a hole until every sequence is
    done. A scheduler that never re-promoted would pass the counting test.
    """

    def test_waiting_sequence_takes_over_a_freed_slot(self):
        scheduler = Scheduler(max_batch_size=1, max_waiting=8)
        first = make_sequence("first")
        scheduler.add_sequence(first)

        assert [seq.request_id for seq in scheduler.schedule()] == ["first"]

        second = make_sequence("second")
        scheduler.add_sequence(second)
        assert len(scheduler.waiting) == 1

        # While the first runs, the full batch means the second waits.
        assert [seq.request_id for seq in scheduler.schedule()] == ["first"]
        assert len(scheduler.waiting) == 1

        # The moment the first finishes, the second must be promoted.
        first.status = RequestState.FINISHED
        scheduled = scheduler.schedule()

        assert [seq.request_id for seq in scheduled] == ["second"], (
            f"freed slot was not reused: scheduled {[seq.request_id for seq in scheduled]}"
        )
        assert len(scheduler.waiting) == 0

    def test_progress_is_not_blocked_by_a_long_running_sequence(self):
        """A queued short request runs alongside a long one while capacity allows.

        With room in the batch there is no reason to hold a waiting sequence back,
        so a scheduler that only ever admitted one sequence at a time would starve
        short requests behind long ones.
        """
        scheduler = Scheduler(max_batch_size=2, max_waiting=8)
        scheduler.add_sequence(make_sequence("long"))
        scheduler.schedule()

        scheduler.add_sequence(make_sequence("short"))
        scheduled = scheduler.schedule()

        assert {seq.request_id for seq in scheduled} == {"long", "short"}, (
            f"a waiting request was held back despite free capacity: {[s.request_id for s in scheduled]}"
        )
