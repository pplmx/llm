"""
Priority Scheduler for request management.

Implements multi-level priority queues for request scheduling.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from llm.serving.schemas import RequestState, Sequence


class Priority(IntEnum):
    """Request priority levels."""

    CRITICAL = 0  # Highest priority (e.g., system prompts)
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority (e.g., batch jobs)


@dataclass(order=True)
class PrioritizedSequence:
    """Wrapper for priority queue ordering."""

    priority: int
    timestamp: float = field(compare=True)
    sequence: Sequence = field(compare=False)


class PriorityScheduler:
    """
    Multi-level priority scheduler for inference requests.

    Features:
    - Priority-based scheduling (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
    - Aging mechanism to prevent starvation
    - Preemption support for high-priority requests
    - Fair scheduling within same priority level
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        enable_aging: bool = True,
        aging_interval_sec: float = 5.0,
        aging_boost: int = 1,
        enable_preemption: bool = False,
    ):
        """
        Initialize priority scheduler.

        Args:
            max_batch_size: Maximum concurrent sequences.
            enable_aging: Whether to boost priority of waiting requests.
            aging_interval_sec: Time before priority boost.
            aging_boost: Priority levels to boost per interval.
            enable_preemption: Whether high priority can preempt running.
        """
        self.max_batch_size = max_batch_size
        self.enable_aging = enable_aging
        self.aging_interval_sec = aging_interval_sec
        self.aging_boost = aging_boost
        self.enable_preemption = enable_preemption

        # Priority queues (one per level)
        self.queues: dict[Priority, deque[Sequence]] = {p: deque() for p in Priority}

        # Running sequences
        self.running: list[Sequence] = []

        # Tracking for aging
        self.arrival_times: dict[str, float] = {}  # request_id -> arrival time
        self.current_priorities: dict[str, Priority] = {}  # request_id -> current priority

        # Stats
        self.total_scheduled = 0
        self.total_preemptions = 0

    @property
    def has_pending_work(self) -> bool:
        """Check if there's any work to do."""
        return len(self.running) > 0 or any(len(q) > 0 for q in self.queues.values())

    @property
    def num_waiting(self) -> int:
        """Total number of waiting sequences."""
        return sum(len(q) for q in self.queues.values())

    def add_sequence(
        self,
        seq: Sequence,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """
        Add a sequence to the appropriate priority queue.

        Args:
            seq: Sequence to add.
            priority: Initial priority level.
        """
        self.queues[priority].append(seq)
        self.arrival_times[seq.request_id] = time.time()
        self.current_priorities[seq.request_id] = priority

    def schedule(self) -> list[Sequence]:
        """
        Schedule sequences for the next inference step.

        Returns:
            List of sequences to process.
        """
        # Apply aging if enabled
        if self.enable_aging:
            self._apply_aging()

        # Clean up finished sequences
        self.running = [s for s in self.running if not s.is_finished()]

        # Calculate available slots
        available_slots = self.max_batch_size - len(self.running)

        if available_slots <= 0:
            return self.running

        # Fill slots from highest to lowest priority
        for priority in Priority:
            while self.queues[priority] and available_slots > 0:
                seq = self.queues[priority].popleft()
                seq.status = RequestState.RUNNING
                self.running.append(seq)
                self.total_scheduled += 1
                available_slots -= 1

                # Clean up tracking
                self._cleanup_tracking(seq.request_id)

        return self.running

    def _apply_aging(self) -> None:
        """Boost priority of requests that have been waiting too long."""
        current_time = time.time()

        for priority in list(Priority)[1:]:  # Skip CRITICAL (can't go higher)
            aged_sequences = []

            for seq in list(self.queues[priority]):
                arrival = self.arrival_times.get(seq.request_id, current_time)
                wait_time = current_time - arrival

                if wait_time >= self.aging_interval_sec:
                    aged_sequences.append(seq)

            # Boost aged sequences
            for seq in aged_sequences:
                self.queues[priority].remove(seq)
                new_priority = Priority(max(0, priority - self.aging_boost))
                self.queues[new_priority].append(seq)
                self.current_priorities[seq.request_id] = new_priority

    def preempt_for_priority(self, priority: Priority) -> list[Sequence] | None:
        """
        Attempt to preempt lower-priority running sequences.

        Args:
            priority: Priority level requesting preemption.

        Returns:
            List of preempted sequences, or None if preemption failed.
        """
        if not self.enable_preemption:
            return None

        # Find running sequences with lower priority
        preemptable = []
        for seq in self.running:
            seq_priority = self.current_priorities.get(seq.request_id, Priority.NORMAL)
            if seq_priority > priority:  # Lower priority number = higher priority
                preemptable.append((seq_priority, seq))

        if not preemptable:
            return None

        # Preempt lowest priority first
        preemptable.sort(key=lambda x: x[0], reverse=True)
        preempted_seq = preemptable[0][1]

        # Move back to waiting queue
        self.running.remove(preempted_seq)
        preempted_seq.status = RequestState.PENDING
        original_priority = self.current_priorities.get(preempted_seq.request_id, Priority.NORMAL)
        self.queues[original_priority].appendleft(preempted_seq)

        self.total_preemptions += 1

        return [preempted_seq]

    def get_sequence(self, request_id: str) -> Sequence | None:
        """Find a sequence by its request_id."""
        for s in self.running:
            if s.request_id == request_id:
                return s

        for queue in self.queues.values():
            for s in queue:
                if s.request_id == request_id:
                    return s

        return None

    def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics about queue states."""
        return {
            "running": len(self.running),
            "waiting": self.num_waiting,
            "queues": {p.name: len(q) for p, q in self.queues.items()},
            "total_scheduled": self.total_scheduled,
            "total_preemptions": self.total_preemptions,
        }

    def _cleanup_tracking(self, request_id: str) -> None:
        """Clean up tracking data for a request."""
        self.arrival_times.pop(request_id, None)
        self.current_priorities.pop(request_id, None)

    def clear(self) -> None:
        """Clear all queues and running sequences."""
        for queue in self.queues.values():
            queue.clear()
        self.running.clear()
        self.arrival_times.clear()
        self.current_priorities.clear()
