"""Performance monitoring helpers for the training engine."""

from __future__ import annotations

import torch


class PerformanceMonitor:
    """Performance monitor for tracking training metrics."""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device
        self.batch_times: list[float] = []
        self.losses: list[float] = []
        self.gradient_norms: list[float] = []

    def log_batch_time(self, time_taken: float):
        self.batch_times.append(time_taken)

    def log_loss(self, loss: float):
        self.losses.append(loss)

    def log_gradient_norm(self, grad_norm: float):
        self.gradient_norms.append(grad_norm)

    def get_avg_batch_time(self) -> float:
        return sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0.0

    def get_current_gpu_memory(self) -> tuple[float, float]:
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return memory_allocated, memory_reserved
        return 0.0, 0.0

    # Track peak GPU memory usage
    def get_peak_gpu_memory(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024**3
        return 0.0

    def reset_epoch_stats(self):
        self.batch_times.clear()
        self.losses.clear()
        self.gradient_norms.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
