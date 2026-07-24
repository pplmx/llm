"""Tests for :class:`llm.training.core.monitor.PerformanceMonitor`."""

from __future__ import annotations

import pytest
import torch

from llm.training.core.monitor import PerformanceMonitor


class TestPerformanceMonitor:
    def test_init_with_cpu_device(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        assert pm.rank == 0
        assert pm.batch_times == []
        assert pm.losses == []
        assert pm.gradient_norms == []

    def test_log_batch_time(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm.log_batch_time(0.5)
        pm.log_batch_time(1.5)
        assert pm.batch_times == [0.5, 1.5]

    def test_log_loss(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm.log_loss(3.2)
        pm.log_loss(1.8)
        assert pm.losses == [3.2, 1.8]

    def test_log_gradient_norm(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm.log_gradient_norm(0.01)
        pm.log_gradient_norm(0.05)
        assert pm.gradient_norms == [0.01, 0.05]

    def test_get_avg_batch_time_with_data(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm.log_batch_time(1.0)
        pm.log_batch_time(2.0)
        pm.log_batch_time(3.0)
        assert pm.get_avg_batch_time() == pytest.approx(2.0)

    def test_get_avg_batch_time_empty(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        assert pm.get_avg_batch_time() == 0.0

    def test_get_current_gpu_memory_cpu_device(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        allocated, reserved = pm.get_current_gpu_memory()
        assert allocated == 0.0
        assert reserved == 0.0

    def test_get_peak_gpu_memory_cpu_device(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        assert pm.get_peak_gpu_memory() == 0.0

    def test_reset_epoch_stats(self):
        pm = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm.log_batch_time(1.0)
        pm.log_loss(5.0)
        pm.log_gradient_norm(0.1)
        pm.reset_epoch_stats()
        assert pm.batch_times == []
        assert pm.losses == []
        assert pm.gradient_norms == []

    def test_multiple_ranks(self):
        pm0 = PerformanceMonitor(rank=0, device=torch.device("cpu"))
        pm1 = PerformanceMonitor(rank=1, device=torch.device("cpu"))
        pm0.log_loss(4.0)
        pm1.log_loss(6.0)
        assert pm0.losses == [4.0]
        assert pm1.losses == [6.0]
