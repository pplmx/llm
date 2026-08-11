"""Behavioral tests for shared GPU device selection helpers."""

import torch

from tests.support import devices


def test_cuda_device_inventory_is_consistent():
    gpu_devices = devices.all_gpu_devices()

    assert devices.cuda_device_count() == len(gpu_devices)
    assert devices.cuda_usable() is bool(gpu_devices)
    assert len(gpu_devices) == len(set(gpu_devices))
    assert all(device.type == "cuda" for device in gpu_devices)
    assert all(devices.cuda_usable(device) for device in gpu_devices)


def test_all_devices_falls_back_to_cpu():
    gpu_devices = devices.all_gpu_devices()

    if gpu_devices:
        assert devices.all_devices() == gpu_devices
    else:
        assert devices.all_devices() == [torch.device("cpu")]


def test_device_strings_match_device_objects():
    # Call both in the same order and compare immediately to avoid a
    # TOCTOU where GPU free-VRAM shifts between the two calls,
    # reordering the VRAM-sorted lists.
    strings = devices.cuda_device_strings()
    objects = devices.all_gpu_devices()
    assert strings == [str(device) for device in objects]


def test_default_device_is_usable_or_cpu():
    gpu_devices = devices.all_gpu_devices()

    if gpu_devices:
        assert devices.DEFAULT_DEVICE in gpu_devices
        assert devices.cuda_usable(devices.DEFAULT_DEVICE)
    else:
        assert torch.device("cpu") == devices.DEFAULT_DEVICE


def test_cached_device_lists_match_dynamic_inventory():
    """Cached device lists must only reference currently-visible devices.

    ``ALL_GPU_DEVICES`` / ``ALL_DEVICES`` are deliberately cached at import
    time: pytest parametrize decorators must see a fixed list when modules
    are collected, long before any test executes.  The reusable GPU set is
    filtered by *free VRAM*, which fluctuates on shared GPU hosts — other
    suite runs or processes can push a GPU below the ``MIN_FREE_VRAM_BYTES``
    floor at import and release it by test time (or vice versa).  So the
    cached snapshot is not guaranteed equal to the live, VRAM-filtered view.

    What *is* a stable invariant is that the cache never references a device
    that is not currently visible to torch: the visible device count only
    changes with ``CUDA_VISIBLE_DEVICES``, which the test process does not
    mutate.  A phantom entry (e.g. a stale index after a renumber) is a real
    defect we must catch.
    """
    visible = {f"cuda:{index}" for index in range(torch.cuda.device_count())}
    assert set(devices.ALL_GPU_DEVICES) <= visible
    # ALL_DEVICES is exactly ALL_GPU_DEVICES, or the documented CPU fallback.
    if devices.ALL_GPU_DEVICES:
        assert devices.ALL_DEVICES == devices.ALL_GPU_DEVICES
    else:
        assert devices.ALL_DEVICES == ["cpu"]
    # Dynamic views also only reference visible devices.
    assert set(devices.cuda_device_strings()) <= visible
    assert {str(device) for device in devices.all_devices()} <= visible


def test_min_free_vram_threshold_is_engine_aligned():
    """The test VRAM floor must match the engine's _cuda_usable guard."""
    from llm.training.core.engine import _MIN_FREE_VRAM_BYTES as ENGINE_MIN_FREE_VRAM

    assert devices.MIN_FREE_VRAM_BYTES == ENGINE_MIN_FREE_VRAM


def _fake_cuda(monkeypatch, device_count, free_bytes_map):
    """Stub CUDA so tests can reason about VRAM thresholds without a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def _mem_get_info(index):
        return free_bytes_map.get(index, 0), 0

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)
    # _free_memory lives in devices but delegates to torch.cuda.mem_get_info.
    monkeypatch.setattr(devices, "_free_memory", lambda index: _mem_get_info(index)[0])


def test_threshold_excludes_low_vram_devices(monkeypatch):
    # device 0: 256 MiB free (< 512 MiB floor), device 1: 1 GiB free (>= floor)
    _fake_cuda(monkeypatch, device_count=2, free_bytes_map={0: 256 << 20, 1: 1 << 30})

    assert devices.cuda_device_count() == 1
    assert devices.all_gpu_devices() == [torch.device("cuda:1")]
    assert devices.cuda_usable() is True
    assert devices.cuda_usable(torch.device("cuda:0")) is False
    assert devices.cuda_usable(torch.device("cuda:1")) is True


def test_threshold_defaults_to_cpu_when_all_low_vram(monkeypatch):
    _fake_cuda(monkeypatch, device_count=2, free_bytes_map={0: 100 << 20, 1: 200 << 20})

    assert devices.cuda_device_count() == 0
    assert devices.all_gpu_devices() == []
    assert devices.cuda_usable() is False
    assert devices._default_device() == torch.device("cpu")


def test_default_device_prefers_most_free_vram(monkeypatch):
    _fake_cuda(monkeypatch, device_count=3, free_bytes_map={0: 1 << 30, 1: 4 << 30, 2: 2 << 30})

    # device 1 has the most free VRAM -> picked as DEFAULT_DEVICE.
    assert devices._default_device() == torch.device("cuda:1")


def test_default_device_tie_prefers_lower_index(monkeypatch):
    _fake_cuda(monkeypatch, device_count=2, free_bytes_map={0: 2 << 30, 1: 2 << 30})

    assert devices._default_device() == torch.device("cuda:0")


def test_multi_gpu_devices_sorted_by_vram_desc(monkeypatch):
    """all_gpu_devices() returns devices sorted by free VRAM descending."""
    _fake_cuda(
        monkeypatch,
        device_count=3,
        free_bytes_map={0: 1 << 30, 1: 4 << 30, 2: 2 << 30},
    )

    result = devices.all_gpu_devices()
    assert result == [torch.device("cuda:1"), torch.device("cuda:2"), torch.device("cuda:0")]


def test_multi_gpu_devices_alias_matches_all_gpu_devices(monkeypatch):
    """multi_gpu_devices() is a named alias for all_gpu_devices()."""
    _fake_cuda(monkeypatch, device_count=2, free_bytes_map={0: 1 << 30, 1: 2 << 30})

    assert devices.multi_gpu_devices() == devices.all_gpu_devices()


def test_select_gpus_by_free_vram_limits_count(monkeypatch):
    """select_gpus_by_free_vram(n) returns at most n devices, fattest first."""
    _fake_cuda(
        monkeypatch,
        device_count=4,
        free_bytes_map={0: 2 << 30, 1: 4 << 30, 2: 3 << 30, 3: 1 << 30},
    )

    top2 = devices.select_gpus_by_free_vram(max_gpus=2)
    assert top2 == [torch.device("cuda:1"), torch.device("cuda:2")]
    assert len(top2) == 2

    # None -> all devices.
    all_devs = devices.select_gpus_by_free_vram()
    assert all_devs == devices.all_gpu_devices()


def test_gpu_vram_info_returns_sorted_pairs(monkeypatch):
    """gpu_vram_info() returns (index, free_bytes) pairs sorted by VRAM desc."""
    _fake_cuda(
        monkeypatch,
        device_count=3,
        free_bytes_map={0: 1 << 30, 1: 4 << 30, 2: 2 << 30},
    )

    info = devices.gpu_vram_info()
    assert info == [(1, 4 << 30), (2, 2 << 30), (0, 1 << 30)]


def test_multi_device_fixture_uses_dynamic_lookup(monkeypatch):
    """The multi_device fixture calls the dynamic function, not a cached snapshot."""
    _fake_cuda(monkeypatch, device_count=2, free_bytes_map={0: 2 << 30, 1: 3 << 30})

    # Simulate VRAM dropping between import and test time.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx: (0, 0) if idx == 1 else (2 << 30, 0))
    monkeypatch.setattr(devices, "_free_memory", lambda idx: 0 if idx == 1 else (2 << 30))

    # Dynamic call should reflect the drop; only cuda:0 remains usable.
    dynamic = devices.all_gpu_devices()
    assert dynamic == [torch.device("cuda:0")]
