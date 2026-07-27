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
    assert devices.cuda_device_strings() == [str(device) for device in devices.all_gpu_devices()]


def test_default_device_is_usable_or_cpu():
    gpu_devices = devices.all_gpu_devices()

    if gpu_devices:
        assert devices.DEFAULT_DEVICE in gpu_devices
        assert devices.cuda_usable(devices.DEFAULT_DEVICE)
    else:
        assert torch.device("cpu") == devices.DEFAULT_DEVICE


def test_cached_device_lists_match_dynamic_inventory():
    assert devices.cuda_device_strings() == devices.ALL_GPU_DEVICES
    assert [str(device) for device in devices.all_devices()] == devices.ALL_DEVICES
