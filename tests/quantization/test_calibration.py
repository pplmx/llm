"""Tests for the calibration data collector.

``CalibrationDataCollector`` attaches forward hooks to every ``nn.Linear``,
accumulates per-layer activation statistics, and derives the quantization
scales from them. Two properties matter for correctness and neither was
covered before:

1. ``register_hooks`` must be idempotent. It was not: calling it twice attached
   a second hook to each layer, and both hooks append to the SAME
   ``ActivationStats``, so every ``num_samples`` count doubled (measured 2048 ->
   4096). The scales derived from those statistics were therefore wrong, with
   no error raised.
2. ``collect`` accepts several batch shapes (a raw tensor, a tuple/list, or a
   dict with ``input_ids``/``inputs``). Each branch is exercised separately so a
   format change is caught here rather than at quantization time.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm.models.decoder import DecoderModel
from llm.quantization.calibration import CalibrationDataCollector


def _tiny_model() -> DecoderModel:
    model = DecoderModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16)
    model.eval()
    return model


def _tensor_loader(batch_size: int = 8, batches: int = 2) -> DataLoader:
    data = torch.randint(0, 32, (batch_size * batches, 16))
    return DataLoader(TensorDataset(data), batch_size=batch_size)


def _hook_count(model) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def test_register_hooks_is_idempotent():
    """A repeated call must REPLACE the hooks, not stack a second set.

    Stacking doubles every statistic, which silently skews the derived scales —
    the failure mode this guards.
    """
    model = _tiny_model()
    collector = CalibrationDataCollector(model)

    collector.register_hooks()
    after_one = _hook_count(model)
    assert after_one > 0

    collector.register_hooks()
    assert _hook_count(model) == after_one, "register_hooks stacked a second set of hooks"
    collector.remove_hooks()
    assert _hook_count(model) == 0, "remove_hooks left hooks behind"


def test_double_registration_does_not_change_collected_statistics():
    """The end-to-end consequence: doubling must not change the sample counts."""
    loader = _tensor_loader()

    single_model = _tiny_model()
    single = CalibrationDataCollector(single_model)
    single.register_hooks()
    single.collect(loader)
    expected = {name: stats.num_samples for name, stats in single.stats.items()}
    single.remove_hooks()

    double_model = _tiny_model()
    double = CalibrationDataCollector(double_model)
    double.register_hooks()
    double.register_hooks()
    double.collect(loader)
    got = {name: stats.num_samples for name, stats in double.stats.items()}
    double.remove_hooks()

    assert got == expected, "double registration changed the activation statistics"


def test_re_registering_resets_statistics_for_a_fresh_pass():
    """Re-registering starts a clean sweep instead of accumulating across runs."""
    model = _tiny_model()
    loader = _tensor_loader()
    collector = CalibrationDataCollector(model)

    collector.register_hooks()
    collector.collect(loader)
    first_pass = {name: stats.num_samples for name, stats in collector.stats.items()}

    collector.register_hooks()  # a second calibration run on the same collector
    assert all(stats.num_samples == 0 for stats in collector.stats.values()), (
        "re-registering kept the previous pass's counts, so the next collect accumulates instead of replacing"
    )

    collector.collect(loader)
    second_pass = {name: stats.num_samples for name, stats in collector.stats.items()}
    collector.remove_hooks()
    assert second_pass == first_pass


def test_collect_accepts_a_raw_tensor_batch():
    """``collect`` normalizes a bare tensor batch (``input_ids`` directly)."""
    model = _tiny_model()
    collector = CalibrationDataCollector(model)
    collector.register_hooks()
    collector.collect([torch.randint(0, 32, (4, 16)), torch.randint(0, 32, (4, 16))])
    collector.remove_hooks()
    assert collector.stats, "no statistics collected from raw tensors"
    assert all(stats.num_samples > 0 for stats in collector.stats.values())


def test_collect_accepts_dict_and_sequence_batches():
    """A dict batch (``input_ids``) and a tuple/list batch must both work."""
    model = _tiny_model()

    dict_collector = CalibrationDataCollector(model)
    dict_collector.register_hooks()
    dict_collector.collect([{"input_ids": torch.randint(0, 32, (4, 16))}])
    dict_collector.remove_hooks()
    assert dict_collector.stats

    list_collector = CalibrationDataCollector(model)
    list_collector.register_hooks()
    list_collector.collect([(torch.randint(0, 32, (4, 16)),)])
    list_collector.remove_hooks()
    assert list_collector.stats


def test_num_batches_limits_the_sweep():
    """``num_batches`` caps how many batches are consumed.

    ``num_samples`` counts ELEMENTS, not batches (each layer sees a different
    feature count), so the assertion compares two sweep lengths against each
    other instead of guessing an absolute ceiling.
    """
    loader = _tensor_loader(batch_size=4, batches=4)

    def sweep(num_batches: int | None) -> dict[str, int]:
        model = _tiny_model()
        collector = CalibrationDataCollector(model)
        collector.register_hooks()
        collector.collect(loader, num_batches=num_batches)
        collector.remove_hooks()
        return {name: stats.num_samples for name, stats in collector.stats.items()}

    one_batch = sweep(1)
    all_batches = sweep(None)
    assert one_batch, "no statistics collected"
    for name, limited in one_batch.items():
        assert limited < all_batches[name], (
            f"num_batches=1 did not reduce the sweep for {name}: {limited} vs {all_batches[name]}"
        )


def test_collect_defaults_to_the_host_device_not_cuda(monkeypatch):
    """``collect`` must work on a CPU-only host when no device is passed.

    The ``device`` argument defaulted to the literal ``"cuda"``, so the very
    first line of ``collect`` (``self.model.to(device)``) raised
    ``RuntimeError: Found no NVIDIA driver`` on any GPU-less machine — every
    quantization run on a CPU CI runner died before collecting a single batch.
    The default now resolves through ``resolve_device``.
    """
    import llm.utils.common as common

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)

    model = _tiny_model()
    collector = CalibrationDataCollector(model)
    collector.register_hooks()
    collector.collect(_tensor_loader(batch_size=4, batches=1))
    collector.remove_hooks()

    assert collector.stats, "no statistics collected on the default device"
    assert next(iter(collector.stats.values())).num_samples > 0


def test_collect_still_honours_an_explicit_device(monkeypatch):
    """The new default must not override a caller that names a device.

    A resolver that ignored its argument and always picked the host device
    would pass the test above while breaking ``collect(..., device="cpu")``
    on a GPU host.
    """
    import llm.utils.common as common

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)

    collector = CalibrationDataCollector(_tiny_model())
    collector.register_hooks()
    collector.collect(_tensor_loader(batch_size=4, batches=1), device="cpu")
    collector.remove_hooks()

    assert collector.stats
    assert all(stats.num_samples > 0 for stats in collector.stats.values())
