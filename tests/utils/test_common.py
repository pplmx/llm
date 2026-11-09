import pytest
import torch
import torch.nn as nn

from llm.utils.common import get_activation_layer, resolve_device


@pytest.mark.quick
def test_get_activation_layer_supported_activations():
    """Test that get_activation_layer returns the correct activation class for supported names."""
    # Test cases: (activation_name, expected_class)
    test_cases = [
        ("gelu", nn.GELU),
        ("GELU", nn.GELU),  # Test case-insensitivity
        ("relu", nn.ReLU),
        ("ReLU", nn.ReLU),
        ("silu", nn.SiLU),
        ("SiLU", nn.SiLU),
        ("swish", nn.SiLU),  # swish is an alias for SiLU
        ("SWISH", nn.SiLU),
        ("glu", nn.GLU),  # GLU returns a lambda that instantiates GLU
        ("GLU", nn.GLU),
        ("leaky_relu", nn.LeakyReLU),
        ("LEAKY_RELU", nn.LeakyReLU),
    ]

    for name, expected_class in test_cases:
        activation_class = get_activation_layer(name)
        # For GLU, the function returns a lambda that creates an instance.
        # We need to check if the returned callable creates an instance of the expected class.
        if expected_class == nn.GLU:
            # Call the lambda to get an instance, then check its type
            instance = activation_class()  # GLU requires dim, but for type check, we can pass a dummy
            assert isinstance(instance, expected_class)
        else:
            assert activation_class == expected_class


@pytest.mark.quick
def test_get_activation_layer_unsupported_activation():
    """Test that get_activation_layer raises ValueError for unsupported activation names."""
    with pytest.raises(
        ValueError, match="Unsupported activation function: unknown_activation"
    ):  # Updated error message
        get_activation_layer("unknown_activation")


@pytest.mark.quick
def test_get_activation_layer_case_insensitivity():
    """Test that get_activation_layer is case-insensitive."""
    assert get_activation_layer("GeLu") == nn.GELU
    assert get_activation_layer("ReLU") == nn.ReLU
    assert get_activation_layer("SiLu") == nn.SiLU
    assert get_activation_layer("SwIsH") == nn.SiLU
    assert get_activation_layer("GlU") == nn.GLU
    assert get_activation_layer("LeAkY_ReLu") == nn.LeakyReLU


# --- resolve_device ---------------------------------------------------------
#
# Three components (CalibrationDataCollector.collect, PagedKVCache.__init__,
# ContinuousBatchingEngine.__init__) defaulted their ``device`` argument to the
# literal "cuda". On a CPU-only host that raised
# ``RuntimeError: Found no NVIDIA driver`` while merely CONSTRUCTING them —
# before a caller had any chance to pass an explicit device — so every
# GPU-less CI runner failed. These tests pin the resolver those defaults now
# use; the component-level regressions live beside each component.


@pytest.mark.quick
def test_resolve_device_auto_follows_the_host(monkeypatch):
    """``"auto"`` means "CUDA when present, otherwise CPU" — not "always CUDA"."""
    import llm.utils.common as common

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    assert common.resolve_device("auto") == torch.device("cpu")

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    assert common.resolve_device("auto") == torch.device("cuda")


@pytest.mark.quick
def test_resolve_device_none_is_the_same_as_auto(monkeypatch):
    """``None`` must not fall through to ``torch.device(None)``.

    ``None`` is the default at every call site, so treating it as a literal
    device spec would raise ``TypeError`` on every construction instead of
    selecting the host's device.
    """
    import llm.utils.common as common

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    assert common.resolve_device(None) == common.resolve_device("auto")


@pytest.mark.quick
def test_resolve_device_passes_explicit_devices_through(monkeypatch):
    """An explicit device is honoured even when CUDA is available."""
    import llm.utils.common as common

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    assert common.resolve_device("cpu") == torch.device("cpu")
    assert common.resolve_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.quick
def test_resolve_device_returns_the_same_torch_device_object():
    """An already-built ``torch.device`` is returned as-is, not re-wrapped."""
    device = torch.device("cpu")
    assert resolve_device(device) is device
