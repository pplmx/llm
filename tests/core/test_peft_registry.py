"""Tests for the unified PEFT registry (T2 PEFT #43).

Covers:

- Built-in methods registered (7 methods: lora, qlora, adalora, prefix_tuning,
  ia3, bitfit, adapter)
- ``apply_peft`` dispatches to the right method with kwargs forwarded
- ``get_peft_parameters`` / ``count_peft_parameters`` / ``merge_peft`` /
  ``unmerge_peft`` / ``disable_peft`` / ``enable_peft`` raise when the
  method doesn't expose that helper
- Unknown method raises at lookup time
- ``PEFT_REGISTRY.names()`` is sorted and contains the built-ins
- ``ensure_methods_registered`` is idempotent
- Plugin entry-point loading is plumbed (the actual entry-point load is
  exercised by the ``onnx`` / ``torchscript`` precedent — the smoke here
  just confirms the group name is wired)
- ``apply_peft`` matches the per-method ``apply_*`` function byte-for-byte
  (via reference comparison)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.adalora import apply_adalora
from llm.core.adapter import apply_adapter
from llm.core.bitfit import apply_bitfit
from llm.core.ia3 import apply_ia3
from llm.core.lora import apply_lora
from llm.core.peft import (
    PEFT_REGISTRY,
    PEFTMethod,
    apply_peft,
    count_peft_parameters,
    disable_peft,
    enable_peft,
    ensure_methods_registered,
    get_peft_parameters,
    merge_peft,
    unmerge_peft,
)
from llm.core.peft import registry as _peft_registry_module  # for flag introspection
from llm.core.peft.types import TargetModuleFilter
from llm.core.prefix_tuning import apply_prefix_tuning
from llm.core.qlora import apply_qlora

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Minimal MLP for PEFT dispatch tests.

    Two ``nn.Linear`` layers + one ``nn.LayerNorm`` so we can exercise
    per-method ``target_modules`` filtering without the cost of building
    a real DecoderModel.
    """

    def __init__(self, hidden: int = 16, intermediate: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.act(self.fc1(x))))


@pytest.fixture
def model() -> _TinyMLP:
    return _TinyMLP()


@pytest.fixture(autouse=True)
def _ensure_registered() -> None:
    """Make sure built-ins are registered before every test."""
    ensure_methods_registered()


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_is_a_registry_instance():
    from llm.runtime.registry import Registry

    assert isinstance(PEFT_REGISTRY, Registry)


def test_registry_holds_peftmethod_dataclass():
    # All entries must be ``PEFTMethod`` instances (not raw callables).
    for name in PEFT_REGISTRY.names():
        entry = PEFT_REGISTRY.get(name)
        assert isinstance(entry, PEFTMethod)
        assert entry.name == name


def test_all_seven_builtins_registered():
    names = PEFT_REGISTRY.names()
    for expected in ("lora", "qlora", "adalora", "prefix_tuning", "ia3", "bitfit", "adapter"):
        assert expected in names, f"Built-in PEFT method '{expected}' missing from registry"


def test_registry_names_are_sorted():
    assert PEFT_REGISTRY.names() == sorted(PEFT_REGISTRY.names())


def test_unknown_method_raises_with_helpful_message():
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        PEFT_REGISTRY.get("not_a_real_method")


# ---------------------------------------------------------------------------
# apply_peft dispatch
# ---------------------------------------------------------------------------


def test_apply_peft_lora_matches_apply_lora(model: _TinyMLP) -> None:
    """The dispatched call must reach the same per-method function."""
    apply_peft(model, "lora", rank=4, alpha=8.0, target_modules=["fc1"])
    # fc1 was wrapped -> LoRALinear; fc2 was not -> still nn.Linear.
    from llm.core.lora import LoRALinear

    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, nn.Linear)


def test_apply_peft_ia3_matches_apply_ia3(model: _TinyMLP) -> None:
    apply_peft(model, "ia3", init_scale=0.5, target_modules=["fc1"])
    from llm.core.ia3 import IA3Linear

    assert isinstance(model.fc1, IA3Linear)


def test_apply_peft_bitfit_freezes_weights_enables_biases(model: _TinyMLP) -> None:
    apply_peft(model, "bitfit")
    # Every weight frozen
    for p in model.parameters():
        if p.ndim >= 2:
            assert not p.requires_grad
    # fc1.bias is enabled (fc2 has bias=False, norm.bias is enabled)
    assert model.fc1.bias.requires_grad
    assert model.norm.bias.requires_grad


def test_apply_peft_adapter_wraps_with_bottleneck(model: _TinyMLP) -> None:
    apply_peft(model, "adapter", bottleneck_dim=4, target_modules=["fc1"])
    from llm.core.adapter import AdapterLinear

    assert isinstance(model.fc1, AdapterLinear)
    assert model.fc1.bottleneck_dim == 4


def test_apply_peft_adalora_wraps_adalora(model: _TinyMLP) -> None:
    apply_peft(model, "adalora", init_rank=4, target_rank=2, target_modules=["fc1"])
    from llm.core.adalora import AdaLoRALinear

    assert isinstance(model.fc1, AdaLoRALinear)


def test_apply_peft_prefix_tuning_is_noop_on_non_mha(model: _TinyMLP) -> None:
    """Prefix Tuning only wraps ``MultiHeadAttention`` — on a pure-MLP
    model it returns silently (the per-method apply walks
    ``model.named_modules()`` and finds no matches).

    The loud failure is inside :class:`PrefixTuningAttention.__init__`
    when a non-MHA module is passed directly — that's covered by
    tests/core/test_prefix_tuning.py.
    """
    out = apply_peft(model, "prefix_tuning", prefix_len=4)
    assert out is model
    # No wrapper was installed.
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc2, nn.Linear)


def test_apply_peft_raises_when_model_has_nothing_to_adapt() -> None:
    """A model with no nn.Linear / attention must not report a successful adapt.

    Quantization is the realistic trigger: ``quantize_model_gptq`` replaces
    every ``nn.Linear`` with ``GPTQQuantizedLinear``, so a later
    ``apply_peft(model, "lora")`` found nothing to wrap, returned the model
    untouched, and reported success. The caller then saved an "adapter" (or
    served a model) believing the wrapper was applied.
    """

    class _EmbeddingOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(8, 8)

        def forward(self, x):
            return self.emb(x)

    with pytest.raises(ValueError, match="changed nothing"):
        apply_peft(_EmbeddingOnly(), "lora", rank=4, alpha=8.0)


def test_apply_peft_guard_names_the_quantization_trigger() -> None:
    """The message must explain WHY, since quantized layers look like linears."""

    class _EmbeddingOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(8, 8)

    with pytest.raises(ValueError, match="adapt before quantizing"):
        apply_peft(_EmbeddingOnly(), "lora", rank=4, alpha=8.0)


def test_apply_peft_still_adapts_a_normal_decoder() -> None:
    """The guard must not fire on the ordinary path."""
    from llm.models.decoder import DecoderModel

    model = DecoderModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16)
    before = sum(1 for _ in model.parameters())
    apply_peft(model, "lora", rank=4, alpha=8.0)
    assert sum(1 for _ in model.parameters()) > before


def test_apply_peft_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        apply_peft(model, "definitely_not_a_method")


def test_apply_peft_returns_model(model: _TinyMLP) -> None:
    """``apply_peft`` should be chainable — returns the same model."""
    out = apply_peft(model, "lora", rank=4, alpha=8.0)
    assert out is model


# ---------------------------------------------------------------------------
# apply_* function identity — registry stores the real callable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lora", apply_lora),
        ("qlora", apply_qlora),
        ("adalora", apply_adalora),
        ("prefix_tuning", apply_prefix_tuning),
        ("ia3", apply_ia3),
        ("bitfit", apply_bitfit),
        ("adapter", apply_adapter),
    ],
)
def test_registry_apply_is_per_module_function(name: str, expected) -> None:
    """Each entry's ``apply`` is the actual module-level function."""
    assert PEFT_REGISTRY.get(name).apply is expected


# ---------------------------------------------------------------------------
# get_peft_parameters / count_peft_parameters dispatch
# ---------------------------------------------------------------------------


def test_get_peft_parameters_lora(model: _TinyMLP) -> None:
    apply_peft(model, "lora", rank=4, alpha=8.0, target_modules=["fc1"])
    params = list(get_peft_parameters(model, "lora"))
    # fc1 (16 in, 32 out, rank=4) wrapped → LoRA A=(16,4)=64, LoRA B=(4,32)=128 → 192.
    # fc2 unwrapped (filter is "fc1").
    assert len(params) >= 1
    total = sum(p.numel() for p in params)
    assert total == 192


def test_get_peft_parameters_ia3(model: _TinyMLP) -> None:
    apply_peft(model, "ia3", target_modules=["fc1"])
    params = list(get_peft_parameters(model, "ia3"))
    # Only fc1 wrapped → out_features=32 → ia3_l has 32 trainable params.
    assert sum(p.numel() for p in params) == 32


def test_get_peft_parameters_unknown_method_raises(model: _TinyMLP) -> None:
    with pytest.raises(ValueError, match="not found in PEFTMethod registry"):
        list(get_peft_parameters(model, "definitely_not_a_method"))


def test_count_peft_parameters_lora(model: _TinyMLP) -> None:
    apply_peft(model, "lora", rank=4, alpha=8.0, target_modules=["fc1"])
    trainable, total = count_peft_parameters(model, "lora")
    # Total trainable = LoRA params on fc1 (64 + 128 = 192) + fc2.weight (512,
    # fc2 not wrapped) + norm.weight (16) + norm.bias (16) = 736.
    # Note: ``count_lora_parameters`` reports total trainable in the model,
    # not LoRA-specific trainable — for the LoRA-specific count use
    # ``get_peft_parameters`` + ``sum(p.numel())``.
    assert trainable == 192 + 512 + 16 + 16
    assert total > trainable  # fc1.base.weight is part of total but frozen


def test_count_peft_parameters_bitfit(model: _TinyMLP) -> None:
    apply_peft(model, "bitfit")
    trainable, _ = count_peft_parameters(model, "bitfit")
    # fc1.bias (out_features=32) + norm.bias (16) = 48 trainable.
    # fc2.bias is absent (bias=False). norm.weight ends in .weight so it's
    # frozen — BitFit enables biases only.
    assert trainable == 32 + 16


# ---------------------------------------------------------------------------
# merge_peft / unmerge_peft — None helpers raise NotImplementedError
# ---------------------------------------------------------------------------


def test_merge_peft_lora_round_trip(model: _TinyMLP) -> None:
    apply_peft(model, "lora", rank=4, alpha=8.0)
    merge_peft(model, "lora")
    # After merge, base weight has lora_A @ lora_B folded in. fc2 unchanged.
    unmerge_peft(model, "lora")  # No-op correctness: must not raise.


def test_merge_peft_bitfit_raises_not_implemented(model: _TinyMLP) -> None:
    apply_peft(model, "bitfit")
    with pytest.raises(NotImplementedError, match="bitfit"):
        merge_peft(model, "bitfit")


def test_merge_peft_prefix_tuning_raises_not_implemented(model: _TinyMLP) -> None:
    """Prefix Tuning's inference-time fold is ``fold_reparameterization``,
    not the merge/unmerge protocol — ``merge_peft`` is unsupported.
    """
    with pytest.raises(NotImplementedError, match="prefix_tuning"):
        merge_peft(model, "prefix_tuning")


def test_merge_peft_qlora_raises_not_implemented(model: _TinyMLP) -> None:
    apply_peft(model, "qlora", rank=4)
    with pytest.raises(NotImplementedError, match="qlora"):
        merge_peft(model, "qlora")


# ---------------------------------------------------------------------------
# disable_peft / enable_peft
# ---------------------------------------------------------------------------


def test_disable_enable_lora(model: _TinyMLP) -> None:
    apply_peft(model, "lora", rank=4, alpha=8.0)
    disable_peft(model, "lora")
    enable_peft(model, "lora")


def test_disable_peft_bitfit_raises_not_implemented(model: _TinyMLP) -> None:
    apply_peft(model, "bitfit")
    with pytest.raises(NotImplementedError, match="bitfit"):
        disable_peft(model, "bitfit")


# ---------------------------------------------------------------------------
# ensure_methods_registered idempotency
# ---------------------------------------------------------------------------


def test_ensure_methods_registered_is_idempotent() -> None:
    """Calling twice must not raise (registry rejects duplicate names)."""
    ensure_methods_registered()
    ensure_methods_registered()
    # Still the same set of names.
    assert "lora" in PEFT_REGISTRY.names()
    assert "adapter" in PEFT_REGISTRY.names()


def test_ensure_methods_registered_sets_module_flag() -> None:
    ensure_methods_registered()
    # Read the flag through the module so we don't capture a stale
    # ``False`` snapshot at ``from llm.core.peft.registry import
    # _methods_registered`` time — Python rebinds the module-level name
    # but our local binding still points to the original ``False``.
    assert _peft_registry_module._methods_registered is True


# ---------------------------------------------------------------------------
# PEFTMethod metadata
# ---------------------------------------------------------------------------


def test_adalora_requires_callback_flag() -> None:
    """AdaLoRA is the only method that needs a periodic trainer callback."""
    assert PEFT_REGISTRY.get("adalora").requires_callback is True


def test_non_adalora_methods_do_not_require_callback() -> None:
    for name in ("lora", "qlora", "prefix_tuning", "ia3", "bitfit", "adapter"):
        assert PEFT_REGISTRY.get(name).requires_callback is False, name


@pytest.mark.parametrize(
    ("name", "expected_filter"),
    [
        ("lora", TargetModuleFilter.LINEAR),
        ("qlora", TargetModuleFilter.LINEAR),
        ("adalora", TargetModuleFilter.LINEAR),
        ("ia3", TargetModuleFilter.LINEAR),
        ("bitfit", TargetModuleFilter.ANY),
        ("adapter", TargetModuleFilter.LINEAR),
        ("prefix_tuning", TargetModuleFilter.MHA),
    ],
)
def test_target_module_filter_metadata(name: str, expected_filter: TargetModuleFilter) -> None:
    assert PEFT_REGISTRY.get(name).target_module_filter is expected_filter


# ---------------------------------------------------------------------------
# Plugin entry-point group
# ---------------------------------------------------------------------------


def test_pyproject_declares_peft_methods_entry_point_group() -> None:
    """The ``llm.peft_methods`` group must be declared so third-party
    plugins can register custom PEFT methods without forking the registry.

    The group can be empty — the test only checks the section exists.
    """
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    eps = data["project"]["entry-points"]
    assert "llm.peft_methods" in eps, (
        "pyproject.toml must declare the 'llm.peft_methods' entry-point group "
        "so third parties can register custom PEFT methods via setuptools."
    )


# ---------------------------------------------------------------------------
# core/registry.py single-point re-export
# ---------------------------------------------------------------------------


def test_peft_registry_reexported_from_core_registry() -> None:
    """``from llm.core.registry import PEFT_REGISTRY`` must work — this is
    the canonical import path used by trainer-side modules.
    """
    from llm.core.registry import PEFT_REGISTRY as R

    assert R is PEFT_REGISTRY
    assert "lora" in R.names()


def test_ensure_methods_registered_reexported_from_core_registry() -> None:
    from llm.core.registry import ensure_peft_methods_registered

    assert callable(ensure_peft_methods_registered)


# ---------------------------------------------------------------------------
# merge_peft / unmerge_peft: the round trip must be numerically invisible
# ---------------------------------------------------------------------------


def test_merge_peft_lora_preserves_and_restores_model_output() -> None:
    """Merging must not change the model's output; unmerging must restore it.

    The existing ``test_merge_peft_lora_round_trip`` only asserts "must not
    raise" with no numeric check, so a merge that dropped or double-counted the
    adapter — the exact failure mode the scaling snapshot exists to prevent —
    would still pass.

    The baseline is captured AFTER ``apply_peft``: applying an adapter consumes
    RNG, so a model built before and after are not comparable (that mistake read
    as a 2.66 delta and looked like a merge bug).
    """
    from llm.models.decoder import DecoderModel

    torch.manual_seed(0)
    model = DecoderModel(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=16)
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    # Control: capture the adapter-free output of a model built from the SAME seed.
    torch.manual_seed(0)
    plain_model = DecoderModel(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=16)
    plain_model.eval()
    with torch.no_grad():
        plain_output = plain_model(input_ids).clone()

    apply_peft(model, "lora", rank=4, alpha=8.0, dropout=0.0)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.1)

    with torch.no_grad():
        adapted = model(input_ids).clone()

    assert not torch.allclose(adapted, plain_output, atol=1e-3), (
        "the adapter did not change the output, so the round-trip checks below are vacuous"
    )

    merge_peft(model, "lora")
    with torch.no_grad():
        merged = model(input_ids).clone()
    assert torch.allclose(merged, adapted, atol=1e-4), (
        f"merge changed the model output: max|delta|={(merged - adapted).abs().max().item():.2e}"
    )

    unmerge_peft(model, "lora")
    with torch.no_grad():
        unmerged = model(input_ids).clone()
    assert torch.allclose(unmerged, adapted, atol=1e-4), (
        f"unmerge did not restore the adapter-path output: max|delta|={(unmerged - adapted).abs().max().item():.2e}"
    )


def test_merge_peft_restores_every_wrapped_layer() -> None:
    """Every wrapped layer must be restored, not just the first one.

    A dispatch bug that stopped after one module would leave the rest folded at
    serve time; the aggregate output check can mask that when layers differ in
    scale.
    """
    from llm.models.decoder import DecoderModel

    torch.manual_seed(0)
    model = DecoderModel(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=16)
    model.eval()
    apply_peft(model, "lora", rank=4, alpha=8.0)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.1)

    wrapped = [module for module in model.modules() if type(module).__name__ == "LoRALinear"]
    assert len(wrapped) > 1, "expected several wrapped layers in this model"
    snapshots = [module.base_layer.weight.detach().clone() for module in wrapped]

    merge_peft(model, "lora")
    unmerge_peft(model, "lora")

    for module, snapshot in zip(wrapped, snapshots, strict=True):
        delta = (module.base_layer.weight - snapshot).abs().max().item()
        assert delta < 1e-5, f"a wrapped layer was not restored after unmerge: max|delta|={delta:.2e}"


# ---------------------------------------------------------------------------
# disable_peft / enable_peft: the adapter must actually disappear and return
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs", "markers"),
    [
        ("lora", {"rank": 4, "alpha": 8.0}, ("lora_A", "lora_B")),
        ("ia3", {}, ("ia3_l",)),
        ("adapter", {"bottleneck_dim": 8}, (".down.", ".up.")),
    ],
)
def test_disable_peft_removes_and_enable_restores_the_adapter(method, kwargs, markers) -> None:
    """Each method's disable/enable pair must be numerically exact.

    ``test_disable_enable_lora`` only asserts that calling them does not raise.
    Both directions matter and neither was checked: a ``disable`` that left the
    adapter partially active would silently corrupt every ablation run, and an
    ``enable`` that failed to restore it would make the adapter vanish after an
    evaluation sweep.

    Only ADAPTER parameters are perturbed. ``apply_peft`` leaves the embedding
    and norms trainable too; nudging those would move the base model and no
    disable could bring the output back (that mistake read as "disable is
    broken" for all four methods).
    """
    from llm.models.decoder import DecoderModel

    torch.manual_seed(0)
    model = DecoderModel(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=16)
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        baseline = model(input_ids).clone()

    apply_peft(model, method, **kwargs)
    adapter_params = [(name, param) for name, param in model.named_parameters() if any(m in name for m in markers)]
    assert adapter_params, f"no adapter parameters matched {markers} for {method}"
    for _name, param in adapter_params:
        param.data.normal_(0, 0.05)

    with torch.no_grad():
        adapted = model(input_ids).clone()
    assert not torch.allclose(adapted, baseline, atol=1e-4), (
        f"{method}: the adapter did not change the output, so the checks below would be vacuous"
    )

    disable_peft(model, method)
    with torch.no_grad():
        disabled = model(input_ids).clone()
    assert torch.allclose(disabled, baseline, atol=1e-4), (
        f"{method}: disable left the adapter active, max|delta|={(disabled - baseline).abs().max().item():.2e}"
    )

    enable_peft(model, method)
    with torch.no_grad():
        reenabled = model(input_ids).clone()
    assert torch.allclose(reenabled, adapted, atol=1e-4), (
        f"{method}: enable did not restore the adapter, max|delta|={(reenabled - adapted).abs().max().item():.2e}"
    )
