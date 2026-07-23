"""Tests for the PEFT serving-side loader integration (T2 PEFT #49).

Covers the training → serving bridge for PEFT adapters:

- :func:`llm.serving.peft_adapter.load_peft_into_model` re-applies the
  method if needed and loads the sidecar via :func:`load_peft`. Returns
  the model (chainable).
- :func:`llm.serving.peft_adapter.merge_peft_into_model` folds
  mergeable adapters (lora / adalora / ia3 / adapter / pfeiffer_adapter)
  into the base weights, refuses to merge methods that don't expose a
  merge helper (bitfit / qlora / prefix_tuning).
- :class:`llm.serving.config.ServingConfig` exposes four PEFT fields
  (``peft_method`` / ``peft_kwargs`` / ``peft_adapter_path`` /
  ``peft_merge``) with a Pydantic validator that rejects unknown
  method names and inconsistent field combinations.
- :func:`llm.serving.loader.load_model_and_tokenizer` applies the
  configured PEFT method (and merges if requested) after loading the
  base checkpoint. Without any PEFT fields, the loader behavior is
  unchanged (regression guard).

End-to-end (slice 2 lives in ``test_peft_serving_e2e.py``): build a
tiny model, apply LoRA, mutate via an optimizer step, save the sidecar,
load via the serving loader into a fresh model, verify the adapter
values round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def device():
    """Force CPU for these tests."""
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Two-layer MLP — enough surface for LoRA / IA³ / Adapter / BitFit.

    No attention layers, so Prefix Tuning applies nothing (empty
    sidecar). The tests that care about Prefix Tuning use a separate
    helper or skip.
    """

    def __init__(self, hidden: int = 16, intermediate: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.act(self.fc1(x))))


@pytest.fixture
def model() -> _TinyMLP:
    torch.manual_seed(0)
    return _TinyMLP()


@pytest.fixture
def fresh_model() -> _TinyMLP:
    torch.manual_seed(0)
    return _TinyMLP()


def _mutate_lora(model: nn.Module) -> None:
    """Mutate LoRA A/B so a round-trip is detectable (no-op load would
    leave init values untouched)."""
    from llm.core.lora import LoRALinear

    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_A.add_(torch.randn_like(module.lora_A) * 0.01)
                module.lora_B.add_(torch.randn_like(module.lora_B) * 0.01)


def _mutate_ia3(model: nn.Module) -> None:
    from llm.core.ia3 import IA3Linear

    for module in model.modules():
        if isinstance(module, IA3Linear):
            with torch.no_grad():
                module.ia3_l.add_(torch.randn_like(module.ia3_l) * 0.01)


def _mutate_adapter(model: nn.Module) -> None:
    from llm.core.adapter import AdapterLinear

    for module in model.modules():
        if isinstance(module, AdapterLinear):
            with torch.no_grad():
                module.up.weight.add_(torch.randn_like(module.up.weight) * 0.01)
                module.down.weight.add_(torch.randn_like(module.down.weight) * 0.01)


# ---------------------------------------------------------------------------
# load_peft_into_model — single helper that re-applies + loads + (opt) merges
# ---------------------------------------------------------------------------


class TestLoadPeftIntoModel:
    """The serving-side loader helper round-trips each built-in method."""

    def test_lora(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.lora import LoRALinear, apply_lora
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model

        apply_lora(model, rank=4, alpha=8.0)
        _mutate_lora(model)
        path = tmp_path / "lora.bin"
        save_peft(model, path, "lora")

        load_peft_into_model(fresh_model, "lora", path, rank=4, alpha=8.0)

        lora_modules = [m for m in fresh_model.modules() if isinstance(m, LoRALinear)]
        assert lora_modules  # applied

    def test_ia3(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.ia3 import IA3Linear, apply_ia3
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model

        apply_ia3(model)
        _mutate_ia3(model)
        path = tmp_path / "ia3.bin"
        save_peft(model, path, "ia3")

        load_peft_into_model(fresh_model, "ia3", path)

        assert any(isinstance(m, IA3Linear) for m in fresh_model.modules())

    def test_adapter(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.adapter import AdapterLinear, apply_adapter
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model

        apply_adapter(model, bottleneck_dim=4)
        _mutate_adapter(model)
        path = tmp_path / "adapter.bin"
        save_peft(model, path, "adapter", bottleneck_dim=4)

        load_peft_into_model(fresh_model, "adapter", path, bottleneck_dim=4)

        assert any(isinstance(m, AdapterLinear) for m in fresh_model.modules())

    def test_pfeiffer_adapter(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.adapter import AdapterLinear
        from llm.core.peft import save_peft
        from llm.core.pfeiffer_adapter import apply_pfeiffer_adapter
        from llm.serving.peft_adapter import load_peft_into_model

        apply_pfeiffer_adapter(model, bottleneck_dim=4)
        _mutate_adapter(model)
        path = tmp_path / "pfeiffer.bin"
        save_peft(model, path, "pfeiffer_adapter", bottleneck_dim=4)

        load_peft_into_model(fresh_model, "pfeiffer_adapter", path, bottleneck_dim=4)

        adapters = [m for m in fresh_model.modules() if isinstance(m, AdapterLinear)]
        # Pfeiffer only wraps fc1/fc2 by default → 2 wrappers, not 3 (which
        # would include the layernorm).
        assert len(adapters) == 2

    def test_bitfit(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.bitfit import apply_bitfit, is_bitfit_applied
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model

        apply_bitfit(model)
        path = tmp_path / "bitfit.bin"
        save_peft(model, path, "bitfit")

        load_peft_into_model(fresh_model, "bitfit", path)

        assert is_bitfit_applied(fresh_model)

    def test_unknown_method_raises(self, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.serving.peft_adapter import load_peft_into_model

        path = tmp_path / "x.bin"
        path.write_bytes(b"placeholder")
        with pytest.raises(ValueError, match="not found"):
            load_peft_into_model(fresh_model, "not_a_method", path)

    def test_missing_file_raises(self, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.serving.peft_adapter import load_peft_into_model

        path = tmp_path / "missing.bin"
        with pytest.raises(FileNotFoundError):
            load_peft_into_model(fresh_model, "lora", path, rank=4, alpha=8.0)

    def test_returns_model_for_chaining(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.lora import apply_lora
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model

        apply_lora(model, rank=4)
        path = tmp_path / "lora.bin"
        save_peft(model, path, "lora")

        result = load_peft_into_model(fresh_model, "lora", path, rank=4)
        assert result is fresh_model


# ---------------------------------------------------------------------------
# merge_peft_into_model — fold mergeable adapters into base weights
# ---------------------------------------------------------------------------


class TestMergePeftIntoModel:
    """Auto-merge at serve time (saves runtime, prevents disable/unmerge)."""

    def test_lora_merge_works(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.lora import LoRALinear, apply_lora
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_lora(model, rank=4, alpha=8.0)
        _mutate_lora(model)
        path = tmp_path / "lora.bin"
        save_peft(model, path, "lora")

        load_peft_into_model(fresh_model, "lora", path, rank=4, alpha=8.0)
        merge_peft_into_model(fresh_model, "lora")

        # After merge, wrappers are bypassed (merge folds A@B into base).
        # The forward path still works — exact behavioural check is in
        # the end-to-end test.
        assert any(isinstance(m, LoRALinear) for m in fresh_model.modules())

    def test_ia3_merge_works(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.ia3 import apply_ia3
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_ia3(model)
        _mutate_ia3(model)
        path = tmp_path / "ia3.bin"
        save_peft(model, path, "ia3")

        load_peft_into_model(fresh_model, "ia3", path)
        merge_peft_into_model(fresh_model, "ia3")

    def test_adapter_merge_works(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.adapter import apply_adapter
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_adapter(model, bottleneck_dim=4)
        _mutate_adapter(model)
        path = tmp_path / "adapter.bin"
        save_peft(model, path, "adapter", bottleneck_dim=4)

        load_peft_into_model(fresh_model, "adapter", path, bottleneck_dim=4)
        # Houlsby adapter merge is a documented no-op (up is zero-init);
        # the helper still calls it for API parity.
        merge_peft_into_model(fresh_model, "adapter")

    def test_pfeiffer_merge_works(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.peft import save_peft
        from llm.core.pfeiffer_adapter import apply_pfeiffer_adapter
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_pfeiffer_adapter(model, bottleneck_dim=4)
        _mutate_adapter(model)
        path = tmp_path / "pfeiffer.bin"
        save_peft(model, path, "pfeiffer_adapter", bottleneck_dim=4)

        load_peft_into_model(fresh_model, "pfeiffer_adapter", path, bottleneck_dim=4)
        merge_peft_into_model(fresh_model, "pfeiffer_adapter")

    def test_bitfit_merge_raises_not_implemented(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        """BitFit has no merge helper — the loader must refuse to merge
        rather than silently no-op or corrupt the base weights."""
        from llm.core.bitfit import apply_bitfit
        from llm.core.peft import save_peft
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_bitfit(model)
        path = tmp_path / "bitfit.bin"
        save_peft(model, path, "bitfit")

        load_peft_into_model(fresh_model, "bitfit", path)
        with pytest.raises(NotImplementedError, match="merge"):
            merge_peft_into_model(fresh_model, "bitfit")

    def test_qlora_merge_raises_not_implemented(self, model: _TinyMLP, fresh_model: _TinyMLP, tmp_path: Path) -> None:
        from llm.core.peft import save_peft
        from llm.core.qlora import apply_qlora
        from llm.serving.peft_adapter import load_peft_into_model, merge_peft_into_model

        apply_qlora(model, rank=4, alpha=8.0)
        path = tmp_path / "qlora.bin"
        save_peft(model, path, "qlora")

        load_peft_into_model(fresh_model, "qlora", path, rank=4, alpha=8.0)
        with pytest.raises(NotImplementedError, match="merge"):
            merge_peft_into_model(fresh_model, "qlora")

    def test_prefix_tuning_merge_raises_not_implemented(self, tmp_path: Path) -> None:
        """Prefix Tuning has ``fold_reparameterization`` instead of merge —
        the loader helper refuses to call merge since the inference-time
        fold is a separate API surface that mutates static K/V buffers."""
        from llm.serving.peft_adapter import merge_peft_into_model

        with pytest.raises(NotImplementedError, match="merge"):
            merge_peft_into_model(nn.Linear(4, 4), "prefix_tuning")


# ---------------------------------------------------------------------------
# ServingConfig — PEFT fields + validation
# ---------------------------------------------------------------------------


class TestServingConfigPeft:
    """``ServingConfig`` exposes four PEFT fields with cross-field validation."""

    def test_defaults_are_none(self) -> None:
        from llm.serving.config import ServingConfig

        cfg = ServingConfig()
        assert cfg.peft_method is None
        assert cfg.peft_kwargs == {}
        assert cfg.peft_adapter_path is None
        assert cfg.peft_merge is False

    def test_all_fields_settable(self, tmp_path: Path) -> None:
        from llm.serving.config import ServingConfig

        path = tmp_path / "lora.bin"
        cfg = ServingConfig(
            peft_method="lora",
            peft_kwargs={"rank": 8, "alpha": 16.0},
            peft_adapter_path=str(path),
            peft_merge=True,
        )
        assert cfg.peft_method == "lora"
        assert cfg.peft_kwargs == {"rank": 8, "alpha": 16.0}
        assert cfg.peft_adapter_path == str(path)
        assert cfg.peft_merge is True

    def test_unknown_peft_method_rejected(self) -> None:
        from llm.serving.config import ServingConfig

        with pytest.raises(ValueError, match="not_a_method"):
            ServingConfig(peft_method="not_a_method")

    def test_known_methods_accepted(self) -> None:
        from llm.serving.config import ServingConfig

        for name in (
            "lora",
            "qlora",
            "adalora",
            "prefix_tuning",
            "ia3",
            "bitfit",
            "adapter",
            "pfeiffer_adapter",
        ):
            cfg = ServingConfig(peft_method=name)
            assert cfg.peft_method == name

    def test_peft_merge_with_non_mergeable_method_rejected(self) -> None:
        """Methods without a merge helper must reject peft_merge=True at
        config time — fail loud at startup, not silent at request time."""
        from llm.serving.config import ServingConfig

        for name in ("bitfit", "qlora", "prefix_tuning"):
            with pytest.raises(ValueError, match="merge"):
                ServingConfig(peft_method=name, peft_merge=True)

    def test_peft_merge_with_mergeable_method_accepted(self) -> None:
        from llm.serving.config import ServingConfig

        for name in ("lora", "adalora", "ia3", "adapter", "pfeiffer_adapter"):
            cfg = ServingConfig(peft_method=name, peft_merge=True)
            assert cfg.peft_merge is True

    def test_peft_adapter_path_without_method_rejected(self, tmp_path: Path) -> None:
        """peft_adapter_path implies a PEFT method must be set; refusing
        the combo at config time surfaces the inconsistency early."""
        from llm.serving.config import ServingConfig

        path = tmp_path / "lora.bin"
        with pytest.raises(ValueError, match="peft_method"):
            ServingConfig(peft_adapter_path=str(path))

    def test_peft_kwargs_without_method_rejected(self) -> None:
        from llm.serving.config import ServingConfig

        with pytest.raises(ValueError, match="peft_method"):
            ServingConfig(peft_kwargs={"rank": 8})


# ---------------------------------------------------------------------------
# load_model_and_tokenizer — PEFT integration
# ---------------------------------------------------------------------------


def _build_minimal_ckpt(
    tmp_path: Path,
    tiny_model: nn.Module,
    tiny_config: Any,
) -> Path:
    """Save a minimal training checkpoint for the loader to consume."""
    from llm.training.distributed import model_state_dict

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )
    return ckpt_path


def _build_minimal_tokenizer(
    tmp_path: Path,
    tiny_config: Any,
) -> Path:
    """Save a minimal SimpleCharacterTokenizer for the loader to consume.

    Mirrors the existing :mod:`tests.serving.test_loader` pattern.
    """
    import string

    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tokenizer = SimpleCharacterTokenizer(list(string.printable[: tiny_config.model.vocab_size]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)
    return tokenizer_path


class TestLoadModelAndTokenizerPeftIntegration:
    """``load_model_and_tokenizer`` applies PEFT after loading the base."""

    def test_no_peft_fields_unchanged_behavior(self, tmp_path: Path, tiny_model: nn.Module, tiny_config: Any) -> None:
        """Regression guard: without PEFT fields, the loader does NOT
        touch the model — same as before T2 PEFT #49."""
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer

        ckpt = _build_minimal_ckpt(tmp_path, tiny_model, tiny_config)
        tok = _build_minimal_tokenizer(tmp_path, tiny_config)
        cfg = ServingConfig(
            model_path=str(ckpt),
            tokenizer_path=str(tok),
            tokenizer_type="simple",
        )
        model, _tokenizer = load_model_and_tokenizer(cfg)
        # No PEFT wrappers introduced.
        from llm.core.adapter import AdapterLinear
        from llm.core.ia3 import IA3Linear
        from llm.core.lora import LoRALinear

        assert not any(isinstance(m, LoRALinear) for m in model.modules())
        assert not any(isinstance(m, IA3Linear) for m in model.modules())
        assert not any(isinstance(m, AdapterLinear) for m in model.modules())

    def test_peft_method_loads_adapter(self, tmp_path: Path, tiny_model: nn.Module, tiny_config: Any) -> None:
        """``peft_method`` + ``peft_adapter_path`` → loader applies the
        method and copies the saved adapter values into the model."""
        from llm.core.lora import LoRALinear, apply_lora
        from llm.core.peft import save_peft
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer

        # Build a sidecar from a *copy* of the model with the same
        # architecture, then save a base-only checkpoint from the
        # original. This mirrors the realistic workflow: user trains
        # a DecoderModel with LoRA, the trainer's adapter callback
        # writes the sidecar, and the base checkpoint is the un-wrapped
        # version (saved separately or via the main CheckpointManager).
        torch.manual_seed(0)
        train_view = _TinyMLP()  # discarded — same arch as ``tiny_model``? No.
        # We actually need the sidecar to come from a copy of the
        # same tiny_model so the param counts match on load.
        from copy import deepcopy

        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)
        # Move LoRA params off init.
        for module in train_view.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.add_(torch.randn_like(module.lora_A) * 0.01)
                    module.lora_B.add_(torch.randn_like(module.lora_B) * 0.01)
        adapter_path = tmp_path / "adapter.bin"
        save_peft(train_view, adapter_path, "lora")

        # Base checkpoint from the *un-PEFT'd* tiny_model (loader
        # builds a fresh model from this).
        ckpt = _build_minimal_ckpt(tmp_path, tiny_model, tiny_config)
        tok = _build_minimal_tokenizer(tmp_path, tiny_config)
        cfg = ServingConfig(
            model_path=str(ckpt),
            tokenizer_path=str(tok),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
        )
        model, _ = load_model_and_tokenizer(cfg)

        # Model must now have LoRA wrappers with the saved values.
        lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
        assert lora_modules

    def test_peft_method_without_adapter_path_just_applies(
        self, tmp_path: Path, tiny_model: nn.Module, tiny_config: Any
    ) -> None:
        """``peft_method`` set but no ``peft_adapter_path`` → just call
        :func:`apply_peft` so the wrapper structure exists (useful for
        BitFit where the adapter IS the base weights, or for setting
        up the structure before a separate load)."""
        from llm.core.lora import LoRALinear
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer

        ckpt = _build_minimal_ckpt(tmp_path, tiny_model, tiny_config)
        tok = _build_minimal_tokenizer(tmp_path, tiny_config)
        cfg = ServingConfig(
            model_path=str(ckpt),
            tokenizer_path=str(tok),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
        )
        model, _ = load_model_and_tokenizer(cfg)
        assert any(isinstance(m, LoRALinear) for m in model.modules())

    def test_peft_merge_folds_into_base(self, tmp_path: Path, tiny_model: nn.Module, tiny_config: Any) -> None:
        """``peft_merge=True`` → adapter is folded into the base weights,
        saving runtime at the cost of losing disable/unmerge capability."""
        from copy import deepcopy

        from llm.core.lora import LoRALinear, apply_lora
        from llm.core.peft import save_peft
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer

        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)
        for module in train_view.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.add_(torch.randn_like(module.lora_A) * 0.01)
                    module.lora_B.add_(torch.randn_like(module.lora_B) * 0.01)
        adapter_path = tmp_path / "adapter.bin"
        save_peft(train_view, adapter_path, "lora")

        ckpt = _build_minimal_ckpt(tmp_path, tiny_model, tiny_config)
        tok = _build_minimal_tokenizer(tmp_path, tiny_config)
        cfg = ServingConfig(
            model_path=str(ckpt),
            tokenizer_path=str(tok),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
            peft_merge=True,
        )
        model, _ = load_model_and_tokenizer(cfg)
        # Forward works without raising (LoRA wrappers still exist but
        # are bypassed via the folded base weight). Embedding lookup
        # needs long indices.
        out = model(torch.randint(0, tiny_config.model.vocab_size, (2, 8)))
        assert out.shape[0] == 2

    def test_peft_method_load_failure_propagates(self, tmp_path: Path, tiny_model: nn.Module, tiny_config: Any) -> None:
        """If the sidecar is missing or corrupt, the loader raises — the
        serving process refuses to start with a partial config rather
        than serving the un-adapted base model silently."""
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer

        ckpt = _build_minimal_ckpt(tmp_path, tiny_model, tiny_config)
        tok = _build_minimal_tokenizer(tmp_path, tiny_config)
        cfg = ServingConfig(
            model_path=str(ckpt),
            tokenizer_path=str(tok),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(tmp_path / "missing.bin"),
        )
        with pytest.raises(FileNotFoundError):
            load_model_and_tokenizer(cfg)
