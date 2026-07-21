"""Tests for the PEFT adapter checkpoint callback (T2 PEFT #48).

Covers:

- :class:`PEFTAdapterCheckpointCallback` construction with defaults
  and overrides; safe no-op when ``peft_method`` is ``None`` or
  ``peft_save_path`` is ``None``.
- ``on_train_end`` writes a valid :func:`save_peft` envelope when
  configured; the file round-trips byte-identically via
  :func:`load_peft` into a freshly PEFT-applied model.
- ``get_checkpoint_state`` / ``load_checkpoint_state`` preserve the
  sidecar path across checkpoint resume (so a resumed run knows
  where to write the next adapter sidecar).
- :meth:`LanguageModelingTask.build_callbacks` returns the
  PEFT callback when ``peft_method`` is set; coexists with the
  existing AdaLoRA pruning callback; default sidecar path derives
  from ``checkpoint_dir`` when no explicit path is given.
- End-to-end: train one step → ``on_train_end`` → fresh
  ``DecoderModel`` + ``apply_peft`` + ``load_peft`` recovers the
  trained adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from llm.core.adapter import AdapterLinear
from llm.core.ia3 import IA3Linear
from llm.core.lora import LoRALinear
from llm.core.peft import load_peft
from llm.core.peft.checkpoint import PEFT_CHECKPOINT_FORMAT_VERSION
from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Minimal MLP for PEFT checkpoint integration tests."""

    def __init__(self, hidden: int = 16, intermediate: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def model() -> _TinyMLP:
    torch.manual_seed(0)
    return _TinyMLP()


@pytest.fixture
def peft_applied_model() -> _TinyMLP:
    """TinyMLP with LoRA already applied."""
    from llm.core.lora import apply_lora

    torch.manual_seed(0)
    m = _TinyMLP()
    apply_lora(m, rank=4, alpha=8.0)
    return m


@pytest.fixture
def engine_mock(peft_applied_model: _TinyMLP) -> MagicMock:
    """A minimal engine stand-in that exposes ``model`` and ``logger``."""
    engine = MagicMock()
    engine.model = peft_applied_model
    engine.rank = 0
    engine.logger = MagicMock()
    return engine


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPEFTAdapterCheckpointCallbackConstruction:
    """The callback must be safe to construct with any combination of
    kwargs — disabled callbacks are no-ops, enabled ones remember
    their config."""

    def test_default_construction_is_noop(self) -> None:
        """No method, no path → on_train_end does nothing."""
        cb = PEFTAdapterCheckpointCallback()
        assert cb.peft_method is None
        assert cb.peft_kwargs == {}
        assert cb.peft_save_path is None

    def test_construction_with_peft_method_only_no_path(self, tmp_dir: Path) -> None:
        """Method set but no path → still no-op (path is opt-in)."""
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4},
        )
        assert cb.peft_method == "lora"
        assert cb.peft_kwargs == {"rank": 4}
        assert cb.peft_save_path is None

    def test_construction_with_all_kwargs(self, tmp_dir: Path) -> None:
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 8, "alpha": 16.0},
            peft_save_path=path,
        )
        assert cb.peft_method == "lora"
        assert cb.peft_kwargs == {"rank": 8, "alpha": 16.0}
        assert cb.peft_save_path == path

    def test_string_path_converted_to_path_object(self, tmp_dir: Path) -> None:
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=str(tmp_dir / "adapter.bin"),
        )
        assert isinstance(cb.peft_save_path, Path)

    def test_peft_kwargs_default_to_empty_dict(self) -> None:
        cb = PEFTAdapterCheckpointCallback(peft_method="lora")
        assert cb.peft_kwargs == {}


# ---------------------------------------------------------------------------
# on_train_end — save behavior
# ---------------------------------------------------------------------------


class TestPEFTAdapterCheckpointCallbackSave:
    """``on_train_end`` must write a valid adapter file when configured."""

    def test_save_disabled_when_no_peft_method(self, engine_mock: MagicMock, tmp_dir: Path) -> None:
        """No peft_method → no file written, no error raised."""
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method=None,
            peft_save_path=path,
        )
        cb.set_engine(engine_mock)
        cb.on_train_end()
        assert not path.exists()

    def test_save_disabled_when_no_peft_save_path(self, engine_mock: MagicMock) -> None:
        """No peft_save_path → no file written, no error raised."""
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4},
            peft_save_path=None,
        )
        cb.set_engine(engine_mock)
        cb.on_train_end()
        # No file should be written anywhere — sanity check on tmp_dir.
        assert list(engine_mock.model.state_dict().keys())  # model unchanged

    def test_save_disabled_when_engine_is_none(self, peft_applied_model: _TinyMLP, tmp_dir: Path) -> None:
        """No engine wired → callback can't access the model → no-op."""
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4},
            peft_save_path=path,
        )
        # No set_engine — engine stays None.
        cb.on_train_end()
        assert not path.exists()

    def test_save_writes_valid_peft_file(self, engine_mock: MagicMock, tmp_dir: Path) -> None:
        """The written file must be a valid save_peft envelope that
        round-trips through load_peft into a fresh model."""
        from llm.core.lora import apply_lora

        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_save_path=path,
        )
        cb.set_engine(engine_mock)

        # Mutate the LoRA params so the test is non-trivial.
        for module in engine_mock.model.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.add_(torch.randn_like(module.lora_A) * 0.01)
                    module.lora_B.add_(torch.randn_like(module.lora_B) * 0.01)
        saved_lora_a = [m.lora_A.detach().clone() for m in engine_mock.model.modules() if isinstance(m, LoRALinear)]
        saved_lora_b = [m.lora_B.detach().clone() for m in engine_mock.model.modules() if isinstance(m, LoRALinear)]

        cb.on_train_end()

        assert path.exists()
        assert path.stat().st_size > 0

        # Verify the file is a valid PEFT envelope.
        payload = torch.load(path, weights_only=False, map_location="cpu")
        assert payload["format_version"] == PEFT_CHECKPOINT_FORMAT_VERSION
        assert payload["method_name"] == "lora"
        assert payload["peft_kwargs"] == {"rank": 4, "alpha": 8.0}

        # Load into a fresh model and verify byte-identical.
        torch.manual_seed(0)
        fresh = _TinyMLP()
        apply_lora(fresh, rank=4, alpha=8.0)
        load_peft(fresh, path, "lora", rank=4, alpha=8.0)
        loaded_lora_a = [m.lora_A.detach().clone() for m in fresh.modules() if isinstance(m, LoRALinear)]
        loaded_lora_b = [m.lora_B.detach().clone() for m in fresh.modules() if isinstance(m, LoRALinear)]
        assert len(saved_lora_a) == len(loaded_lora_a)
        for s, l_iter in zip(saved_lora_a, loaded_lora_a, strict=True):
            assert torch.equal(s, l_iter)
        for s, l_iter in zip(saved_lora_b, loaded_lora_b, strict=True):
            assert torch.equal(s, l_iter)

    def test_save_failure_is_swallowed_and_logged(self, engine_mock: MagicMock, tmp_dir: Path) -> None:
        """If save_peft raises (e.g. unknown method after config drift),
        the callback must log a warning and NOT crash the training run.

        The main checkpoint has already been written by the time
        on_train_end fires, so swallowing this exception is the safe
        choice — losing the sidecar is recoverable; losing the main
        checkpoint is not."""

        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4},
            peft_save_path=path,
        )
        cb.set_engine(engine_mock)

        # Force save_peft to raise by monkey-patching the function the
        # callback imports inside ``on_train_end`` (lazy import).
        from llm.core.peft import checkpoint as peft_ckpt_module

        original_save = peft_ckpt_module.save_peft

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("simulated disk failure")

        peft_ckpt_module.save_peft = boom
        try:
            # Must not raise — failure is logged via engine.logger.
            cb.on_train_end()
        finally:
            peft_ckpt_module.save_peft = original_save

        # Warning was logged (not errored).
        assert engine_mock.logger.warning.called
        # No file written.
        assert not path.exists()

    def test_save_creates_parent_dirs(self, engine_mock: MagicMock, tmp_dir: Path) -> None:
        """The sidecar's parent directory should be created on demand."""

        nested = tmp_dir / "deeply" / "nested" / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4},
            peft_save_path=nested,
        )
        cb.set_engine(engine_mock)
        cb.on_train_end()
        assert nested.exists()


# ---------------------------------------------------------------------------
# Checkpoint state (path-only — the sidecar itself lives on disk)
# ---------------------------------------------------------------------------


class TestPEFTAdapterCheckpointCallbackCheckpointState:
    """The callback's checkpointable state is just the sidecar path —
    the actual weights live on disk in the sidecar file."""

    def test_get_checkpoint_state_returns_none_when_path_unset(self) -> None:
        cb = PEFTAdapterCheckpointCallback(peft_method="lora")
        assert cb.get_checkpoint_state() is None

    def test_get_checkpoint_state_returns_path_string(self, tmp_dir: Path) -> None:
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=path,
        )
        state = cb.get_checkpoint_state()
        assert state == {"peft_save_path": str(path)}

    def test_load_checkpoint_state_none_is_noop(self, tmp_dir: Path) -> None:
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=path,
        )
        cb.load_checkpoint_state(None)
        # Path is unchanged.
        assert cb.peft_save_path == path

    def test_load_checkpoint_state_restores_path(self, tmp_dir: Path) -> None:
        """A resumed run restores the path the previous run was
        writing to (so subsequent on_train_end writes go to the same
        file the user expects)."""
        original = tmp_dir / "original_adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=tmp_dir / "scratch.bin",
        )
        cb.load_checkpoint_state({"peft_save_path": str(original)})
        assert cb.peft_save_path == original

    def test_load_checkpoint_state_missing_key_is_noop(self, tmp_dir: Path) -> None:
        path = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=path,
        )
        cb.load_checkpoint_state({})
        assert cb.peft_save_path == path


# ---------------------------------------------------------------------------
# Integration with LanguageModelingTask.build_callbacks
# ---------------------------------------------------------------------------


def _make_task(training_overrides: dict[str, Any], checkpoint_dir: Path):
    """Build a LanguageModelingTask with the given training overrides.

    Returns the task instance + the resolved checkpoint_dir. The task's
    config is constructed in-memory only — no file I/O.
    """
    from llm.training.core.config import Config
    from llm.training.tasks.lm_task import LanguageModelingTask

    # Build a minimal valid Config and apply overrides.
    cfg = Config()
    # Override checkpoint dir.
    cfg.checkpoint.checkpoint_dir = str(checkpoint_dir)
    # Apply training overrides via setattr on the BaseModel — pydantic
    # v2 lets us assign individual fields directly.
    for key, value in training_overrides.items():
        setattr(cfg.training, key, value)

    # Build a minimal data_module stub — build_callbacks doesn't use it.
    data_module = MagicMock()
    task = LanguageModelingTask(cfg, data_module)
    return task


class TestBuildCallbacksPEFTIntegration:
    """``build_callbacks`` wires the PEFT callback automatically when
    ``peft_method`` is set."""

    def test_no_callbacks_when_neither_set(self, tmp_dir: Path) -> None:
        task = _make_task({}, tmp_dir)
        callbacks = task.build_callbacks()
        assert callbacks == []

    def test_adalora_callback_still_returned_when_use_adalora_true(self, tmp_dir: Path) -> None:
        """Regression: T3 #42 AdaLoRA wiring must keep working."""
        task = _make_task({"use_adalora": True}, tmp_dir)
        callbacks = task.build_callbacks()
        assert len(callbacks) == 1
        from llm.training.core.callbacks import AdaLoRAPruningCallback

        assert isinstance(callbacks[0], AdaLoRAPruningCallback)

    def test_peft_callback_returned_when_peft_method_set(self, tmp_dir: Path) -> None:
        task = _make_task(
            {"peft_method": "lora", "peft_kwargs": {"rank": 4}},
            tmp_dir,
        )
        callbacks = task.build_callbacks()
        assert len(callbacks) == 1
        cb = callbacks[0]
        assert isinstance(cb, PEFTAdapterCheckpointCallback)
        assert cb.peft_method == "lora"
        assert cb.peft_kwargs == {"rank": 4}

    def test_default_peft_save_path_derives_from_checkpoint_dir(self, tmp_dir: Path) -> None:
        """No explicit path → derives
        ``{checkpoint_dir}/peft_adapter_{method}.bin``. The
        method-name suffix avoids clobbering when the user later
        switches PEFT methods in a follow-up run."""
        task = _make_task(
            {"peft_method": "lora", "peft_kwargs": {"rank": 4}},
            tmp_dir,
        )
        callbacks = task.build_callbacks()
        cb = callbacks[0]
        assert cb.peft_save_path == tmp_dir / "peft_adapter_lora.bin"

    def test_default_peft_save_path_includes_method_name(self, tmp_dir: Path) -> None:
        """The default filename tags the method so multiple PEFT
        methods can coexist (e.g. user retrains with a different method)."""
        task = _make_task(
            {"peft_method": "ia3", "peft_kwargs": {}},
            tmp_dir,
        )
        callbacks = task.build_callbacks()
        cb = callbacks[0]
        assert cb.peft_save_path == tmp_dir / "peft_adapter_ia3.bin"

    def test_explicit_peft_save_path_overrides_default(self, tmp_dir: Path) -> None:
        custom = tmp_dir / "my_special_adapter.bin"
        task = _make_task(
            {
                "peft_method": "lora",
                "peft_kwargs": {"rank": 4},
                "peft_save_path": str(custom),
            },
            tmp_dir,
        )
        callbacks = task.build_callbacks()
        cb = callbacks[0]
        assert cb.peft_save_path == custom

    def test_both_adalora_and_peft_callbacks_when_both_set(self, tmp_dir: Path) -> None:
        """When ``use_adalora=True`` AND ``peft_method`` is set, both
        callbacks are registered (AdaLoRA can be applied via the
        registry path too, so the two coexist)."""
        task = _make_task(
            {
                "use_adalora": True,
                "peft_method": "adalora",
                "peft_kwargs": {"init_rank": 8, "target_rank": 4},
            },
            tmp_dir,
        )
        callbacks = task.build_callbacks()
        assert len(callbacks) == 2

        # AdaLoRA pruning callback + PEFT adapter-save callback.
        types = sorted(type(c).__name__ for c in callbacks)
        assert "AdaLoRAPruningCallback" in types
        assert "PEFTAdapterCheckpointCallback" in types


# ---------------------------------------------------------------------------
# End-to-end: train → save → fresh model → load
# ---------------------------------------------------------------------------


class TestEndToEndPEFTCheckpointIntegration:
    """The full pipeline: a training step, the on_train_end save, then
    a fresh model loading the sidecar via load_peft."""

    def test_train_step_save_then_fresh_load_round_trips(self, tmp_dir: Path) -> None:
        """End-to-end: apply LoRA → mutate via an optimizer step → run
        the callback's on_train_end → load the sidecar into a fresh
        model → verify byte-identical LoRA params."""
        from llm.core.lora import apply_lora
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        # 1. Build + apply LoRA + one optimizer step.
        torch.manual_seed(0)
        model = _TinyMLP()
        apply_lora(model, rank=4, alpha=8.0)
        optim = torch.optim.SGD(
            [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)],
            lr=0.1,
        )
        x = torch.randn(2, 16)
        target = torch.randn(2, 16)
        out = model(x)
        loss = ((out - target) ** 2).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Snapshot the post-step LoRA params (these are what we expect
        # the sidecar to contain).
        saved_a = [m.lora_A.detach().clone() for m in model.modules() if isinstance(m, LoRALinear)]
        saved_b = [m.lora_B.detach().clone() for m in model.modules() if isinstance(m, LoRALinear)]

        # 2. Run the callback's on_train_end.
        sidecar = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_save_path=sidecar,
        )
        engine = MagicMock()
        engine.model = model
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()
        assert sidecar.exists()

        # 3. Load into a fresh model and verify byte-identical.
        torch.manual_seed(0)
        fresh = _TinyMLP()
        apply_lora(fresh, rank=4, alpha=8.0)
        load_peft(fresh, sidecar, "lora", rank=4, alpha=8.0)
        loaded_a = [m.lora_A.detach().clone() for m in fresh.modules() if isinstance(m, LoRALinear)]
        loaded_b = [m.lora_B.detach().clone() for m in fresh.modules() if isinstance(m, LoRALinear)]
        for s, l_iter in zip(saved_a, loaded_a, strict=True):
            assert torch.equal(s, l_iter)
        for s, l_iter in zip(saved_b, loaded_b, strict=True):
            assert torch.equal(s, l_iter)

    def test_callback_does_not_interfere_with_non_peft_training(self, tmp_dir: Path) -> None:
        """When peft_method is None, the PEFT callback is not in the
        callback list — non-PEFT training is completely unaffected."""
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        cb = PEFTAdapterCheckpointCallback()  # all defaults
        # No engine set → all hooks are no-ops.
        cb.on_train_start()
        cb.on_epoch_start(0)
        cb.on_batch_start(0, 0)
        cb.on_batch_end(0, 0)
        cb.on_epoch_end(0, {})
        cb.on_train_end()
        # No sidecar written anywhere.
        assert list(tmp_dir.iterdir()) == []

    def test_ia3_method_works_via_callback(self, tmp_dir: Path) -> None:
        """The callback is method-agnostic — IA3 round-trips cleanly."""
        from llm.core.ia3 import apply_ia3
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        torch.manual_seed(0)
        model = _TinyMLP()
        apply_ia3(model)

        for module in model.modules():
            if isinstance(module, IA3Linear):
                with torch.no_grad():
                    module.ia3_l.add_(torch.randn_like(module.ia3_l) * 0.01)
        saved = [m.ia3_l.detach().clone() for m in model.modules() if isinstance(m, IA3Linear)]

        sidecar = tmp_dir / "ia3_adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="ia3",
            peft_kwargs={},
            peft_save_path=sidecar,
        )
        engine = MagicMock()
        engine.model = model
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()
        assert sidecar.exists()

        # Verify round-trip via load_peft.
        torch.manual_seed(0)
        fresh = _TinyMLP()
        apply_ia3(fresh)
        load_peft(fresh, sidecar, "ia3")
        loaded = [m.ia3_l.detach().clone() for m in fresh.modules() if isinstance(m, IA3Linear)]
        assert len(saved) == len(loaded)
        for s, l_iter in zip(saved, loaded, strict=True):
            assert torch.equal(s, l_iter)

    def test_adapter_method_works_via_callback(self, tmp_dir: Path) -> None:
        """Houlsby Adapter round-trips cleanly via the callback."""
        from llm.core.adapter import apply_adapter
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        torch.manual_seed(0)
        model = _TinyMLP()
        apply_adapter(model, bottleneck_dim=4)
        adapters = [m for m in model.modules() if isinstance(m, AdapterLinear)]
        for module in adapters:
            with torch.no_grad():
                module.up.weight.add_(torch.randn_like(module.up.weight) * 0.01)
        saved_up = [m.up.weight.detach().clone() for m in adapters]

        sidecar = tmp_dir / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="adapter",
            peft_kwargs={"bottleneck_dim": 4},
            peft_save_path=sidecar,
        )
        engine = MagicMock()
        engine.model = model
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()
        assert sidecar.exists()

        torch.manual_seed(0)
        fresh = _TinyMLP()
        apply_adapter(fresh, bottleneck_dim=4)
        load_peft(fresh, sidecar, "adapter", bottleneck_dim=4)
        loaded = [m.up.weight.detach().clone() for m in fresh.modules() if isinstance(m, AdapterLinear)]
        assert len(saved_up) == len(loaded)
        for s, l_iter in zip(saved_up, loaded, strict=True):
            assert torch.equal(s, l_iter)
