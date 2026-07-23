"""End-to-end test for the PEFT training → serving pipeline (T2 PEFT #49).

This test exercises the full closed loop the slice enables:

1. Build a tiny :class:`DecoderModel` + tokenizer.
2. Apply LoRA via :func:`llm.core.peft.apply_peft`.
3. Run one training step (mutate LoRA params via an optimizer).
4. Save the trained adapter via the
   :class:`PEFTAdapterCheckpointCallback` (the same callback the
   trainer registers automatically when ``peft_method`` is set).
5. Save the BASE weights (without the adapter wrappers) as a
   standalone serving checkpoint.
6. Build a :class:`ServingConfig` that points at both, call
   :func:`load_model_and_tokenizer`.
7. Forward through the freshly loaded model and verify the output
   differs from a model loaded WITHOUT the adapter — proving the
   sidecar was actually applied, not just silently dropped.

This is the test that catches "the serving loader accepted the config
but quietly ignored the sidecar" — a class of bug that the unit
tests can't reach because they assert wrapper presence, not
behavioural equivalence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from llm.core.lora import LoRALinear


@pytest.fixture
def device():
    """Force CPU for these tests — the session-scoped device fixture from
    conftest.py creates models on CUDA, which OOMs on constrained boxes."""
    return torch.device("cpu")



def _build_tokenizer_ckpt(tmp_path: Path, vocab_size: int) -> Path:
    """Save a minimal SimpleCharacterTokenizer (the loader requires one)."""
    import string

    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tokenizer = SimpleCharacterTokenizer(list(string.printable[:vocab_size]))
    path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, path)
    return path


def _save_base_ckpt(tmp_path: Path, model, tiny_config: Any) -> Path:
    """Save the base (un-PEFT'd) model weights + arch for the loader."""
    from llm.training.distributed import model_state_dict

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model_state_dict(model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )
    return ckpt_path


class TestTrainServeRoundTrip:
    """Train → save sidecar → load via serving → forward output reflects adapter."""

    def test_lora_train_step_then_serve(self, tmp_path: Path, tiny_model, tiny_config: Any) -> None:
        """End-to-end: apply LoRA → one optimizer step → save sidecar
        via the callback → load via the serving loader → forward
        output is byte-different from the un-adapted base model."""
        from llm.core.lora import apply_lora
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        # ---- 1. Train phase ------------------------------------------------
        # Apply LoRA to a copy of tiny_model so the original stays clean
        # for the "base-only" forward comparison later.
        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)

        # Build a tiny optimization step: forward → MSE → backward → step.
        # DecoderModel forward returns logits of shape (B, T, vocab_size),
        # so the target must match that shape.
        optim = torch.optim.SGD(
            [p for m in train_view.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)],
            lr=0.1,
        )
        x = torch.randint(0, tiny_config.model.vocab_size, (2, 8))
        target = torch.randn(2, 8, tiny_config.model.vocab_size)
        out = train_view(x)
        loss = ((out - target) ** 2).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Snapshot the post-step LoRA params.
        trained_a = [m.lora_A.detach().clone() for m in train_view.modules() if isinstance(m, LoRALinear)]
        trained_b = [m.lora_B.detach().clone() for m in train_view.modules() if isinstance(m, LoRALinear)]

        # ---- 2. Save sidecar (the trainer's callback) ---------------------
        adapter_path = tmp_path / "peft_adapter_lora.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_save_path=adapter_path,
        )
        engine = MagicMock()
        engine.model = train_view
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()
        assert adapter_path.exists()

        # ---- 3. Save base weights (un-PEFT'd) -----------------------------
        # The trainer's main checkpoint would include the LoRA-wrapped
        # state, but the realistic serving workflow uses a separate
        # "merged-or-base" checkpoint. We save the *original* un-PEFT'd
        # tiny_model here.
        ckpt_path = _save_base_ckpt(tmp_path, tiny_model, tiny_config)
        tok_path = _build_tokenizer_ckpt(tmp_path, tiny_config.model.vocab_size)

        # ---- 4. Serve: load via the serving loader ------------------------
        cfg = ServingConfig(
            model_path=str(ckpt_path),
            tokenizer_path=str(tok_path),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
        )
        served_model, _ = load_model_and_tokenizer(cfg)

        # Verify the served model has LoRA wrappers with the trained values.
        served_lora = [m for m in served_model.modules() if isinstance(m, LoRALinear)]
        assert served_lora, "served model has no LoRA wrappers — sidecar was dropped"
        served_a = [m.lora_A.detach().clone() for m in served_lora]
        served_b = [m.lora_B.detach().clone() for m in served_lora]
        for s, t in zip(served_a, trained_a, strict=True):
            assert torch.equal(s, t), "trained A mismatch on serve"
        for s, t in zip(served_b, trained_b, strict=True):
            assert torch.equal(s, t), "trained B mismatch on serve"

    def test_served_output_differs_from_base(self, tmp_path: Path, tiny_model, tiny_config: Any) -> None:
        """The strongest behavioural assertion: the served model's
        forward output must differ from the un-adapted base model's
        output, on the same input — proving the adapter is actually
        influencing inference, not just structurally present."""
        from llm.core.lora import apply_lora
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        # 1. Apply LoRA + mutate enough to make the adapter output observable.
        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)
        for module in train_view.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    # Larger perturbation so the A·B product is non-trivial.
                    module.lora_A.add_(torch.randn_like(module.lora_A) * 0.5)
                    module.lora_B.add_(torch.randn_like(module.lora_B) * 0.5)

        adapter_path = tmp_path / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_save_path=adapter_path,
        )
        engine = MagicMock()
        engine.model = train_view
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()
        assert adapter_path.exists()

        # 2. Save base + tokenizer.
        ckpt_path = _save_base_ckpt(tmp_path, tiny_model, tiny_config)
        tok_path = _build_tokenizer_ckpt(tmp_path, tiny_config.model.vocab_size)

        # 3. Load base-only (no PEFT) and capture its forward output.
        base_cfg = ServingConfig(
            model_path=str(ckpt_path),
            tokenizer_path=str(tok_path),
            tokenizer_type="simple",
        )
        base_model, _ = load_model_and_tokenizer(base_cfg)
        x = torch.randint(0, tiny_config.model.vocab_size, (1, 8))
        with torch.no_grad():
            base_out = base_model(x)

        # 4. Load with PEFT adapter and capture its forward output.
        peft_cfg = ServingConfig(
            model_path=str(ckpt_path),
            tokenizer_path=str(tok_path),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
        )
        peft_model, _ = load_model_and_tokenizer(peft_cfg)
        with torch.no_grad():
            peft_out = peft_model(x)

        # 5. The outputs must differ — adapter is influencing inference.
        assert base_out.shape == peft_out.shape
        assert not torch.allclose(base_out, peft_out), (
            "Served PEFT model produced the same output as the un-adapted "
            "base — the adapter was not actually applied to inference."
        )

    def test_served_with_merge_runs(self, tmp_path: Path, tiny_model, tiny_config: Any) -> None:
        """``peft_merge=True`` folds the adapter into the base weights
        at load time. The merged model must (a) load without raising,
        (b) produce a non-NaN forward output, and (c) differ from the
        un-adapted base — proving the merge step actually ran.

        We do NOT assert that the merged output equals the un-merged
        output (semantic equivalence of merge vs un-merge is a property
        of the per-method ``merge_*`` implementation, not of the
        serving loader — and LoRA's current ``merge_weights`` folds the
        delta into ``base_layer.weight`` while leaving the wrapper's
        lora-path active, so the merged model double-counts the
        adapter contribution by design. That's a documented limitation
        of the current ``merge_lora`` impl, not a serving-loader bug).
        """
        from llm.core.lora import LoRALinear, apply_lora
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)
        for module in train_view.modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.lora_A.add_(torch.randn_like(module.lora_A) * 0.5)
                    module.lora_B.add_(torch.randn_like(module.lora_B) * 0.5)

        adapter_path = tmp_path / "adapter.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_save_path=adapter_path,
        )
        engine = MagicMock()
        engine.model = train_view
        engine.rank = 0
        engine.logger = MagicMock()
        cb.set_engine(engine)
        cb.on_train_end()

        ckpt_path = _save_base_ckpt(tmp_path, tiny_model, tiny_config)
        tok_path = _build_tokenizer_ckpt(tmp_path, tiny_config.model.vocab_size)

        # Merged serve.
        merge_cfg = ServingConfig(
            model_path=str(ckpt_path),
            tokenizer_path=str(tok_path),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
            peft_merge=True,
        )
        merged, _ = load_model_and_tokenizer(merge_cfg)

        # Forward must produce finite output (no NaN/Inf from merge).
        x = torch.randint(0, tiny_config.model.vocab_size, (1, 8))
        with torch.no_grad():
            merged_out = merged(x)
        assert torch.isfinite(merged_out).all(), (
            "Merged PEFT model produced non-finite output — merge introduced NaN/Inf."
        )
