"""Tests for the HuggingFace publish helpers (audit T3 #7).

Pins the **roundtrip guarantee** of :mod:`llm.compat.hf_publisher`:

1. ``save_pretrained`` writes a directory that the existing
   ``from_pretrained`` can load back into an equivalent
   :class:`llm.models.DecoderModel`.
2. The forward pass produces equivalent logits within numerical
   tolerance (no random init between save and load).
3. ``convert_our_weights`` is the inverse of ``convert_hf_weights``
   for the supported Llama-style names.
4. ``push_to_hub`` is a soft-dependency contract: missing
   ``huggingface_hub`` → clear ``ImportError``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import llm.compat.hf_publisher  # noqa: F401 — import for side effects
from llm.compat.hf_loader import from_pretrained
from llm.compat.hf_publisher import (
    HF_HUB_AVAILABLE,
    SAFETENSORS_AVAILABLE,
    push_to_hub,
    save_pretrained,
)
from llm.compat.weight_mapping import convert_hf_weights, convert_our_weights
from tests.support.models import decoder_model_kwargs

# --- Helpers ---------------------------------------------------------------


def _make_small_decoder() -> torch.nn.Module:
    """Construct a tiny CPU-only DecoderModel."""
    from llm.models.decoder import DecoderModel

    kwargs = decoder_model_kwargs(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        attn_impl="mha",
        mlp_impl="mlp",
        # Match the loader's ``from_pretrained`` config so the roundtrip
        # truly exercises a save -> load with matching architectures.
        use_glu=True,
    )
    return DecoderModel(**kwargs)


# --- Soft-dependency contract ---------------------------------------------


def test_module_imports_cleanly():
    """Module imports even when ``safetensors`` / ``huggingface_hub`` are absent."""
    assert isinstance(SAFETENSORS_AVAILABLE, bool)
    assert isinstance(HF_HUB_AVAILABLE, bool)


# --- Reverse mapping (unit) ----------------------------------------------


def test_reverse_mapping_roundtrip():
    """``convert_our_weights`` inverts ``convert_hf_weights`` for Llama.

    The mapping covers the **rename-only** keys (o_proj, gate_proj,
    up_proj, down_proj, layer norms, embeddings, lm_head). The Q/K/V
    split/concat is exercised separately in the publisher roundtrip
    test.
    """
    hf_sd = {
        "model.embed_tokens.weight": torch.zeros(8, 16),
        "model.norm.weight": torch.ones(16),
        "lm_head.weight": torch.zeros(8, 16),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(16, 16),
        "model.layers.0.self_attn.o_proj.bias": torch.zeros(16),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(32, 16),
        "model.layers.0.mlp.gate_proj.bias": torch.zeros(32),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(32, 16),
        "model.layers.0.mlp.up_proj.bias": torch.zeros(32),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(16, 32),
        "model.layers.0.mlp.down_proj.bias": torch.zeros(16),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(16),
    }
    our_sd = convert_hf_weights(hf_sd, architecture="llama", num_layers=1)
    roundtrip = convert_our_weights(our_sd, architecture="llama", num_layers=1)
    # Round-tripped set must equal the original HF names.
    assert set(roundtrip.keys()) == set(hf_sd.keys())
    for name, original_tensor in hf_sd.items():
        assert torch.equal(roundtrip[name], original_tensor), name


# --- save_pretrained (roundtrip) ------------------------------------------


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_writes_config_and_safetensors(tmp_path: Path):
    """``save_pretrained`` writes both files into the target directory."""
    model = _make_small_decoder()
    out_dir = save_pretrained(model, tmp_path)

    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_config_is_llama_shaped(tmp_path: Path):
    """Written ``config.json`` carries the keys ``from_pretrained`` reads."""
    model = _make_small_decoder()
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "llama"
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "torch_dtype",
    ):
        assert key in config, f"missing key: {key}"


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_through_from_pretrained(tmp_path: Path):
    """Roundtrip: ``save_pretrained`` → ``from_pretrained`` → equivalent forward.

    We compare logits from the same input on the saved-then-loaded
    model and the original. With identical weights they should match
    within fp32 tolerance.
    """
    model = _make_small_decoder()
    model.eval()
    save_pretrained(model, tmp_path)

    # Reload via the existing HF loader.
    reloaded = from_pretrained(tmp_path, device="cpu", dtype=torch.float32)
    reloaded.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8))

    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


# --- push_to_hub (mocked) --------------------------------------------------


@pytest.mark.skipif(not HF_HUB_AVAILABLE, reason="huggingface_hub not installed")
def test_push_to_hub_uploads_saved_directory(tmp_path: Path):
    """``push_to_hub`` writes a staging dir then calls ``upload_folder``."""
    model = _make_small_decoder()
    save_dir = tmp_path / "stage"

    # Stub the huggingface_hub module so we don't need network or auth.
    fake_api = MagicMock()

    class _FakeHfHub:
        HfApi = MagicMock(return_value=fake_api)

    with patch.dict("sys.modules", {"huggingface_hub": _FakeHfHub()}):
        url = push_to_hub(
            model,
            repo_id="test-org/test-model",
            save_directory=save_dir,
        )

    # Local files must exist (the save step still runs).
    assert (save_dir / "config.json").exists()
    assert (save_dir / "model.safetensors").exists()

    # HfApi methods were called.
    fake_api.create_repo.assert_called_once()
    fake_api.upload_folder.assert_called_once()
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "test-org/test-model"
    assert Path(kwargs["folder_path"]) == save_dir

    assert url == "https://huggingface.co/test-org/test-model"


# --- ImportError gating ----------------------------------------------------


@pytest.mark.skipif(SAFETENSORS_AVAILABLE, reason="safetensors is installed — gate on the no-install branch")
def test_save_pretrained_raises_without_safetensors(tmp_path: Path):
    """Without ``safetensors`` installed, ``save_pretrained`` raises ImportError."""
    model = _make_small_decoder()
    with pytest.raises(ImportError, match="safetensors"):
        save_pretrained(model, tmp_path)


@pytest.mark.skipif(HF_HUB_AVAILABLE, reason="huggingface_hub is installed — gate on the no-install branch")
def test_push_to_hub_raises_without_huggingface_hub(tmp_path: Path):
    """Without ``huggingface_hub`` installed, ``push_to_hub`` raises ImportError."""
    model = _make_small_decoder()
    with pytest.raises(ImportError, match="huggingface_hub"):
        push_to_hub(model, "test-org/test-model", save_directory=tmp_path)


# --- Smoke test for the public surface ------------------------------------


def test_module_exposes_save_and_push():
    """Both helpers are importable from the module."""
    module = importlib.import_module("llm.compat.hf_publisher")
    assert module.save_pretrained is save_pretrained
    assert module.push_to_hub is push_to_hub
