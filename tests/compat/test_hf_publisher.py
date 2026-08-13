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
from tests.support.devices import DEFAULT_DEVICE
from tests.support.models import decoder_model_kwargs

# --- Helpers ---------------------------------------------------------------


def _make_small_decoder() -> torch.nn.Module:
    """Construct a tiny DecoderModel on the default device (GPU-first)."""
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
        device=str(DEFAULT_DEVICE),
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
        # MLP activation must be persisted so from_pretrained rebuilds the
        # model with the same MLP function (real llama = silu; a gelu-trained
        # model must not silently switch to silu on reload).
        "hidden_act",
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
    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)

    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


def _make_non_glu_decoder() -> torch.nn.Module:
    """A DecoderModel with the DEFAULT ``use_glu=False`` MLP.

    The default model's MLP has only ``fc1`` (activated) + ``fc2`` — no
    ``gate_proj``. An earlier save->load roundtrip silently broke for this
    configuration (RIL ISS-056): the publisher emitted ``mlp.up_proj`` from
    the (nonexistent) ``mlp.gate_proj``, so the artifact was missing
    ``up_proj``; then the loader rebuilt a GLU MLP (hardcoded
    ``use_glu=True``) and left every ``gate_proj`` at random init.
    """
    from llm.models.decoder import DecoderModel

    return DecoderModel(
        **decoder_model_kwargs(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=32,
            attn_impl="mha",
            mlp_impl="mlp",
            device=str(DEFAULT_DEVICE),
            # DEFAULT use_glu=False
        )
    )


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_learned_pos_encoding(tmp_path: Path):
    """Regression (RIL ISS-063): a ``pos_encoding_learned=True`` model must
    roundtrip save->load preserving its trained positional-encoding weights.

    The trained ``pos_embedding.weight`` had no weight-mapping entry so it was
    dropped on save, and the loader never honored ``pos_encoding_learned`` —
    reloading silently fell back to sinusoidal and lost the trained PE."""
    from llm.models.decoder import DecoderModel

    model = DecoderModel(
        **decoder_model_kwargs(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=32,
            attn_impl="mha",
            mlp_impl="mlp",
            pos_encoding_learned=True,
            device=str(DEFAULT_DEVICE),
        )
    )
    model.eval()

    # Randomize the PE so a silent drop is visible as a logit mismatch.
    with torch.no_grad():
        model.embedding_layer.positional_encoding.pos_embedding.weight.copy_(
            torch.randn_like(model.embedding_layer.positional_encoding.pos_embedding.weight)
        )

    save_pretrained(model, tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("pos_encoding_learned") is True

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert isinstance(reloaded, DecoderModel)
    assert reloaded.embedding_layer.positional_encoding.learned is True

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_rms_norm(tmp_path: Path):
    """Regression (RIL ISS-062): ``norm_impl='rms_norm'`` must roundtrip
    save->load with the same normalization, not silently become LayerNorm.

    The loader/build never persisted nor honored ``norm_impl``, so an
    RMSNorm-trained model was rebuilt with LayerNorm after save->load — a
    different normalization function with the same weights."""
    from llm.models.decoder import DecoderModel

    model = DecoderModel(
        **decoder_model_kwargs(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=32,
            attn_impl="mha",
            mlp_impl="mlp",
            norm_impl="rms_norm",
            device=str(DEFAULT_DEVICE),
        )
    )
    model.eval()
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("norm_impl") == "rms_norm"

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert isinstance(reloaded, DecoderModel)
    assert reloaded.transformer_blocks[0].norm1.__class__.__name__ == "RMSNorm"

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_rope(tmp_path: Path):
    """Regression (RIL ISS-062): a RoPE model must roundtrip save->load
    with the same rotary position embedding — not silently become a
    non-RoPE model (or vice versa).

    Before the fix ``core.rope`` had zero callers and neither the publisher
    nor the loader persisted ``use_rope``/``rope_theta``: a RoPE model was
    rebuilt without rotation (position came only from additive PE), a
    functionally different network."""
    from llm.models.decoder import DecoderModel

    model = DecoderModel(
        **decoder_model_kwargs(
            vocab_size=64,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=32,
            attn_impl="mha",
            mlp_impl="mlp",
            use_rope=True,
            rope_theta=500000.0,
            device=str(DEFAULT_DEVICE),
        )
    )
    model.eval()
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("use_rope") is True
    assert config.get("rope_theta") == 500000.0

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert isinstance(reloaded, DecoderModel)
    assert reloaded.use_rope is True
    assert getattr(reloaded, "rope_theta", None) == 500000.0
    # The MHA backend must actually own a rotary module.
    assert hasattr(reloaded.transformer_blocks[0].self_attn, "rope")
    assert reloaded.embedding_layer.use_rope is True

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_default_keeps_non_rope(tmp_path: Path):
    """Roundtrip self-consistency (RIL ISS-062): a DEFAULT (non-RoPE) model
    must persist ``use_rope: false`` so the loader (which defaults missing
    ``use_rope`` to True for external checkpoints) does NOT rebuild it as a
    rotary model. Otherwise save->load would silently change the network."""
    from llm.models.decoder import DecoderModel

    model = DecoderModel(**decoder_model_kwargs(device=str(DEFAULT_DEVICE)))
    model.eval()
    assert model.use_rope is False
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("use_rope") is False

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert isinstance(reloaded, DecoderModel)
    assert reloaded.use_rope is False
    assert not hasattr(reloaded.transformer_blocks[0].self_attn, "rope")
    assert reloaded.embedding_layer.use_rope is False

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_persists_bias_flags(tmp_path: Path):
    """Regression (RIL ISS-062): bias flags are model-defining. A biased
    (repo-default) model must roundtrip with its biases, and a bias-free
    model must NOT silently gain random biases on load.

    Before the fix neither publisher nor loader persisted
    ``qkv_bias/mlp_bias/lm_head_bias``; an external bias-free Llama was built
    with the repo's biased defaults, so ``lm_head.bias`` (and qkv/mlp biases)
    stayed at random init — a functionally different network."""
    from llm.models.decoder import DecoderModel

    # Biased default model: persists + honors bias=True.
    biased = DecoderModel(**decoder_model_kwargs(device=str(DEFAULT_DEVICE)))
    biased.eval()
    save_pretrained(biased, tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["qkv_bias"] is True
    assert cfg["mlp_bias"] is True
    assert cfg["lm_head_bias"] is True

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert reloaded.qkv_bias is True
    assert reloaded.lm_head_bias is True
    assert reloaded.transformer_blocks[0].self_attn.qkv_proj.bias is not None
    assert reloaded.lm_head.bias is not None

    torch.manual_seed(0)
    ids = torch.randint(0, biased.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = biased(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)

    # Bias-free model (external-Llama-shaped): stays bias-free on roundtrip.
    unbiased = DecoderModel(
        **decoder_model_kwargs(
            qkv_bias=False,
            mlp_bias=False,
            lm_head_bias=False,
            device=str(DEFAULT_DEVICE),
        )
    )
    unbiased.eval()
    out_dir = tmp_path / "unbiased"
    save_pretrained(unbiased, out_dir)
    cfg2 = json.loads((out_dir / "config.json").read_text())
    assert cfg2["qkv_bias"] is False
    assert cfg2["lm_head_bias"] is False

    reloaded2 = from_pretrained(out_dir, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded2.eval()
    assert reloaded2.lm_head_bias is False
    assert reloaded2.transformer_blocks[0].self_attn.qkv_proj.bias is None
    assert reloaded2.lm_head.bias is None

    torch.manual_seed(0)
    with torch.no_grad():
        original_logits2 = unbiased(input_ids=ids).detach()
        reloaded_logits2 = reloaded2(input_ids=ids).detach()
    assert torch.allclose(original_logits2, reloaded_logits2, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_roundtrip_default_non_glu_model(tmp_path: Path):
    """Regression (RIL ISS-056): a DEFAULT (non-GLU) model must roundtrip
    save->load with equivalent logits — not leave the MLP at random init."""
    from llm.models.decoder import DecoderModel

    model = _make_non_glu_decoder()
    model.eval()
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("use_glu") is False

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()

    # The reloaded MLP must be non-GLU and carry real weights (not random).
    assert isinstance(reloaded, DecoderModel)
    assert reloaded.transformer_blocks[0].mlp.use_glu is False

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
    with torch.no_grad():
        original_logits = model(input_ids=ids).detach()
        reloaded_logits = reloaded(input_ids=ids).detach()
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
def test_save_pretrained_persists_mlp_activation(tmp_path: Path):
    """``from_pretrained`` rebuilds the model with the *saved* MLP activation.

    Real Llama uses ``silu``; before this fix the loader defaulted to gelu
    and never persisted ``hidden_act``, so a silu-trained (real-llama-style)
    model would silently switch to gelu across a save -> load roundtrip — a
    different MLP function with the same weights.
    """
    from llm.models.decoder import DecoderModel

    model = DecoderModel(
        **decoder_model_kwargs(
            vocab_size=64,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            max_seq_len=16,
            attn_impl="mha",
            mlp_impl="mlp",
            mlp_activation="silu",  # real-llama-style SwiGLU activation
            use_glu=True,
            device=str(DEFAULT_DEVICE),
        )
    )
    model.eval()
    save_pretrained(model, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["hidden_act"] == "silu"

    reloaded = from_pretrained(tmp_path, device=str(DEFAULT_DEVICE), dtype=torch.float32)
    reloaded.eval()
    assert reloaded.transformer_blocks[0].mlp.activation_name == "silu"

    torch.manual_seed(0)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8), device=DEFAULT_DEVICE)
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
