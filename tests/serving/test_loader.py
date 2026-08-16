"""Tests for serving checkpoint loader."""

import os
import pickle
import string

import pytest
import torch

from llm.serving.config import ServingConfig
from llm.serving.loader import (
    infer_num_layers,
    infer_vocab_size,
    load_model_and_tokenizer,
    load_training_checkpoint,
)
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.distributed import model_state_dict


def test_infer_vocab_size_from_lm_head(tiny_model, tiny_config):
    state = model_state_dict(tiny_model)
    assert infer_vocab_size(state) == tiny_config.model.vocab_size


def test_infer_vocab_size_from_tied_embedding(tiny_model, tiny_config):
    """Regression (RIL ISS-167): a config-less checkpoint with tied embeddings
    has no ``lm_head.weight`` — the embedding fallback must use the REAL key
    (``embedding_layer.token_embeddings.weight``), not the misspelled one."""
    state = model_state_dict(tiny_model)
    state.pop("lm_head.weight", None)
    assert infer_vocab_size(state) == tiny_config.model.vocab_size


def test_infer_num_layers(tiny_model, tiny_config):
    state = model_state_dict(tiny_model)
    assert infer_num_layers(state) == tiny_config.model.num_layers


def test_loader_refuses_malicious_pickle_without_executing(tmp_path):
    """Regression (RIL ISS-170): ``load_training_checkpoint`` on an
    attacker-controlled ``.pt`` must NOT run arbitrary ``__reduce__`` code.
    The file below would ``os.system``-touch a marker if executed under the
    old ``weights_only=False`` probe; the hardened loader refuses it."""
    marker = tmp_path / "pwned"
    payload = tmp_path / "evil.pt"

    class _Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    with payload.open("wb") as f:
        pickle.dump(_Exploit(), f)

    with pytest.raises((Exception,)):
        # The probe (weights_only=True) refuses the foreign global; the
        # legacy-payload path must not re-load it with weights_only=False.
        load_training_checkpoint(payload)

    assert not marker.exists(), "malicious pickle executed code during load"


def test_load_training_checkpoint_roundtrip(tmp_path, tiny_model, tiny_config):
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 0,
            "loss": 1.0,
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )

    checkpoint = load_training_checkpoint(ckpt_path)
    assert checkpoint.model_config["hidden_size"] == tiny_config.model.hidden_size
    assert checkpoint.epoch == 0


def test_load_training_checkpoint_strips_compile_prefix(tmp_path, tiny_model, tiny_config):
    """Legacy checkpoints with torch.compile's ``_orig_mod.`` prefix load fine.

    ``llm-train`` compiles the model by default, and checkpoints saved
    before the save-side fix carried ``_orig_mod.*`` keys. The serving
    loader must strip the prefix before ``load_state_dict`` or every
    weight is silently dropped and the served model stays at random init.
    """
    ckpt_path = tmp_path / "model.pt"
    prefixed = {f"_orig_mod.{key}": value for key, value in model_state_dict(tiny_model).items()}
    torch.save(
        {"model_state": prefixed, "model_config": tiny_config.model.model_dump()},
        ckpt_path,
    )

    checkpoint = load_training_checkpoint(ckpt_path)
    assert not any(key.startswith("_orig_mod.") for key in checkpoint.model_state)
    assert checkpoint.model_state["lm_head.weight"] is not None


def test_load_training_checkpoint_v2_split_layout(tmp_path, tiny_model, tiny_config):
    """The serving loader accepts the modern v2 split layout written by
    CheckpointManager — referenced by stem, by sidecar path, or by a
    ``.pt``-suffixed stem (legacy-style path that resolves to the trio)."""
    import logging

    from llm.training.core.checkpoint import CheckpointManager
    from llm.training.core.config import CheckpointConfig

    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval=1, keep_last_n=2),
        rank=0,
        logger=logging.getLogger("test-loader"),
    )
    manager.save_checkpoint(
        epoch=0,
        model=tiny_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        loss=1.5,
        extra_state={"stream_data": {"0": {"line_index": 3}}},
        model_config=tiny_config.model.model_dump(),
    )

    for path in (
        tmp_path / "epoch_1",  # stem (no suffix)
        tmp_path / "epoch_1.safetensors",  # weights sidecar
        tmp_path / "epoch_1.pt",  # legacy-style path, resolves to the trio
    ):
        checkpoint = load_training_checkpoint(path)
        assert checkpoint.model_config["hidden_size"] == tiny_config.model.hidden_size
        assert checkpoint.epoch == 0
        assert checkpoint.loss == 1.5


def test_load_training_checkpoint_missing_raises(tmp_path):
    """A path with neither the legacy nor the v2 layout fails loudly."""
    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(tmp_path / "does-not-exist")


def test_load_quantized_model_blob_roundtrip(tmp_path, tiny_model, tiny_config):
    """The loader accepts the bare quantized-model blob emitted by
    ``llm-quantize`` (``torch.save`` of the whole module) and serves it
    directly, preserving the quantized layers and their outputs."""
    from llm.quantization import GPTQConfig, GPTQQuantizedLinear, quantize_model_gptq

    vocab = tiny_config.model.vocab_size
    model_device = str(next(tiny_model.parameters()).device)
    calib = [torch.randint(0, vocab, (2, 16), device=model_device) for _ in range(3)]
    quantized = quantize_model_gptq(
        tiny_model,
        iter(calib),
        GPTQConfig(bits=4, group_size=128),
    )
    quantized.eval()

    ckpt_path = tmp_path / "quantized.pt"
    torch.save(quantized, ckpt_path)

    tokenizer = SimpleCharacterTokenizer(list(string.printable[:vocab]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    config = ServingConfig(
        model_path=str(ckpt_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_type="simple",
    )
    model, loaded_tokenizer = load_model_and_tokenizer(config)
    model.eval()

    # The quantized layers survive the load — this is the whole point of
    # the blob path (state-dict reconstruction could not recover per-layer
    # group_size / bits from tensor shapes alone).
    assert isinstance(model, type(quantized))
    assert any(isinstance(m, GPTQQuantizedLinear) for m in model.modules())
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("ab")) == "ab"

    # Move loaded model to the same device as the original tiny_model so
    # the forward parity comparison runs on GPU (the loader reads the blob
    # with map_location="cpu" by default).
    model.to(model_device)

    # Forward parity with the direct quantized model (same seed, eval mode).
    ids = torch.randint(0, vocab, (2, 8), device=model_device)
    torch.manual_seed(0)
    expected = quantized(ids)
    torch.manual_seed(0)
    actual = model(ids)
    torch.testing.assert_close(actual, expected)


def test_load_model_and_tokenizer_from_checkpoint(tmp_path, tiny_model, tiny_config):
    tokenizer = SimpleCharacterTokenizer(list(string.printable[: tiny_config.model.vocab_size]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )

    config = ServingConfig(
        model_path=str(ckpt_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_type="simple",
    )
    model, loaded_tokenizer = load_model_and_tokenizer(config)

    assert model.lm_head.out_features == tiny_config.model.vocab_size
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("ab")) == "ab"


def test_load_model_requires_tokenizer_path(tmp_path, tiny_model, tiny_config):
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"model_state": model_state_dict(tiny_model), "model_config": tiny_config.model.model_dump()},
        ckpt_path,
    )

    config = ServingConfig(model_path=str(ckpt_path))
    with pytest.raises(ValueError, match="tokenizer_path is required"):
        load_model_and_tokenizer(config)


def test_dummy_fallback_without_checkpoint():
    config = ServingConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    assert model.lm_head.out_features == tokenizer.vocab_size
