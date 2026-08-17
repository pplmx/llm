"""GGUF load-back milestone: export → load_gguf_model round-trips (round 71).

The exporter was write-only: a GGUF file could not be read back into a
usable model. Now ``export_to_gguf(..., model_config=config.model_dump())``
persists the full architecture config as ``general.llm_model_config`` and
:func:`llm.export.gguf.loader.load_gguf_model` rebuilds the exact model
from it. These tests pin the round trip:

- F32 exports restore the state dict bit-exactly;
- F16 / block-quantized exports restore within the quantizer's error;
- a GGUF without the config blob is refused with a clear error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from llm.export.gguf import GGUFReader, GGUFWriter, export_to_gguf, load_gguf_model
from llm.export.gguf.spec import GGUFError
from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig


@pytest.fixture
def roundtrip_config() -> ModelConfig:
    """Tiny decoder config — small enough to export/load fast."""
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        use_rope=True,
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
    )


def _build_model(cfg: ModelConfig) -> torch.nn.Module:
    ensure_builtins_registered()
    return ModelFactory.from_config(cfg)


def _state_dict_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu() for k, v in state.items()}


def test_roundtrip_f32_is_exact(roundtrip_config, tmp_path):
    torch.manual_seed(0)
    model = _build_model(roundtrip_config).eval()
    path = export_to_gguf(
        model,
        tmp_path / "f32.gguf",
        quantize="f32",
        model_config=roundtrip_config.model_dump(),
    )

    restored = load_gguf_model(path)

    assert isinstance(restored, torch.nn.Module)
    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())
    assert set(original) == set(recovered)
    for name in original:
        assert torch.equal(original[name], recovered[name]), (
            f"{name} drifted ({float((original[name] - recovered[name]).abs().max()):.2e})"
        )


def test_roundtrip_f16_within_quantizer_error(roundtrip_config, tmp_path):
    torch.manual_seed(1)
    model = _build_model(roundtrip_config).eval()
    path = export_to_gguf(
        model,
        tmp_path / "f16.gguf",
        quantize="f16",
        model_config=roundtrip_config.model_dump(),
    )

    restored = load_gguf_model(path)
    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())

    for name in original:
        max_abs = float(original[name].abs().max())
        # fp16 relative rounding ~2^-11 plus a tolerance for accumulation.
        bound = max(1e-4, max_abs * (2**-10) + 1e-6)
        assert float((original[name] - recovered[name]).abs().max()) <= bound, name


def test_roundtrip_q8_within_quantizer_error(roundtrip_config, tmp_path):
    torch.manual_seed(2)
    model = _build_model(roundtrip_config).eval()
    path = export_to_gguf(
        model,
        tmp_path / "q8.gguf",
        quantize="q8_0",
        model_config=roundtrip_config.model_dump(),
    )

    restored = load_gguf_model(path)
    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())

    for name in original:
        if recovered[name].shape[-1] == 1 and name not in ("lm_head.weight",):
            # scalar-ish rows quantize coarse; give the block error bound
            max_abs = float(original[name].abs().max())
            bound = max(1e-3, max_abs / 254.0 + max_abs / 2048.0 + 1e-4)
        else:
            max_abs = float(original[name].abs().max())
            bound = max(1e-3, max_abs * (2**-11) * 4 + max_abs / 254.0 + 1e-4)
        assert float((original[name] - recovered[name]).abs().max()) <= bound, name


def test_restored_model_forward_runs_and_matches(roundtrip_config, tmp_path):
    torch.manual_seed(3)
    model = _build_model(roundtrip_config).eval()
    path = export_to_gguf(
        model,
        tmp_path / "f32.gguf",
        quantize="f32",
        model_config=roundtrip_config.model_dump(),
    )

    restored = load_gguf_model(path)

    ids = torch.randint(0, roundtrip_config.vocab_size, (1, 8))
    with torch.no_grad():
        a = model(ids)
        b = restored(ids)
    if isinstance(a, tuple):
        a = a[0]
    if isinstance(b, tuple):
        b = b[0]
    assert a.shape == b.shape
    assert torch.equal(a, b), "restored model output differs from the original"


def test_config_persisted_in_metadata(roundtrip_config, tmp_path):
    model = _build_model(roundtrip_config).eval()
    path = export_to_gguf(
        model,
        tmp_path / "meta.gguf",
        quantize="f32",
        model_config=roundtrip_config.model_dump(),
    )
    reader = GGUFReader(path)
    blob = reader.metadata.get("general.llm_model_config")
    assert isinstance(blob, str)
    cfg = ModelConfig.model_validate(json.loads(blob))
    assert cfg.hidden_size == roundtrip_config.hidden_size
    assert cfg.num_layers == roundtrip_config.num_layers
    assert cfg.use_rope == roundtrip_config.use_rope
    assert cfg.qkv_bias == roundtrip_config.qkv_bias


def test_missing_config_is_refused(tmp_path):
    """A GGUF with no config blob (e.g. a third-party file) is rejected."""
    writer = GGUFWriter(tmp_path / "foreign.gguf")
    writer.add_metadata("general.architecture", "llama")
    writer.add_tensor("some.weight", np.ones((4, 4), dtype=np.float32), ggml_type="f32")
    path = writer.write()

    with pytest.raises(GGUFError, match=r"general\.llm_model_config"):
        load_gguf_model(path)


def test_bad_config_json_is_refused(tmp_path):
    writer = GGUFWriter(tmp_path / "badcfg.gguf")
    writer.add_metadata("general.llm_model_config", "{not json")
    writer.add_tensor("some.weight", np.ones((4, 4), dtype=np.float32), ggml_type="f32")
    path = writer.write()

    with pytest.raises(GGUFError, match="invalid 'general\\.llm_model_config'"):
        load_gguf_model(path)
