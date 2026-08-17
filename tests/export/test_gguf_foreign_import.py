"""Foreign llama.cpp GGUF import milestone (round 72).

``load_gguf_model`` now imports GGUF files that are NOT self-exports —
standard llama.cpp files carrying ``general.architecture`` + ``llama.*``
metadata and llama-style tensor names (``token_embd`` / ``blk.N.attn_*`` /
``blk.N.ffn_*`` / ``output_norm`` / ``output``). The metadata rebuilds a
:class:`ModelConfig` and ``llm.compat.weight_mapping.convert_gguf_weights``
maps the tensor names into llm state-dict naming (reusing the same q/k/v
fusion and tied-head fallback as ``from_pretrained``).

The fixtures are synthetic "llama.cpp" files written with GGUFWriter from a
trained llm model's state dict renamed to the canonical llama.cpp tensor
names — the exact layout a real llama.cpp file has, but tiny and
deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm.export.gguf import GGUFWriter, load_gguf_model
from llm.export.gguf.spec import GGUFError
from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig


@pytest.fixture
def llama_config() -> ModelConfig:
    """Tiny dense Llama-style decoder — mirrors a real llama.cpp model."""
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        use_rope=True,
        rope_theta=10000.0,
    )


@pytest.fixture
def llama_config_gqa() -> ModelConfig:
    """Same shape with grouped-query attention (head_count > head_count_kv)."""
    return ModelConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        use_rope=True,
        rope_theta=10000.0,
    )


def _build_model(cfg: ModelConfig, *, norm_eps: float = 1e-5) -> torch.nn.Module:
    ensure_builtins_registered()
    return ModelFactory.from_config(cfg, norm_eps=norm_eps).eval()


def _llama_cpp_metadata(cfg: ModelConfig, prefix: str = "llama") -> dict:
    """The llama.cpp metadata a converter would write for ``cfg``.

    ``prefix`` is the ``general.architecture`` value — gguf-py prefixes every
    model key with it (``llama.*`` for Llama, ``qwen2.*`` for Qwen2, ...).
    """
    return {
        "general.architecture": prefix,
        "general.name": f"tiny-{prefix}",
        "general.file_type": 0,
        "general.quantization_version": 2,
        f"{prefix}.context_length": cfg.max_seq_len,
        f"{prefix}.embedding_length": cfg.hidden_size,
        f"{prefix}.block_count": cfg.num_layers,
        f"{prefix}.feed_forward_length": cfg.intermediate_size,
        f"{prefix}.attention.head_count": cfg.num_heads,
        f"{prefix}.attention.head_count_kv": cfg.num_kv_heads or cfg.num_heads,
        f"{prefix}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{prefix}.rope.freq_base": cfg.rope_theta,
        f"{prefix}.vocab_size": cfg.vocab_size,
    }


def _llama_cpp_tensors(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Rename a model's state dict into canonical llama.cpp GGUF names.

    This is the inverse of ``GGUF_MAPPING``: it splits the combined
    ``qkv_proj`` back into ``attn_q`` / ``attn_k`` / ``attn_v`` (the layout a
    llama.cpp converter writes) and swaps the MLP role names.
    """
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {}
    head_dim = model.transformer_blocks[0].self_attn.head_dim
    kv_dim = model.transformer_blocks[0].self_attn.kv_dim
    q_size = model.num_heads * head_dim

    out["token_embd.weight"] = sd["embedding_layer.token_embeddings.weight"]
    out["output_norm.weight"] = sd["final_norm.weight"]
    out["output.weight"] = sd["lm_head.weight"]

    for i in range(len(model.transformer_blocks)):
        p = f"transformer_blocks.{i}"
        out[f"blk.{i}.attn_norm.weight"] = sd[f"{p}.norm1.weight"]
        out[f"blk.{i}.ffn_norm.weight"] = sd[f"{p}.norm2.weight"]
        out[f"blk.{i}.attn_output.weight"] = sd[f"{p}.self_attn.out_proj.weight"]
        out[f"blk.{i}.ffn_gate.weight"] = sd[f"{p}.mlp.fc1.weight"]
        out[f"blk.{i}.ffn_up.weight"] = sd[f"{p}.mlp.gate_proj.weight"]
        out[f"blk.{i}.ffn_down.weight"] = sd[f"{p}.mlp.fc2.weight"]
        qkv = sd[f"{p}.self_attn.qkv_proj.weight"]
        out[f"blk.{i}.attn_q.weight"] = qkv[:q_size]
        out[f"blk.{i}.attn_k.weight"] = qkv[q_size : q_size + kv_dim]
        out[f"blk.{i}.attn_v.weight"] = qkv[q_size + kv_dim :]

    return out


def _write_foreign_gguf(
    tmp_path,
    cfg: ModelConfig,
    tensors: dict[str, np.ndarray],
    *,
    metadata: dict | None = None,
    quantize: str = "f32",
    drop: set[str] | None = None,
) -> str:
    path = tmp_path / "foreign.gguf"
    writer = GGUFWriter(path)
    for key, value in (metadata if metadata is not None else _llama_cpp_metadata(cfg)).items():
        writer.add_metadata(key, value)
    for name, arr in tensors.items():
        if drop and name in drop:
            continue
        writer.add_tensor(name, arr, ggml_type=quantize)
    return str(writer.write())


def _state_dict_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu() for k, v in state.items()}


def _forward(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(ids)
    return out[0] if isinstance(out, tuple) else out


def test_foreign_import_f32_matches(llama_config, tmp_path):
    torch.manual_seed(0)
    model = _build_model(llama_config)
    path = _write_foreign_gguf(tmp_path, llama_config, _llama_cpp_tensors(model))

    restored = load_gguf_model(path)

    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())
    assert set(original) == set(recovered)
    for name in original:
        assert torch.equal(original[name], recovered[name]), name

    ids = torch.randint(0, llama_config.vocab_size, (1, 8))
    assert torch.equal(_forward(model, ids), _forward(restored, ids))


def test_foreign_import_q8_within_quantizer_error(tmp_path):
    # Block quantization needs the last dim (hidden/intermediate) as a multiple
    # of 32, exactly like a real llama.cpp model — the tiny 16-wide fixture is
    # only usable for f32/f16.
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        use_rope=True,
        rope_theta=10000.0,
    )
    torch.manual_seed(1)
    model = _build_model(cfg)
    path = _write_foreign_gguf(tmp_path, cfg, _llama_cpp_tensors(model), quantize="q8_0")

    restored = load_gguf_model(path)
    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())

    for name in original:
        max_abs = float(original[name].abs().max())
        # q8_0 block quantizer: 7-bit mantissa scale + int8 body (same bound as
        # the self-export roundtrip test).
        bound = max(1e-3, max_abs * (2**-11) * 4 + max_abs / 254.0 + 1e-4)
        assert float((original[name] - recovered[name]).abs().max()) <= bound, name


def test_foreign_import_gqa_matches(llama_config_gqa, tmp_path):
    torch.manual_seed(2)
    model = _build_model(llama_config_gqa)
    path = _write_foreign_gguf(tmp_path, llama_config_gqa, _llama_cpp_tensors(model))

    restored = load_gguf_model(path)

    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())
    for name in original:
        assert torch.equal(original[name], recovered[name]), name

    ids = torch.randint(0, llama_config_gqa.vocab_size, (1, 8))
    assert torch.equal(_forward(model, ids), _forward(restored, ids))


def test_foreign_import_tied_head(llama_config, tmp_path):
    """llama.cpp omits ``output.weight`` when the head is tied to embeddings."""
    torch.manual_seed(3)
    model = _build_model(llama_config)
    path = _write_foreign_gguf(
        tmp_path,
        llama_config,
        _llama_cpp_tensors(model),
        drop={"output.weight"},
    )

    restored = load_gguf_model(path)

    # Tied head: lm_head must equal the input embeddings (copied, not random).
    assert torch.equal(
        restored.lm_head.weight.detach().float(),
        restored.embedding_layer.token_embeddings.weight.detach().float(),
    )
    # And it must match the source model, where we tie the head by hand.
    with torch.no_grad():
        model.lm_head.weight.copy_(model.embedding_layer.token_embeddings.weight)
    ids = torch.randint(0, llama_config.vocab_size, (1, 8))
    assert torch.equal(_forward(model, ids), _forward(restored, ids))


def test_foreign_import_non_llama_arch_refused(tmp_path):
    model = _build_model(
        _llama_config := ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            max_seq_len=16,
            use_glu=True,
            mlp_activation="silu",
            norm_impl="rms_norm",
            use_rope=True,
        )
    )
    metadata = dict(_llama_cpp_metadata(_llama_config))
    metadata["general.architecture"] = "mixtral"  # MoE — dense mapping can't serve it
    path = _write_foreign_gguf(tmp_path, _llama_config, _llama_cpp_tensors(model), metadata=metadata)

    with pytest.raises(GGUFError, match=r"unsupported GGUF architecture 'mixtral'"):
        load_gguf_model(path)


def test_foreign_import_missing_metadata_refused(tmp_path):
    model = _build_model(
        _cfg := ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            max_seq_len=16,
            use_glu=True,
            mlp_activation="silu",
            norm_impl="rms_norm",
            use_rope=True,
        )
    )
    metadata = dict(_llama_cpp_metadata(_cfg))
    del metadata["llama.vocab_size"]
    path = _write_foreign_gguf(tmp_path, _cfg, _llama_cpp_tensors(model), metadata=metadata)

    with pytest.raises(GGUFError, match=r"missing required llama\.cpp metadata: llama\.vocab_size"):
        load_gguf_model(path)


def test_foreign_import_unmapped_tensor_refused(llama_config, tmp_path):
    model = _build_model(llama_config)
    tensors = _llama_cpp_tensors(model)
    tensors["foo.extra.weight"] = np.zeros((4, 4), dtype=np.float32)  # no GGUF mapping
    path = _write_foreign_gguf(tmp_path, llama_config, tensors)

    with pytest.raises(GGUFError, match=r"tensor\(s\) have no mapping"):
        load_gguf_model(path)


def test_foreign_import_rope_scaling_refused(llama_config, tmp_path):
    """RoPE scaling (YaRN/linear) is not supported — refuse, don't import wrong."""
    torch.manual_seed(4)
    model = _build_model(llama_config)
    metadata = dict(_llama_cpp_metadata(llama_config))
    metadata["llama.rope.scaling.type"] = "yarn"
    path = _write_foreign_gguf(tmp_path, llama_config, _llama_cpp_tensors(model), metadata=metadata)

    with pytest.raises(GGUFError, match=r"RoPE scaling type 'yarn' is not supported"):
        load_gguf_model(path)


def test_foreign_import_qwen2_prefix(llama_config, tmp_path):
    """Qwen2 files carry ``qwen2.*`` metadata (arch-prefixed, NOT ``llama.*``).

    The key prefix must come from ``general.architecture`` (gguf-py writes
    ``{arch}.*`` keys), otherwise a real Qwen2 file is refused or its norm_eps
    silently defaults wrong (round-73 review HIGH).
    """
    cfg = llama_config.model_copy(update={"hidden_size": 32, "num_heads": 4, "intermediate_size": 64})
    torch.manual_seed(5)
    model = _build_model(cfg, norm_eps=1e-6)  # Qwen2's real RMSNorm eps
    metadata = _llama_cpp_metadata(cfg, prefix="qwen2")
    metadata["qwen2.attention.layer_norm_rms_epsilon"] = 1e-6
    path = _write_foreign_gguf(tmp_path, cfg, _llama_cpp_tensors(model), metadata=metadata)

    restored = load_gguf_model(path)

    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())
    for name in original:
        assert torch.equal(original[name], recovered[name]), name
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert torch.equal(_forward(model, ids), _forward(restored, ids))


def test_foreign_import_mistral_prefix(llama_config, tmp_path):
    """Mistral files carry ``mistral.*`` metadata."""
    torch.manual_seed(6)
    model = _build_model(llama_config)
    metadata = _llama_cpp_metadata(llama_config, prefix="mistral")
    path = _write_foreign_gguf(tmp_path, llama_config, _llama_cpp_tensors(model), metadata=metadata)

    restored = load_gguf_model(path)

    ids = torch.randint(0, llama_config.vocab_size, (1, 8))
    assert torch.equal(_forward(model, ids), _forward(restored, ids))


def test_foreign_import_multilayer(tmp_path):
    """{{layer}} expansion: hidden/attention/ffn tensors for every block."""
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        use_rope=True,
        rope_theta=10000.0,
    )
    torch.manual_seed(7)
    model = _build_model(cfg)
    path = _write_foreign_gguf(tmp_path, cfg, _llama_cpp_tensors(model))

    restored = load_gguf_model(path)

    original = _state_dict_tensors(model.state_dict())
    recovered = _state_dict_tensors(restored.state_dict())
    assert set(original) == set(recovered)
    for name in original:
        assert torch.equal(original[name], recovered[name]), name


def test_foreign_import_undividable_kv_refused(llama_config, tmp_path):
    """head_count not divisible by head_count_kv is not representable by MHA.

    The source model is divisible (2/2); the file metadata claims 3/2, which
    the loader rejects on metadata before weights are ever read.
    """
    torch.manual_seed(8)
    model = _build_model(llama_config)
    metadata = dict(_llama_cpp_metadata(llama_config))
    metadata["llama.attention.head_count"] = 4  # divides hidden (16)
    metadata["llama.attention.head_count_kv"] = 3  # but 4 % 3 != 0
    path = _write_foreign_gguf(tmp_path, llama_config, _llama_cpp_tensors(model), metadata=metadata)

    with pytest.raises(GGUFError, match=r"head_count 4 not divisible by head_count_kv 3"):
        load_gguf_model(path)


def test_foreign_import_nonstandard_head_dim_refused(tmp_path):
    """key_length != hidden//heads (non-standard head dim) is rejected."""
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        use_rope=True,
        rope_theta=10000.0,
    )
    torch.manual_seed(9)
    model = _build_model(cfg)
    metadata = dict(_llama_cpp_metadata(cfg))
    metadata["llama.attention.key_length"] = 4  # != head_dim 8
    path = _write_foreign_gguf(tmp_path, cfg, _llama_cpp_tensors(model), metadata=metadata)

    with pytest.raises(GGUFError, match=r"non-standard head dimensions"):
        load_gguf_model(path)
