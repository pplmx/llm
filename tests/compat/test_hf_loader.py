"""Tests for HuggingFace Compatibility module."""

import json

import pytest
import torch

from llm.compat.weight_mapping import (
    ARCHITECTURE_MAPPINGS,
    convert_hf_weights,
    detect_architecture,
    expand_layer_mapping,
    get_config_mapping,
    get_weight_mapping,
)
from tests.support.devices import DEFAULT_DEVICE


class TestWeightMapping:
    """Tests for weight mapping utilities."""

    def test_architecture_mappings_exist(self):
        """Test that all expected architectures have mappings."""
        assert "llama" in ARCHITECTURE_MAPPINGS
        assert "mistral" in ARCHITECTURE_MAPPINGS
        assert "qwen" in ARCHITECTURE_MAPPINGS
        assert "qwen2" in ARCHITECTURE_MAPPINGS

    def test_get_weight_mapping(self):
        """Test getting weight mapping for architecture."""
        mapping = get_weight_mapping("llama")

        assert "model.embed_tokens.weight" in mapping
        assert "lm_head.weight" in mapping
        assert "model.norm.weight" in mapping

    def test_expand_layer_mapping(self):
        """Test expanding layer placeholders."""
        base_mapping = {
            "model.layers.{layer}.attn.weight": "blocks.{layer}.attn.weight",
            "model.norm.weight": "final_norm.weight",
        }

        expanded = expand_layer_mapping(base_mapping, num_layers=3)

        assert "model.layers.0.attn.weight" in expanded
        assert "model.layers.1.attn.weight" in expanded
        assert "model.layers.2.attn.weight" in expanded
        assert "model.norm.weight" in expanded

    def test_detect_architecture_llama(self):
        """Test architecture detection for Llama."""
        config = {"model_type": "llama"}
        assert detect_architecture(config) == "llama"

        config = {"model_type": "LlamaForCausalLM"}
        assert detect_architecture(config) == "llama"

    def test_detect_architecture_mistral(self):
        """Test architecture detection for Mistral."""
        config = {"model_type": "mistral"}
        assert detect_architecture(config) == "mistral"

    def test_detect_architecture_mixtral_is_distinct(self):
        """RIL ISS-144: Mixtral must be detected as its OWN architecture (not
        collapsed into dense 'mistral'), so the loader can reject it instead
        of silently building a dense model and dropping every MoE tensor."""
        config = {"model_type": "MixtralForCausalLM"}
        assert detect_architecture(config) == "mixtral"

    def test_detect_architecture_qwen(self):
        """Test architecture detection for Qwen."""
        config = {"model_type": "qwen"}
        assert detect_architecture(config) == "qwen"

        config = {"model_type": "Qwen2ForCausalLM"}
        assert detect_architecture(config) == "qwen2"

    def test_detect_architecture_qwen_moe_and_qwen3_refused_as_unknown(self):
        """Qwen2MoE / Qwen3(MoE) must NOT collapse onto the dense qwen2 or
        qwen1 mappings (which would silently drop every expert / most-model
        tensor and run from RANDOM init with warnings only). They must route
        to "unknown" so from_pretrained refuses loudly (iss-144 / round-71
        anti-garbage-load philosophy)."""
        for model_type in ("qwen2moe", "qwen2_moe", "qwen3", "qwen3moe", "qwen3_moe"):
            config = {"model_type": model_type}
            assert detect_architecture(config) == "unknown", model_type

    def test_detect_architecture_unknown_is_distinct(self):
        """Round-71 compat fix: an unsupported model_type (gpt2, gemma,
        baichuan, ...) must NOT collapse into the llama mapping — the loader
        previously loaded every weight at RANDOM init with only warnings.
        It must be distinguishable so from_pretrained can refuse it."""
        config = {"model_type": "gpt2"}
        assert detect_architecture(config) == "unknown"
        config = {"model_type": "gemma"}
        assert detect_architecture(config) == "unknown"
        config = {}  # no model_type at all
        assert detect_architecture(config) == "unknown"

    def test_convert_hf_weights(self):
        """Test weight conversion."""
        hf_state_dict = {
            "model.embed_tokens.weight": torch.randn(1000, 64),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(1000, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
        }

        converted = convert_hf_weights(hf_state_dict, architecture="llama", num_layers=1)

        assert "embedding_layer.token_embeddings.weight" in converted
        assert "final_norm.weight" in converted
        assert "lm_head.weight" in converted
        # The mapping writes to ``self_attn.q_proj`` (our model's
        # attribute name); the concat helper then combines q/k/v into
        # ``qkv_proj`` (see ``convert_hf_to_combined_qkv``).
        assert "transformer_blocks.0.self_attn.q_proj.weight" in converted
        assert "transformer_blocks.0.self_attn.k_proj.weight" in converted

    def test_get_config_mapping(self):
        """Test config mapping from HF format."""
        hf_config = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 4096,
        }

        our_config = get_config_mapping(hf_config)

        assert our_config["vocab_size"] == 32000
        assert our_config["hidden_size"] == 4096
        assert our_config["num_layers"] == 32
        assert our_config["num_heads"] == 32
        assert our_config["num_kv_heads"] == 8

    def test_get_config_mapping_rope_defaults_on_for_external(self):
        """External real-Llama/Mistral checkpoints always use RoPE — their HF
        configs carry ``rope_theta`` and no ``use_rope`` key.  Mapping such a
        config must default ``use_rope=True`` (RIL ISS-062), and a persisted
        ``use_rope`` value (from our own save_pretrained) must be honored."""
        # typical external HF Llama/Mistral config: no use_rope key
        external = get_config_mapping(
            {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "rope_theta": 500000.0,
            }
        )
        assert external["use_rope"] is True
        assert external["rope_theta"] == 500000.0

        # our own publisher always persists the flag so roundtrips stay exact
        ours_non_rope = get_config_mapping({"model_type": "llama", "use_rope": False})
        assert ours_non_rope["use_rope"] is False

    def test_get_config_mapping_bias_free_default_for_external(self):
        """Real Llama/Mistral checkpoints are bias-free (``attention_bias``
        defaults to False); mapping an external config (no bias keys) must
        default all of qkv/mlp/lm_head bias off so random biases don't leak
        in (RIL ISS-062). Persisted values from our own publisher are honored."""
        external = get_config_mapping({"model_type": "llama", "rope_theta": 500000.0})
        assert external["qkv_bias"] is False
        assert external["mlp_bias"] is False
        assert external["lm_head_bias"] is False

        ours_biased = get_config_mapping(
            {"model_type": "llama", "qkv_bias": True, "mlp_bias": True, "lm_head_bias": True}
        )
        assert ours_biased["qkv_bias"] is True
        assert ours_biased["lm_head_bias"] is True

    def test_get_config_mapping_honors_hf_attention_bias_for_external(self):
        """RIL ISS-145: external checkpoints declare attention bias under HF's
        canonical ``attention_bias`` key (Qwen-style). Mapping that key must
        turn qkv/mlp/lm_head bias ON — before this, only our repo-custom keys
        were read and every bias was silently dropped (bias-free default)."""
        external = get_config_mapping(
            {
                "model_type": "qwen2",
                "hidden_size": 2048,
                "attention_bias": True,
            }
        )
        assert external["qkv_bias"] is True
        assert external["mlp_bias"] is True
        assert external["lm_head_bias"] is True

        # Our own persisted flags still take precedence over the HF key.
        ours = get_config_mapping(
            {"model_type": "llama", "qkv_bias": False, "mlp_bias": False, "lm_head_bias": False, "attention_bias": True}
        )
        assert ours["qkv_bias"] is False


class TestHFLoader:
    """Tests for HuggingFace loader."""

    @pytest.fixture
    def mock_hf_model_dir(self, tmp_path):
        """Create a mock HuggingFace model directory."""
        # Create config.json
        config = {
            "model_type": "llama",
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 64,
            "torch_dtype": "float32",
        }
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f)

        # Create mock weights (as .bin file)
        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 32),
            "model.norm.weight": torch.randn(32),
            "lm_head.weight": torch.randn(100, 32),
        }
        # Add layer weights
        for layer in range(2):
            state_dict[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(32, 32)
            state_dict[f"model.layers.{layer}.self_attn.k_proj.weight"] = torch.randn(32, 32)
            state_dict[f"model.layers.{layer}.self_attn.v_proj.weight"] = torch.randn(32, 32)
            state_dict[f"model.layers.{layer}.self_attn.o_proj.weight"] = torch.randn(32, 32)
            state_dict[f"model.layers.{layer}.mlp.gate_proj.weight"] = torch.randn(128, 32)
            state_dict[f"model.layers.{layer}.mlp.up_proj.weight"] = torch.randn(128, 32)
            state_dict[f"model.layers.{layer}.mlp.down_proj.weight"] = torch.randn(32, 128)
            state_dict[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(32)
            state_dict[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.randn(32)

        weight_path = tmp_path / "pytorch_model.bin"
        torch.save(state_dict, weight_path)

        return tmp_path

    def test_from_pretrained_ties_lm_head_to_embeddings_when_key_absent(self, tmp_path):
        """RIL ISS-143: an external tied-embedding checkpoint (standard HF
        layout, ``tie_word_embeddings`` with NO ``lm_head.weight``) must copy
        the input embeddings into the LM head — DecoderModel always allocates
        a separate linear head, so before this fix a tied checkpoint loaded
        cleanly (Missing-keys warning only) with the head at RANDOM init,
        producing garbage generations."""
        import json

        from llm.compat.hf_publisher import save_pretrained
        from tests.support.models import decoder_model_kwargs

        # Build + publish a small model, then MUTATE the artifact into a tied
        # external checkpoint: drop lm_head.weight and set tie_word_embeddings.
        torch.manual_seed(1)
        from llm.models.decoder import DecoderModel

        src = DecoderModel(
            **decoder_model_kwargs(
                vocab_size=64,
                hidden_size=32,
                num_layers=1,
                num_heads=2,
                intermediate_size=64,
                max_seq_len=64,
                attn_impl="mha",
                mlp_impl="mlp",
            )
        )
        save_pretrained(src, tmp_path)

        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        config["tie_word_embeddings"] = True
        config_path.write_text(json.dumps(config))

        weights_path = tmp_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import save_file

            from llm.compat.hf_loader import from_pretrained

            state_dict = dict(torch.load(str(tmp_path / "model.safetensors")))
            state_dict.pop("lm_head.weight", None)  # tied layout has no head
            save_file(state_dict, str(weights_path))

            reloaded = from_pretrained(str(tmp_path), device=str(DEFAULT_DEVICE), dtype=torch.float32)
            head = reloaded.lm_head.weight.detach().to("cpu")
            emb = reloaded.embedding_layer.token_embeddings.weight.detach().to("cpu")
            assert torch.equal(head, emb), "tied checkpoint must copy embeddings into lm_head"

    def test_list_supported_architectures(self):
        """Test listing supported architectures."""
        from llm.compat.hf_loader import list_supported_architectures

        archs = list_supported_architectures()

        assert "llama" in archs
        assert "mistral" in archs
        assert "qwen" in archs

    def test_mixtral_is_not_advertised_or_loadable(self, tmp_path):
        """RIL ISS-144: Mixtral (block-sparse MoE) is NOT silently loadable.
        Before the fix ``list_supported_architectures`` advertised it but the
        loader built a DENSE model and dropped every expert/router tensor at
        random init (garbage output, warnings only). It must be absent from
        the advertised list AND rejected by from_pretrained with a clear
        error."""
        import json

        from llm.compat.hf_loader import from_pretrained, list_supported_architectures

        assert "mixtral" not in list_supported_architectures()

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral", "num_experts": 8, "top_k": 2}))
        with pytest.raises(NotImplementedError, match=r"[Mm]ixtral"):
            from_pretrained(str(tmp_path), device=str(DEFAULT_DEVICE), dtype=torch.float32)

    def test_unknown_arch_is_not_loadable(self, tmp_path):
        """Round-71 compat fix: a config with an unsupported model_type (gpt2,
        gemma, ...) must be refused by from_pretrained. Before the fix the
        loader defaulted to the llama mapping and loaded every unmapped weight
        at RANDOM init — garbage generation with warnings only."""
        import json

        from llm.compat.hf_loader import from_pretrained

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
        with pytest.raises(NotImplementedError, match="not supported"):
            from_pretrained(str(tmp_path), device=str(DEFAULT_DEVICE), dtype=torch.float32)

    def test_load_weights_missing_raises(self, tmp_path):
        """Test that missing weights raises error."""
        from llm.compat.hf_loader import _load_weights

        with pytest.raises(FileNotFoundError):
            _load_weights(tmp_path)

    def test_load_from_hub_excludes_bin_files(self, tmp_path, monkeypatch):
        """Hub downloads must skip ``*.bin`` because they are pickled and execute
        arbitrary code on load. Only ``*.json`` and ``*.safetensors`` are allowed.

        This is a regression test for Finding AR in the technical due diligence.
        """
        import sys

        from llm.compat import hf_loader

        captured_kwargs: dict = {}

        class _FakeHub:
            @staticmethod
            def snapshot_download(repo_id, **kwargs):
                captured_kwargs.update(kwargs)
                # Return a path with a config so _load_from_local gets past the
                # first check, but we'll stub _load_from_local to raise immediately
                # so we don't actually build a model.
                return str(tmp_path)

        sys.modules["huggingface_hub"] = _FakeHub  # type: ignore[assignment]
        try:

            def fake_load_local(*_args, **_kwargs):
                raise RuntimeError("stop_after_download")

            monkeypatch.setattr(hf_loader, "_load_from_local", fake_load_local)

            with pytest.raises(RuntimeError, match="stop_after_download"):
                hf_loader._load_from_hub("fake/model", "cpu", None, False)
        finally:
            sys.modules.pop("huggingface_hub", None)

        assert "allow_patterns" in captured_kwargs, "snapshot_download was called without allow_patterns"
        patterns = captured_kwargs["allow_patterns"]
        assert isinstance(patterns, list)
        # Must include config + safetensors
        assert "*.json" in patterns
        assert "*.safetensors" in patterns
        # Must NOT include .bin
        assert "*.bin" not in patterns, f"Hub downloads must skip .bin files (pickle RCE); got patterns={patterns}"


class TestGLURoleMapping:
    """Regression: HF gate_proj/up_proj roles must map by *function*.

    Our GLU MLP computes ``fc2(act(fc1(x)) * gate_proj(x))`` — ``fc1`` is
    the activated (gate) role, ``gate_proj`` the raw multiplier (up) role.
    HF Llama computes ``down(act(gate_proj(x)) * up_proj(x))`` — ``gate_proj``
    is activated.  Loading real Llama/Mistral weights with the old
    name-based mapping swapped the two, so the loaded model computed
    ``silu(up_proj(x)) * gate_proj(x)``, a different function for any real
    checkpoint (the roundtrip save→load tests miss this because they are
    symmetric: both sides used the same wrong mapping).
    """

    def _make_glu_model(self):
        from llm.models.decoder import DecoderModel
        from tests.support.models import decoder_model_kwargs

        kwargs = decoder_model_kwargs(
            vocab_size=64,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            max_seq_len=16,
            attn_impl="mha",
            mlp_impl="mlp",
            mlp_activation="silu",
            use_glu=True,
            device="cpu",
        )
        return DecoderModel(**kwargs)

    def test_hf_glu_roles_map_to_function_not_name(self):
        """HF ``gate_proj`` tensors must land on our *activated* ``fc1``.

        A real Llama state dict stores the gated (activated) projection in
        ``gate_proj`` and the raw multiplier in ``up_proj``.  After
        ``convert_hf_weights`` the loaded model must compute the same
        function as HF's ``down(silu(gate_proj(x)) * up_proj(x))``.  The old
        name-based mapping placed ``gate_proj`` on our non-activated
        ``gate_proj``, so for distinct G vs U tensors the MLP silently
        computed a different function.

        We hand-place distinct tensors at their *HF* names (as a real
        checkpoint would carry them) and verify where they land once converted
        to our namespace and run through the model.  Going through
        ``convert_our_weights`` would re-use the same mapping and stay
        symmetric — masking the defect the way the save→load roundtrip does.
        """
        from llm.compat.weight_mapping import convert_hf_weights, convert_our_weights

        model = self._make_glu_model()
        model.eval()
        mlp = model.transformer_blocks[0].mlp

        # Distinct gate vs up tensors at their REAL HF positions.
        torch.manual_seed(123)
        gate_w = torch.randn(32, 16)  # HF gate_proj weight (gated/activated)
        up_w = torch.randn(32, 16)  # HF up_proj weight (raw multiplier)
        gate_b = torch.randn(32)
        up_b = torch.randn(32)
        down_w = mlp.fc2.weight.detach()
        down_b = mlp.fc2.bias.detach()
        x = torch.randn(3, 16)

        import torch.nn.functional as functional

        hf_out = functional.linear(
            functional.silu(functional.linear(x, gate_w, gate_b)) * functional.linear(x, up_w, up_b),
            down_w,
            down_b,
        )

        # Build the HF state dict with G / U at their real HF names (the rest
        # of the weights come from our model, renamed as the publisher does).
        our_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hf_sd = convert_our_weights(our_sd, architecture="llama", num_layers=1)
        hf_sd["model.layers.0.mlp.gate_proj.weight"] = gate_w
        hf_sd["model.layers.0.mlp.gate_proj.bias"] = gate_b
        hf_sd["model.layers.0.mlp.up_proj.weight"] = up_w
        hf_sd["model.layers.0.mlp.up_proj.bias"] = up_b

        # Convert to our namespace exactly as from_pretrained does.
        back = convert_hf_weights(hf_sd, architecture="llama", num_layers=1)

        # HF's gate (activated) tensor must land on our fc1 (activated);
        # HF's up (raw) tensor on our gate_proj (raw).
        assert torch.allclose(back["transformer_blocks.0.mlp.fc1.weight"], gate_w)
        assert torch.allclose(back["transformer_blocks.0.mlp.gate_proj.weight"], up_w)

        loaded = self._make_glu_model()
        missing, unexpected = loaded.load_state_dict(back, strict=False)
        assert unexpected == []
        assert "transformer_blocks.0.mlp.fc2.weight" not in missing
        loaded.eval()
        with torch.no_grad():
            loaded_out = loaded.transformer_blocks[0].mlp(x)

        assert torch.allclose(loaded_out, hf_out, atol=1e-5), (
            "Loaded MLP diverges from HF's silu(gate_proj)*up_proj — gate_proj/up_proj roles are swapped."
        )
