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

        config = {"model_type": "MixtralForCausalLM"}
        assert detect_architecture(config) == "mistral"

    def test_detect_architecture_qwen(self):
        """Test architecture detection for Qwen."""
        config = {"model_type": "qwen"}
        assert detect_architecture(config) == "qwen"

        config = {"model_type": "Qwen2ForCausalLM"}
        assert detect_architecture(config) == "qwen2"

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
        assert "transformer_blocks.0.attn.q_proj.weight" in converted

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

    def test_list_supported_architectures(self):
        """Test listing supported architectures."""
        from llm.compat.hf_loader import list_supported_architectures

        archs = list_supported_architectures()

        assert "llama" in archs
        assert "mistral" in archs
        assert "qwen" in archs

    def test_load_weights_missing_raises(self, tmp_path):
        """Test that missing weights raises error."""
        from llm.compat.hf_loader import _load_weights

        with pytest.raises(FileNotFoundError):
            _load_weights(tmp_path)
