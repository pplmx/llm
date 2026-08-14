"""Tests for :class:`llm.serving.config.ServingConfig`.

Extends the existing smoke tests with coverage for PEFT validators,
YAML loading, env var prefix, security defaults, and merge guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm.serving.config import ServingConfig


class TestServingConfigDefaults:
    def test_default_values(self):
        cfg = ServingConfig()
        assert cfg.model_path is None
        assert cfg.tokenizer_path is None
        assert cfg.tokenizer_type == "simple"
        assert cfg.device == "auto"
        assert cfg.api_key is None
        assert cfg.log_level == "INFO"
        assert cfg.host == "127.0.0.1"
        assert cfg.generation_backend == "eager"
        assert cfg.compile_model is False
        assert cfg.max_concurrent_requests == 4
        assert cfg.request_timeout == 60.0
        assert cfg.use_paged_attention is False
        assert cfg.max_blocks == 256
        assert cfg.block_size == 16
        assert cfg.hidden_size == 64
        assert cfg.num_layers == 2

    def test_peft_defaults_off(self):
        cfg = ServingConfig()
        assert cfg.peft_method is None
        assert cfg.peft_kwargs == {}
        assert cfg.peft_adapter_path is None
        assert cfg.peft_merge is False


class TestServingConfigFromYaml:
    def test_loads_basic_fields(self, tmp_path: Path):
        yml = tmp_path / "serve.yaml"
        yml.write_text(
            yaml.dump(
                {
                    "model_path": "/fake/ckpt.pt",
                    "tokenizer_type": "hf",
                    "device": "cpu",
                    "host": "0.0.0.0",  # noqa: S104
                }
            )
        )
        cfg = ServingConfig.from_yaml(yml)
        assert cfg.model_path == "/fake/ckpt.pt"
        assert cfg.tokenizer_type == "hf"
        assert cfg.device == "cpu"
        assert cfg.host == "0.0.0.0"  # noqa: S104

    def test_loads_peft_fields(self, tmp_path: Path):
        yml = tmp_path / "serve.yaml"
        yml.write_text(
            yaml.dump(
                {
                    "peft_method": "lora",
                    "peft_kwargs": {"rank": 8},
                    "peft_adapter_path": "/fake/lora.bin",
                    "peft_merge": True,
                }
            )
        )
        cfg = ServingConfig.from_yaml(yml)
        assert cfg.peft_method == "lora"
        assert cfg.peft_kwargs == {"rank": 8}
        assert cfg.peft_adapter_path == "/fake/lora.bin"
        assert cfg.peft_merge is True

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        cfg = ServingConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert cfg.model_path is None

    def test_empty_yaml_returns_defaults(self, tmp_path: Path):
        yml = tmp_path / "serve.yaml"
        yml.write_text("")
        cfg = ServingConfig.from_yaml(yml)
        assert cfg.model_path is None


class TestServingConfigPeftValidation:
    def test_valid_peft_method_accepted(self):
        cfg = ServingConfig(peft_method="lora")
        assert cfg.peft_method == "lora"

    def test_unknown_peft_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown PEFT method"):
            ServingConfig(peft_method="nonexistent_method_xyz")

    def test_peft_adapter_path_requires_method(self):
        with pytest.raises(ValueError, match="peft_adapter_path is set but peft_method is None"):
            ServingConfig(peft_adapter_path="/fake/adapter.bin")

    def test_peft_kwargs_requires_method(self):
        with pytest.raises(ValueError, match="peft_kwargs is set but peft_method is None"):
            ServingConfig(peft_kwargs={"rank": 8})

    def test_merge_refused_for_bitfit(self):
        with pytest.raises(ValueError, match="peft_merge=True is not supported"):
            ServingConfig(peft_method="bitfit", peft_merge=True)

    def test_merge_refused_for_qlora(self):
        with pytest.raises(ValueError, match="peft_merge=True is not supported"):
            ServingConfig(peft_method="qlora", peft_merge=True)

    def test_merge_refused_for_prefix_tuning(self):
        with pytest.raises(ValueError, match="peft_merge=True is not supported"):
            ServingConfig(peft_method="prefix_tuning", peft_merge=True)

    def test_merge_allowed_for_lora(self):
        cfg = ServingConfig(peft_method="lora", peft_merge=True)
        assert cfg.peft_merge is True

    def test_merge_allowed_for_ia3(self):
        cfg = ServingConfig(peft_method="ia3", peft_merge=True)
        assert cfg.peft_merge is True


class TestServingConfigTokenizers:
    def test_tokenizer_type_pattern_validates(self):
        ServingConfig(tokenizer_type="simple")
        ServingConfig(tokenizer_type="hf")

    def test_tokenizer_type_pattern_rejects_unknown(self):
        with pytest.raises(ValueError, match="pattern"):
            ServingConfig(tokenizer_type="bpe")

    def test_tokenizer_type_pattern_rejects_empty(self):
        with pytest.raises(ValueError, match="pattern"):
            ServingConfig(tokenizer_type="")


class TestServingConfigChatTemplate:
    def test_chat_template_defaults_to_none(self):
        cfg = ServingConfig()
        assert cfg.chat_message_template is None
        assert cfg.chat_generation_prefix is None

    def test_chat_template_can_override(self):
        cfg = ServingConfig(
            chat_message_template="<|{role}|>{content}",
            chat_generation_prefix="<|assistant|>",
        )
        assert cfg.chat_message_template == "<|{role}|>{content}"
        assert cfg.chat_generation_prefix == "<|assistant|>"


class TestServingConfigPagedAttention:
    def test_paged_attention_default_off(self):
        cfg = ServingConfig()
        assert cfg.use_paged_attention is False
        assert cfg.max_blocks == 256
        assert cfg.block_size == 16


class TestServingConfigNumericBounds:
    """Sweep finding: the concurrency/timeout/block knobs had no bounds.

    ``max_concurrent_requests=0`` builds ``asyncio.Semaphore(0)`` (every
    request blocks and 504s), ``block_size=0``/``max_blocks=0`` feed a
    divide-by-zero into block management, and ``request_timeout<=0`` makes
    every request time out instantly. Each must be rejected at the config
    boundary (fail fast at startup, not on first request).
    """

    def test_max_concurrent_requests_must_be_positive(self):
        with pytest.raises(ValidationError):
            ServingConfig(max_concurrent_requests=0)

    def test_request_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            ServingConfig(request_timeout=0.0)
        with pytest.raises(ValidationError):
            ServingConfig(request_timeout=-1.0)

    def test_block_knobs_must_be_positive(self):
        with pytest.raises(ValidationError):
            ServingConfig(block_size=0)
        with pytest.raises(ValidationError):
            ServingConfig(max_blocks=0)
        with pytest.raises(ValidationError):
            ServingConfig(max_prefixes=0)

    def test_positive_values_still_accepted(self):
        cfg = ServingConfig(max_concurrent_requests=1, request_timeout=0.1, block_size=1, max_blocks=1, max_prefixes=1)
        assert cfg.max_concurrent_requests == 1
        assert cfg.block_size == 1
