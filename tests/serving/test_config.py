"""Tests for :class:`llm.serving.config.ServingConfig`.

Extends the existing smoke tests with coverage for PEFT validators,
YAML loading, env var prefix, security defaults, and merge guard.
"""

from __future__ import annotations

import logging
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
        assert cfg.port == 8000
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

    def test_loads_custom_port(self, tmp_path: Path):
        yml = tmp_path / "serve.yaml"
        yml.write_text(yaml.dump({"host": "127.0.0.1", "port": 9000}))
        cfg = ServingConfig.from_yaml(yml)
        assert cfg.port == 9000

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


class TestServingConfigGenerationBackendValidation:
    """A typo'd backend name must fail at config load, not at first request.

    ``get_generation_backend`` already raises with the registry's name list,
    but only once a request arrives — the server would boot, load the model,
    and answer ``/health`` before 500-ing every generation call.
    """

    @pytest.mark.parametrize("name", ["eager", "batched", "speculative"])
    def test_registered_backends_accepted(self, name):
        assert ServingConfig(generation_backend=name).generation_backend == name

    def test_unknown_backend_rejected_at_load(self):
        with pytest.raises(ValueError, match="Unknown generation_backend"):
            ServingConfig(generation_backend="batchedd")  # plausible typo

    def test_error_names_the_available_backends(self):
        with pytest.raises(ValueError, match="Registered backends"):
            ServingConfig(generation_backend="nope")


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

    def test_port_must_be_in_valid_range(self):
        with pytest.raises(ValidationError):
            ServingConfig(port=0)
        with pytest.raises(ValidationError):
            ServingConfig(port=65536)


class TestServingCliPort:
    def test_cli_main_forwards_config_port(self, monkeypatch):
        """``llm serve`` must bind the operator's configured
        port, not a hardcoded 8000 (ServingConfig.port -> uvicorn.run)."""
        from llm.serving import cli

        captured: dict = {}

        def fake_run(*_args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("uvicorn.run", fake_run, raising=False)

        cli.main(ServingConfig(host="127.0.0.1", port=9000))

        assert captured.get("port") == 9000, "uvicorn.run must receive ServingConfig.port"


class TestServingCliHelp:
    def test_serve_help_prints_usage_without_launching_server(self, monkeypatch):
        """``llm serve --help`` must print usage, not silently boot uvicorn.

        The script used to point at the bare ``api.main`` (no argv parsing),
        so ``llm serve --help`` launched a real uvicorn server holding a CUDA
        device. It now points at a Typer app (``llm.serving.cli:app``); help
        must print the options and exit 0 with uvicorn never invoked.
        """
        from typer.testing import CliRunner

        from llm.serving import cli

        launched = {"n": 0}

        def fake_run(*_args, **_kwargs):
            launched["n"] += 1

        monkeypatch.setattr("uvicorn.run", fake_run, raising=False)

        from tests.support.ansi import help_render_diag, strip_ansi

        result = CliRunner().invoke(cli.app, ["--help"])
        clean = strip_ansi(result.output)

        assert result.exit_code == 0
        assert "--host" in clean, help_render_diag(result.output)
        assert "--port" in clean, help_render_diag(result.output)
        assert launched["n"] == 0, "llm serve --help must not launch uvicorn"

    def test_serve_invokes_main_without_help(self, monkeypatch):
        """Without ``--help`` the command still runs the real server path
        (ServingConfig -> assert_safe_bind -> uvicorn.run)."""
        from typer.testing import CliRunner

        from llm.serving import cli

        launched = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: launched.update(k), raising=False)
        # A loopback host + no api_key is safe.
        monkeypatch.setenv("LLM_SERVING_HOST", "127.0.0.1")
        result = CliRunner().invoke(cli.app, ["--host", "127.0.0.1", "--port", "8123"])
        assert result.exit_code == 0
        assert launched.get("port") == 8123

    def test_module_entry_help_prints_usage_without_launching_server(self):
        """``python -m llm.serving.cli --help`` must route through the Typer
        app, not the bare :func:`cli.main`, so it prints usage and exits 0
        instead of silently booting uvicorn on a CUDA device.

        The ``llm serve`` script entry point was fixed (Typer app), but the
        ``__main__`` guard previously still called the argv-ignoring
        ``main()`` — the same bug surfaced again via ``python -m``.
        """
        import subprocess
        import sys

        from tests.support.ansi import help_render_diag, strip_ansi

        proc = subprocess.run(
            [sys.executable, "-m", "llm.serving.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = proc.stdout + proc.stderr
        clean = strip_ansi(out)  # see ansi.py: ANSI styling splits option tokens on a tty
        assert proc.returncode == 0, out
        assert "--host" in clean, help_render_diag(out)
        assert "--port" in clean, help_render_diag(out)
        # The module must not fall through to the server boot path.
        assert "uvicorn" not in clean
        assert "Running on" not in clean

    def test_api_module_entry_help_prints_usage_without_launching_server(self):
        """``python -m llm.serving.api --help`` must route through the Typer
        app, not the argv-ignoring :func:`cli.main`, so it prints usage and
        exits 0 instead of booting uvicorn on a CUDA device.

        ``api.py``'s ``__main__`` guard previously called the alias to
        :func:`cli.main` — the third instance of the bug shape.
        A bare ``python -m llm.serving.api`` (no ``--help``) must still boot.
        """
        import subprocess
        import sys

        from tests.support.ansi import help_render_diag, strip_ansi

        proc = subprocess.run(
            [sys.executable, "-m", "llm.serving.api", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = proc.stdout + proc.stderr
        clean = strip_ansi(out)  # see ansi.py: ANSI styling splits option tokens on a tty
        assert proc.returncode == 0, out
        assert "--host" in clean, help_render_diag(out)
        assert "--port" in clean, help_render_diag(out)
        assert "uvicorn" not in clean
        assert "Starting up" not in clean


class TestConfigPrecedence:
    """YAML explicit values > environment variables > field defaults.

    ``ServingConfig`` is a ``BaseSettings``; ``from_yaml`` builds it with
    ``model_validate``, which merges the YAML over the env-derived defaults.
    The consequence is easy to get wrong when reading the docs: an env var only
    fills a field the YAML did NOT set. ``configs/serve_pretrained.yaml`` pins
    ``host`` and ``api_key`` explicitly, so ``LLM_SERVING_HOST`` /
    ``LLM_SERVING_API_KEY`` do nothing for that file — which is exactly the pair
    the tutorial used to name as env-overridable.
    """

    def test_yaml_explicit_value_beats_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_SERVING_PORT", "7777")
        path = tmp_path / "serve.yaml"
        path.write_text("port: 8123\n", encoding="utf-8")
        assert ServingConfig.from_yaml(path).port == 8123

    def test_environment_fills_a_field_the_yaml_omits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_SERVING_PORT", "7777")
        path = tmp_path / "serve.yaml"
        path.write_text("host: 127.0.0.1\n", encoding="utf-8")
        assert ServingConfig.from_yaml(path).port == 7777

    def test_shipped_pretrained_config_pins_host_and_api_key(self):
        """Pin the fact the tutorial now warns about.

        If someone later removes those two lines from the YAML, the tutorial's
        caveat becomes stale — this test makes that regression visible.
        """
        import pathlib

        yaml_path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "serve_pretrained.yaml"
        raw = yaml_path.read_text(encoding="utf-8")
        for field in ("host:", "api_key:"):
            assert field in raw, (
                f"serve_pretrained.yaml no longer pins {field} — update the precedence note in "
                "docs/tutorials/03-inference.md §2.3"
            )


# --------------------------------------------------------------------------- #
# A YAML null must not silently clear an env var the operator set
# --------------------------------------------------------------------------- #


class TestYamlNullShadowsEnvVar:
    """``api_key: null`` in a YAML beats ``LLM_SERVING_API_KEY`` — warn about it.

    Precedence is counter-intuitive: an explicit YAML key is a *supplied* value,
    so pydantic-settings lets it win over the environment. A ``null`` is the
    worst case, because it reads as "not configured here" while actually
    CLEARING what the operator exported.

    This was a live safety bug, not a theoretical one: with
    ``LLM_SERVING_API_KEY`` exported, ``configs/serve_local_demo.yaml`` pinned
    ``api_key: null`` and the server served an unauthenticated request with
    **200**. The shipped configs no longer carry that line, and this warning
    makes the next occurrence visible instead of silent.
    """

    def test_warns_when_null_shadows_a_set_env_var(self, tmp_path: Path, monkeypatch, caplog):
        from llm.serving.config import ServingConfig

        monkeypatch.setenv("LLM_SERVING_API_KEY", "secret-from-env")
        yml = tmp_path / "serve.yaml"
        yml.write_text("host: 127.0.0.1\napi_key: null\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="llm.serving.config"):
            cfg = ServingConfig.from_yaml(yml)

        assert cfg.api_key is None, "an explicit YAML null still wins — that is the documented precedence"
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "api_key" in messages
        assert "LLM_SERVING_API_KEY" in messages
        assert "OVERRIDES" in messages

    def test_silent_when_no_env_var_is_set(self, tmp_path: Path, monkeypatch, caplog):
        """No env var, nothing shadowed — a null field is legitimate then."""
        from llm.serving.config import ServingConfig

        monkeypatch.delenv("LLM_SERVING_API_KEY", raising=False)
        yml = tmp_path / "serve.yaml"
        yml.write_text("host: 127.0.0.1\napi_key: null\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="llm.serving.config"):
            ServingConfig.from_yaml(yml)

        assert not [r for r in caplog.records if "OVERRIDES" in r.getMessage()]

    def test_silent_when_yaml_value_is_not_null(self, tmp_path: Path, monkeypatch, caplog):
        """A real YAML value overriding the env var is normal, not a footgun."""
        from llm.serving.config import ServingConfig

        monkeypatch.setenv("LLM_SERVING_API_KEY", "secret-from-env")
        yml = tmp_path / "serve.yaml"
        yml.write_text("host: 127.0.0.1\napi_key: explicit-from-yaml\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="llm.serving.config"):
            ServingConfig.from_yaml(yml)

        assert not [r for r in caplog.records if "OVERRIDES" in r.getMessage()]

    def test_shipped_serve_configs_do_not_pin_api_key(self):
        """No shipped serving config may pin ``api_key`` — not even to null.

        An explicit value shadows the env var; ``null`` shadows it to "no auth".
        The only safe shape is to omit the key entirely and let the environment
        (or an explicit code-level override) supply it.
        """
        import yaml

        configs_dir = Path(__file__).resolve().parents[2] / "configs"
        offenders = []
        for path in sorted(configs_dir.glob("serve*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "api_key" in raw:
                offenders.append(f"{path.name}: api_key={raw['api_key']!r}")
        assert not offenders, (
            "shipped serving configs must not set api_key at all (an explicit YAML "
            "value shadows LLM_SERVING_API_KEY):\n  " + "\n  ".join(offenders)
        )

    def test_env_var_reaches_the_demo_config(self, monkeypatch):
        """The documented env-var workflow must actually work for the demo."""
        from llm.serving.config import ServingConfig

        monkeypatch.setenv("LLM_SERVING_API_KEY", "b" * 64)
        cfg = ServingConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs" / "serve_local_demo.yaml")
        assert cfg.api_key == "b" * 64
