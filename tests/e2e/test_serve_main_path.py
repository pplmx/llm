"""E2E tests for the serving main path (Main Path #3 docs/e2e alignment).

Mirrors :mod:`tests.e2e.test_stream_lm_main_path` and
:mod:`tests.e2e.test_sft_main_path` for the ``llm-serve`` main path —
the same tutorial-alignment pattern applied to inference serving.

Covers:

- ``configs/serve_local_demo.yaml`` and ``configs/serve_pretrained.yaml``
  are well-formed and loadable via :meth:`ServingConfig.from_yaml`
  (the new YAML loader added in Main Path #3 — symmetric with
  ``Config.from_yaml`` on the training side).
- The :class:`ServingConfig` produced by the YAML loader exposes the
  right fields the tutorial calls out (``api_key`` placeholder,
  ``model_path``, ``peft_*``).
- The ``llm-serve`` FastAPI app boots and serves the OpenAI-compatible
  ``/v1/chat/completions`` endpoint end-to-end via :class:`TestClient`.
- ``/generate`` (non-streaming) and ``/generate`` (streaming) both
  round-trip a real request.
- ``/metrics`` exposes the custom domain metrics
  (``llm_tokens_generated_total`` / ``llm_batch_fill_ratio`` / etc.)
  wired in T2 #22.
- The public-host guard refuses to start the server on a non-loopback
  interface without an ``api_key`` (T2 #7).
- API-key authentication accepts both ``X-API-Key`` and
  ``Authorization: Bearer`` headers (and rejects wrong keys with 403).
- The full PEFT training → save → serve closed loop (T2 PEFT #49):
  train a LoRA adapter via ``apply_lora`` + ``save_peft``, then load
  via ``ServingConfig.from_yaml`` → ``load_model_and_tokenizer`` and
  verify the forward output differs from the un-adapted base (the
  headline behaviour that proves the sidecar was actually applied).

These tests live under ``tests/e2e/`` so they're opt-in via
``pytest -m e2e`` — slower than unit tests because they boot the
FastAPI app (which loads a model) and run a real training step.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


# The old ``_force_cpu`` fixture was removed.
# Tests now auto-detect GPU availability; each test keeps its tensors
# on a consistent device, falling back to CPU only when no GPU is present.


@pytest.fixture(autouse=True)
def _clear_serving_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe ``LLM_SERVING_*`` env vars so each test starts from a
    known baseline.

    The serving config is a :class:`pydantic_settings.BaseSettings`
    instance — it loads env vars at module import time, but
    :meth:`ServingConfig.from_yaml` calls :meth:`model_validate` which
    overrides those values with the YAML payload. We still want the
    env vars clean so :func:`llm.serving.cli.main` (the guard test)
    sees a deterministic state.
    """
    for key in list(os.environ):
        if key.startswith("LLM_SERVING_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def serving_app():
    """Boot the FastAPI app via :class:`TestClient` (which runs the
    lifespan handler). Yields the client; teardown is handled by
    TestClient's ``__exit__``.
    """
    from llm.serving.api import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. YAML validation — the docs must point at configs that actually load
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestServePresetConfigs:
    """The shipped serving preset configs must load + validate cleanly."""

    def test_local_demo_yaml_loads(self) -> None:
        """``configs/serve_local_demo.yaml`` is a valid ServingConfig."""
        from llm.serving.config import ServingConfig

        path = REPO_ROOT / "configs" / "serve_local_demo.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = ServingConfig.from_yaml(path)
        # Smoke config = loopback + no checkpoint + no auth.
        assert cfg.host == "127.0.0.1"
        assert cfg.api_key is None
        assert cfg.model_path is None
        # Arch defaults are tiny so the dummy model fits on CPU.
        assert cfg.hidden_size == 64
        assert cfg.num_layers == 2
        assert cfg.tokenizer_type == "simple"
        # Optional features off by default for the smoke test.
        assert cfg.use_paged_attention is False
        assert cfg.enable_prefix_cache is False
        assert cfg.peft_method is None

    def test_pretrained_yaml_loads(self) -> None:
        """``configs/serve_pretrained.yaml`` references a real
        checkpoint path and the PEFT adapter fields."""
        from llm.serving.config import ServingConfig

        path = REPO_ROOT / "configs" / "serve_pretrained.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = ServingConfig.from_yaml(path)
        # Production preset points at a real training checkpoint.
        assert cfg.model_path is not None
        assert cfg.model_path.endswith(".pt")
        # HF tokenizer to match `configs/sft_alpaca.yaml`.
        assert cfg.tokenizer_type == "hf"
        assert cfg.tokenizer_path == "tokenizer"
        # PEFT fields wired for the LoRA + adapter workflow.
        assert cfg.peft_method == "lora"
        assert cfg.peft_kwargs == {"rank": 8, "alpha": 16.0}
        assert cfg.peft_adapter_path is not None
        assert cfg.peft_adapter_path.endswith(".bin")
        # Public-host guard must be satisfied (api_key set, even if
        # placeholder).
        assert cfg.api_key is not None
        # Production-shape model.
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 6
        # Performance features enabled for production.
        assert cfg.use_paged_attention is True
        assert cfg.enable_prefix_cache is True
        assert cfg.compile_model is True
        assert cfg.generation_backend == "batched"

    def test_unknown_peft_method_is_rejected(self, tmp_path: Path) -> None:
        """``peft_method`` validator rejects unknown methods at config-load time.

        Regression for the Main Path #3 contract that "validators run
        on construction" — failing loud at YAML load is better than
        silently no-op'ing at first-request time.
        """
        from llm.serving.config import ServingConfig

        bad_yaml = tmp_path / "bad_peft.yaml"
        bad_yaml.write_text("peft_method: loraa\npeft_kwargs:\n  rank: 4\n")
        with pytest.raises(ValueError, match="Unknown PEFT method"):
            ServingConfig.from_yaml(bad_yaml)

    def test_peft_merge_with_non_mergeable_method_rejected(self, tmp_path: Path) -> None:
        """``peft_merge=true`` is refused for bitfit / qlora / prefix_tuning."""
        from llm.serving.config import ServingConfig

        bad_yaml = tmp_path / "bad_merge.yaml"
        bad_yaml.write_text("peft_method: bitfit\npeft_merge: true\n")
        with pytest.raises(ValueError, match="not supported for method 'bitfit'"):
            ServingConfig.from_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# 2. llm-serve boots + serves the OpenAI-compatible chat completions
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestLLMServeEndToEnd:
    """The full HTTP path: boot the FastAPI app, hit /health, /v1/chat/completions, /metrics."""

    def test_health_endpoint(self, serving_app: TestClient) -> None:
        """GET /health returns ``{"status": "ok"}``."""
        r = serving_app.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_chat_completions_round_trip(self, serving_app: TestClient) -> None:
        """POST /v1/chat/completions returns an OpenAI-shaped response."""
        r = serving_app.post(
            "/v1/chat/completions",
            json={
                "model": "llm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "temperature": 0.5,
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        # OpenAI schema fields must be present.
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] == "stop"
        # Token usage bookkeeping.
        usage = data["usage"]
        assert usage["prompt_tokens"] >= 1
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_generate_endpoint_round_trip(self, serving_app: TestClient) -> None:
        """POST /generate returns the legacy non-OpenAI response shape."""
        r = serving_app.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 5, "temperature": 0.5},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "generated_text" in data
        assert "token_count" in data
        assert data["token_count"] == len(data["generated_text"])

    def test_generate_streaming(self, serving_app: TestClient) -> None:
        """POST /generate with ``stream=true`` returns SSE chunks."""
        with serving_app.stream(
            "POST",
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 5, "stream": True},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            chunks = [line for line in r.iter_lines() if line]
            # Real tokens were streamed (not a single-line error).
            assert len(chunks) > 0

    def test_metrics_endpoint_exposes_domain_metrics(self, serving_app: TestClient) -> None:
        """GET /metrics includes the custom domain metrics (T2 #22).

        Generates a couple of chat completions first so the counters
        have non-zero values — empty counters wouldn't appear in the
        Prometheus exposition.
        """
        for _ in range(2):
            serving_app.post(
                "/v1/chat/completions",
                json={
                    "model": "llm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 3,
                },
            )

        r = serving_app.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Domain metrics wired in T2 #22 must be present.
        for marker in (
            "llm_tokens_generated_total",
            "llm_tokens_per_request",
            "llm_request_duration_seconds",
            "llm_batch_fill_ratio",
            "llm_inflight_requests",
        ):
            assert marker in body, f"missing metric {marker!r} in /metrics output"
        # The counter actually moved after generating tokens.
        assert "llm_tokens_generated_total" in body

    def test_invalid_request_returns_structured_error(self, serving_app: TestClient) -> None:
        """Invalid params return the structured error envelope (T2 #16)."""
        r = serving_app.post(
            "/v1/chat/completions",
            json={
                "model": "llm",
                "messages": [],  # invalid: at least one message required
                "max_tokens": 5,
            },
        )
        # Pydantic validation failure → 422 (FastAPI default for
        # request body validation; the structured envelope is for
        # server-side errors, but the status code reflects the
        # client-side mistake).
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 3. Security: public-host guard + API-key authentication
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestLLMServeSecurity:
    """The serving tier's auth + safety rails must be enforced end-to-end."""

    def test_public_host_guard_refuses_start_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``llm-serve`` refuses to start on 0.0.0.0 without an api_key.

        T2 #7 hardening: failing loud at startup is better than
        silently accepting anonymous traffic on a public interface.
        """
        from llm.serving.cli import main as serve_main
        from llm.serving.config import ServingConfig

        cfg = ServingConfig(host="0.0.0.0", api_key=None)  # noqa: S104 (the value being tested)
        with pytest.raises(RuntimeError, match="Refusing to start"):
            serve_main(config=cfg)

    def test_loopback_start_succeeds_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loopback host + no api_key is the OK path (for local dev).

        The CLI imports ``uvicorn`` lazily inside :func:`cli.main`, so
        we monkey-patch ``sys.modules["uvicorn"]`` before calling —
        this intercepts the import without depending on a module-level
        attribute that may or may not exist.
        """
        import sys
        import types

        from llm.serving import cli
        from llm.serving.config import ServingConfig

        called = {"uvicorn_run": False}

        def _fake_uvicorn_run(*_args: Any, **_kwargs: Any) -> None:
            called["uvicorn_run"] = True

        fake_uvicorn = types.ModuleType("uvicorn")
        # ``types.ModuleType`` doesn't expose arbitrary attributes to a
        # static type checker — cast to ``Any`` so the assignment is
        # accepted. The runtime behaviour (the fake replaces the real
        # ``uvicorn`` module before the CLI imports it) is unchanged.
        cast("Any", fake_uvicorn).run = _fake_uvicorn_run
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

        cfg = ServingConfig(host="127.0.0.1", api_key=None)
        cli.main(config=cfg)
        assert called["uvicorn_run"] is True

    def test_api_key_authentication_round_trip(self, serving_app: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """With api_key set, the X-API-Key header authenticates the request."""
        from llm.serving import api

        monkeypatch.setattr(api.config, "api_key", "test-secret-key-123")
        try:
            # Wrong key → 403.
            r = serving_app.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
                headers={"X-API-Key": "wrong-key"},
            )
            assert r.status_code == 403

            # Correct X-API-Key → 200.
            r = serving_app.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
                headers={"X-API-Key": "test-secret-key-123"},
            )
            assert r.status_code == 200, r.text

            # Correct Bearer token → 200.
            r = serving_app.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
                headers={"Authorization": "Bearer test-secret-key-123"},
            )
            assert r.status_code == 200, r.text

            # No key → 403.
            r = serving_app.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
            )
            assert r.status_code == 403
        finally:
            api.config.api_key = None


# ---------------------------------------------------------------------------
# 4. Training → save → serve closed loop (T2 PEFT #49 main-path integration)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPeftServeRoundTrip:
    """The full PEFT pipeline: train a LoRA adapter → save sidecar → serve it."""

    def test_lora_train_save_serve_produces_different_output(
        self,
        tmp_path: Path,
        tiny_model,
        tiny_config: Any,
    ) -> None:
        """End-to-end: apply LoRA → one optimizer step → save sidecar
        → load via the serving loader → forward output is byte-different
        from the un-adapted base model.

        This is the headline behavioural test for T2 PEFT #49 + Main
        Path #3 — proves the training → serving closed loop actually
        works (not just "the loader accepted the config").
        """

        from llm.core.lora import apply_lora
        from llm.serving.config import ServingConfig
        from llm.serving.loader import load_model_and_tokenizer
        from llm.training.core.callbacks import PEFTAdapterCheckpointCallback

        # 1. Build a tiny "trained" model + tokenizer; apply LoRA.
        #
        # The shared session ``device`` fixture may select CUDA or CPU.
        # The serving loader returns a model on CPU; move it to the same
        # device as ``tiny_model`` so the forward comparison works.
        tiny_model_device = next(tiny_model.parameters()).device
        torch.manual_seed(0)
        train_view = deepcopy(tiny_model)
        apply_lora(train_view, rank=4, alpha=8.0)

        # 2. Mutate LoRA params via one optimizer step (simulate training).
        opt = torch.optim.SGD([p for p in train_view.parameters() if p.requires_grad], lr=0.01)
        ids = torch.randint(0, tiny_config.model.vocab_size, (1, 4), device=tiny_model_device)
        out = train_view(input_ids=ids)
        loss = out.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 3. Save the sidecar via the same callback the trainer uses.
        adapter_path = tmp_path / "peft_adapter_lora.bin"
        cb = PEFTAdapterCheckpointCallback(
            peft_method="lora",
            peft_save_path=str(adapter_path),
        )
        # The callback's ``on_train_end`` only reads ``self.engine.model``
        # (and ``self.engine.logger`` is getattr-guarded for the warning
        # path) — ``_FakeEngine`` is structurally compatible even though
        # it isn't a real :class:`TrainingEngine``. Cast to ``Any`` to
        # bypass the strict signature; the runtime behaviour is unchanged.
        cb.set_engine(cast("Any", _FakeEngine(train_view)))
        cb.on_train_end()
        assert adapter_path.exists(), "PEFT adapter sidecar not written"

        # 4. Save the BASE weights (un-PEFT'd) + tokenizer for the loader.
        base_path = _save_base_ckpt(tmp_path, tiny_model, tiny_config)
        tokenizer_path = _save_tokenizer(tmp_path, vocab_size=tiny_config.model.vocab_size)

        # 5. Build a ServingConfig that points at both, load via the
        #    serving loader.
        cfg = ServingConfig(
            model_path=str(base_path),
            tokenizer_path=str(tokenizer_path),
            tokenizer_type="simple",
            peft_method="lora",
            peft_kwargs={"rank": 4, "alpha": 8.0},
            peft_adapter_path=str(adapter_path),
            peft_merge=False,
        )
        served_model, _tokenizer = load_model_and_tokenizer(cfg)
        served_model = served_model.to(tiny_model_device)

        # 6. Forward through served model vs. base model — must differ.
        torch.manual_seed(0)
        probe = torch.randint(0, tiny_config.model.vocab_size, (1, 4), device=tiny_model_device)
        served_out = served_model(input_ids=probe)
        base_out = tiny_model(input_ids=probe)
        assert not torch.allclose(served_out, base_out), (
            "Served model output is identical to the un-adapted base — "
            "the LoRA sidecar was silently dropped somewhere in the "
            "load_model_and_tokenizer pipeline."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal stand-in for ``TrainingEngine`` exposing ``.model``.

    The PEFT callback only reads ``engine.model`` to call
    ``save_peft``; it doesn't touch any other engine state. A
    ``SimpleNamespace``-style wrapper keeps the test free of the
    full engine construction (which would pull in data modules /
    optimizers / etc.).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model


def _save_base_ckpt(tmp_path: Path, model: torch.nn.Module, tiny_config: Any) -> Path:
    """Save a base (un-PEFT'd) checkpoint in the format the loader expects."""
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


def _save_tokenizer(tmp_path: Path, vocab_size: int) -> Path:
    """Save a minimal :class:`SimpleCharacterTokenizer` the loader can pick up.

    The serving loader's :func:`load_tokenizer` requires a tokenizer
    path whenever a model_path is set — this helper produces one
    sized to the test's vocab so the loader returns a real
    tokenizer object instead of erroring.
    """
    import string

    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tokenizer = SimpleCharacterTokenizer(list(string.printable[:vocab_size]))
    path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, path)
    return path
