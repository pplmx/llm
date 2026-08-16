import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client(monkeypatch, device):
    """TestClient backed by a real tiny model (GPU-first, falls back to CPU).

    Used by @pytest.mark.slow tests that assert on actual generation
    output. Constructs the model on the session-scoped ``device`` (GPU
    when available) instead of forcing CPU.
    """
    from unittest.mock import MagicMock

    import torch

    from llm.generation.backends import EagerGenerationBackend
    from llm.models.decoder import DecoderModel
    from llm.serving.auth import api_key_header
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.config import ServingConfig
    from llm.serving.generation_service import ServingGenerationService
    from tests.support.tokenizers import StubTokenizer

    torch.manual_seed(42)
    tiny_model = DecoderModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        device=device,
    )
    tokenizer = StubTokenizer()

    real_service = ServingGenerationService(
        model=tiny_model,
        tokenizer=tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )

    fake_engine = MagicMock()
    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: real_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: fake_engine),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        device=str(device),
        generation_backend="eager",
    )

    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", real_service)
        monkeypatch.setattr("llm.serving.routers.generate.config", cfg)
        monkeypatch.setattr("llm.serving.routers.chat.config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c


@pytest.fixture
def oov_client(monkeypatch, device):
    """TestClient backed by a real tiny model + SimpleCharacterTokenizer.

    The char tokenizer raises ``KeyError`` for characters outside its corpus,
    exercising the router's error mapping for out-of-vocabulary prompts (RIL
    ISS-113) — the default ``StubTokenizer`` never raises.
    """
    from unittest.mock import MagicMock

    import torch

    from llm.generation.backends import EagerGenerationBackend
    from llm.models.decoder import DecoderModel
    from llm.serving.auth import api_key_header
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.config import ServingConfig
    from llm.serving.generation_service import ServingGenerationService
    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    torch.manual_seed(0)
    tiny_model = DecoderModel(
        vocab_size=100,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        device=device,
    )
    tokenizer = SimpleCharacterTokenizer(["abc"])  # only a/b/c (+PAD)

    real_service = ServingGenerationService(
        model=tiny_model,
        tokenizer=tokenizer,
        backend=EagerGenerationBackend(),
        device=device,
    )
    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: real_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: MagicMock()),
    )
    monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        device=str(device),
        generation_backend="eager",
    )
    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", real_service)
        monkeypatch.setattr("llm.serving.routers.generate.config", cfg)
        monkeypatch.setattr("llm.serving.routers.chat.config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c


@pytest.mark.slow
def test_generate_oov_prompt_returns_400_not_500(oov_client):
    """Regression (RIL ISS-113): an out-of-vocabulary character in the prompt
    raises ``KeyError`` from the char tokenizer; the router must surface it as
    a 4xx (client-caused), not a 500 'Internal server error'."""
    payload = {"prompt": "héllo", "max_new_tokens": 2, "temperature": 0.0}
    response = oov_client.post("/generate", json=payload)
    assert response.status_code == 400, response.text
    assert "not found in tokenizer" in response.text


@pytest.mark.slow
def test_batch_generate_oov_prompt_returns_400(oov_client):
    """Same OOV→400 mapping on the batch endpoint."""
    payload = {"prompts": ["hello", "wörld"], "max_new_tokens": 2, "temperature": 0.0}
    response = oov_client.post("/batch_generate", json=payload)
    assert response.status_code == 400, response.text
    assert "not found in tokenizer" in response.text


@pytest.mark.slow
def test_generate_stream_oov_prompt_returns_400(oov_client):
    """A streaming OOV prompt must also fail as a client 400 (validated before
    the SSE starts), not leak 'Error: KeyError' inside a 200 stream."""
    payload = {"prompt": "héllo", "max_new_tokens": 2, "temperature": 0.0, "stream": True}
    response = oov_client.post("/generate", json=payload)
    assert response.status_code == 400, response.text
    assert "not found in tokenizer" in response.text


@pytest.mark.slow
def test_chat_completions_oov_prompt_returns_400_not_500(oov_client):
    """Regression (RIL ISS-147): an out-of-vocabulary character in the chat
    message content raises ``KeyError`` from the char tokenizer; the chat
    router must map it to a client 400 (like the ``/generate`` sibling), not
    fall through to the generic handler and surface a 500 'Internal server
    error'."""
    payload = {
        "model": "tiny",
        "messages": [{"role": "user", "content": "héllo"}],
        "max_tokens": 2,
        "temperature": 0.0,
    }
    response = oov_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    assert "not found in tokenizer" in response.text


@pytest.mark.slow
def test_health_check(client):
    """测试健康检查端点."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.slow
def test_generate_text(client):
    """测试普通文本生成 (非流式)."""
    payload = {"prompt": "hello", "max_new_tokens": 10, "temperature": 0.5, "top_k": 5}

    response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "token_count" in data
    # Regression (RIL ISS-148): the eager backend returns prompt+completion;
    # the route must strip the prompt so generated_text is the completion only
    # and token_count counts generated tokens, not prompt chars.
    assert not data["generated_text"].startswith(payload["prompt"])
    assert data["token_count"] == len(data["generated_text"])


@pytest.mark.slow
def test_generate_advanced_params(client):
    """测试高级采样参数 (top_p, repetition_penalty)."""
    payload = {"prompt": "hello", "max_new_tokens": 10, "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.2}

    response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    # Non-streaming /generate must not echo the prompt (RIL ISS-148).
    assert not data["generated_text"].startswith(payload["prompt"])
    assert data["token_count"] == len(data["generated_text"])


@pytest.mark.slow
def test_generate_stream(client):
    """测试流式文本生成."""
    payload = {"prompt": "hello", "max_new_tokens": 10, "stream": True}

    with client.stream("POST", "/generate", json=payload) as response:
        assert response.status_code == 200
        # 检查是否为流式响应
        assert "text/event-stream" in response.headers["content-type"]

        chunks = [line for line in response.iter_lines() if line]

        # 验证确实收到了数据 chunk
        assert len(chunks) > 0
        # 拼接后的文本应该包含 prompt (SimpleTokenizer 特性)
        full_text = "".join(chunks)
        assert len(full_text) >= len(payload["prompt"])


@pytest.mark.slow
def test_generate_invalid_params(client):
    """测试无效参数."""
    payload = {
        "prompt": "hello",
        "max_new_tokens": 0,  # 无效: < 1
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


@pytest.mark.slow
def test_metrics_endpoint(client):
    """测试 Prometheus 指标端点."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


@pytest.mark.slow
def test_batch_generate_basic(client):
    """测试批处理生成 - 多个 prompt."""
    payload = {
        "prompts": ["hello", "world", "test"],
        "max_new_tokens": 5,
        "temperature": 0.5,
    }
    response = client.post("/batch_generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "generated_text" in result
        assert "token_count" in result


@pytest.mark.slow
def test_batch_generate_single(client):
    """测试批处理生成 - 单个 prompt (退化情况)."""
    payload = {"prompts": ["hello"], "max_new_tokens": 5}
    response = client.post("/batch_generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1


@pytest.mark.slow
def test_batch_generate_empty(client):
    """测试批处理生成 - 空列表边界."""
    payload = {"prompts": [], "max_new_tokens": 5}
    response = client.post("/batch_generate", json=payload)
    # Pydantic 验证应拒绝空列表
    assert response.status_code == 422


@pytest.mark.slow
def test_generate_stream_error_yields_error_chunk(monkeypatch):
    """Streaming /generate: if generation raises, the stream yields an error
    chunk rather than crashing the response."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.stream.side_effect = RuntimeError("stream exploded")
        mock.generate.return_value = "ok"
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompt": "hello", "max_new_tokens": 5, "stream": True}
            with c.stream("POST", "/generate", json=payload) as response:
                assert response.status_code == 200
                chunks = [line for line in response.iter_lines() if line]
                assert any("Error:" in chunk for chunk in chunks)
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_generate_stream_acquires_inference_semaphore(monkeypatch):
    """Regression (RIL ISS-042): ``/generate?stream=true`` must acquire the
    inference semaphore for the *lifetime of the stream*, exactly like the
    chat streaming route. The original concurrency-control fix scoped the
    semaphore to the non-streaming handlers and chat streaming but missed
    this route, leaving /generate streaming unbounded past
    ``max_concurrent_requests``.
    """
    import asyncio
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    class _RecordingSemaphore:
        """Wraps a real ``asyncio.Semaphore(1)`` and counts enter/exit."""

        def __init__(self):
            self.sem = asyncio.Semaphore(1)
            self.entered = 0
            self.exited = 0

        async def __aenter__(self):
            await self.sem.acquire()
            self.entered += 1

        async def __aexit__(self, *exc):
            self.sem.release()
            self.exited += 1

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.stream.return_value = iter(["tok1", "tok2"])
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            recording = _RecordingSemaphore()
            monkeypatch.setattr(generate_module, "inference_semaphore", recording)
            payload = {"prompt": "hello", "max_new_tokens": 5, "stream": True}
            with c.stream("POST", "/generate", json=payload) as response:
                assert response.status_code == 200
                chunks = [line for line in response.iter_lines() if line]
                # No stop/newlines in the stream, so the two mock tokens
                # arrive concatenated as one SSE body line.
                assert "".join(chunks) == "tok1tok2"

        # The semaphore was acquired exactly once and released when the
        # stream ended.
        assert recording.entered == 1, f"stream should acquire the semaphore once, got {recording.entered}"
        assert recording.exited == 1, f"stream should release the semaphore, got {recording.exited}"
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_generate_timeout_error_returns_504(monkeypatch):
    """Non-streaming /generate: TimeoutError maps to HTTP 504."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.generate.side_effect = TimeoutError("request took too long")
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompt": "hello", "max_new_tokens": 5}
            response = c.post("/generate", json=payload)
            assert response.status_code == 504
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_generate_runtime_error_returns_503(monkeypatch):
    """Non-streaming /generate: RuntimeError maps to HTTP 503 (model unavailable)."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.generate.side_effect = RuntimeError("model crashed")
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompt": "hello", "max_new_tokens": 5}
            response = c.post("/generate", json=payload)
            assert response.status_code == 503
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_generate_value_error_returns_400(monkeypatch):
    """Non-streaming /generate: ValueError maps to HTTP 400 (invalid request)."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.generate.side_effect = ValueError("bad input")
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompt": "hello", "max_new_tokens": 5}
            response = c.post("/generate", json=payload)
            assert response.status_code == 400
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_generate_unexpected_error_returns_500(monkeypatch):
    """Non-streaming /generate: unexpected Exception maps to HTTP 500."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.generate.side_effect = Exception("something broke")
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompt": "hello", "max_new_tokens": 5}
            response = c.post("/generate", json=payload)
            assert response.status_code == 500
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_batch_generate_runtime_error_returns_503(monkeypatch):
    """Non-streaming /batch_generate: RuntimeError maps to HTTP 503."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    original_key = config.api_key
    config.api_key = "test-key"

    try:
        mock = MagicMock()
        mock.batch_generate.side_effect = RuntimeError("model crashed")
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine,
            "from_serving_config",
            classmethod(lambda cls, config, **kw: MagicMock()),
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            c.headers["X-API-Key"] = "test-key"
            monkeypatch.setattr(generate_module, "generation_service", mock)
            payload = {"prompts": ["hello"], "max_new_tokens": 5}
            response = c.post("/batch_generate", json=payload)
            assert response.status_code == 503
    finally:
        config.api_key = original_key


@pytest.mark.slow
def test_auth_enforcement(monkeypatch):
    """测试 API Key 验证."""
    from unittest.mock import MagicMock

    import llm.serving.routers.generate as generate_module
    from llm.serving.api import app, config
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    # 模拟设置 API Key
    original_key = config.api_key
    config.api_key = "secret-key"

    try:
        mock = MagicMock()
        mock.generate.return_value = "ok"
        monkeypatch.setattr(ServingGenerationService, "from_config", classmethod(lambda cls, config, **kw: MagicMock()))
        monkeypatch.setattr(
            ContinuousBatchingEngine, "from_serving_config", classmethod(lambda cls, config, **kw: MagicMock())
        )
        monkeypatch.setattr("llm.serving.api._log_server_config", lambda *a, **kw: None)

        with TestClient(app) as c:
            monkeypatch.setattr(generate_module, "generation_service", mock)
            # 1. 无 Key 请求 -> 403
            response = c.post("/generate", json={"prompt": "hi"})
            assert response.status_code == 403

            # 2. 错误 Key -> 403
            response = c.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 403

            # 3. 正确 Key -> 200
            response = c.post("/generate", json={"prompt": "hi"}, headers={"X-API-Key": "secret-key"})
            assert response.status_code == 200

    finally:
        # 恢复配置
        config.api_key = original_key
