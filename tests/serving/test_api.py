import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client(monkeypatch):
    """TestClient backed by a real tiny CPU model (not a mock).

    Used by @pytest.mark.slow tests that assert on actual generation
    output. Avoids CUDA OOM by constructing the model on CPU directly
    instead of going through ServingGenerationService.from_config -> load_model_and_tokenizer.
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
        device=torch.device("cpu"),
    )
    tokenizer = StubTokenizer()

    real_service = ServingGenerationService(
        model=tiny_model,
        tokenizer=tokenizer,
        backend=EagerGenerationBackend(),
        device=torch.device("cpu"),
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
        device="cpu",
        generation_backend="eager",
    )

    with TestClient(app) as c:
        monkeypatch.setattr("llm.serving.routers.generate.generation_service", real_service)
        monkeypatch.setattr("llm.serving.routers.generate.config", cfg)
        monkeypatch.setattr("llm.serving.routers.chat.config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c


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
    assert len(data["generated_text"]) >= len(payload["prompt"])


@pytest.mark.slow
def test_generate_advanced_params(client):
    """测试高级采样参数 (top_p, repetition_penalty)."""
    payload = {"prompt": "hello", "max_new_tokens": 10, "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.2}

    response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert len(data["generated_text"]) >= len(payload["prompt"])


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
