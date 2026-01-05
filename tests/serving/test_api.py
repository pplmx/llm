import pytest
from fastapi.testclient import TestClient

from llm.serving.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
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

        chunks = []
        for line in response.iter_lines():
            if line:
                chunks.append(line)

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
def test_auth_enforcement():
    """测试 API Key 验证."""
    from fastapi.testclient import TestClient

    from llm.serving.api import app, config

    # 模拟设置 API Key
    original_key = config.api_key
    config.api_key = "secret-key"

    try:
        with TestClient(app) as c:
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
