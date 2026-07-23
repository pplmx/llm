"""End-to-end HTTP tests for serving with training checkpoints."""

from __future__ import annotations

import string
from collections.abc import Iterator

import pytest
import torch
from fastapi.testclient import TestClient

from llm.serving import api
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.distributed import model_state_dict


def _write_serving_checkpoint(
    tmp_path,
    tiny_model,
    tiny_config,
) -> tuple[str, str]:
    """Persist a training checkpoint and matching tokenizer for serving tests."""
    # Leave room for auto-added PAD so tokenizer vocab matches model vocab_size.
    corpus_size = tiny_config.model.vocab_size - 1
    tokenizer = SimpleCharacterTokenizer(list(string.printable[:corpus_size]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 0,
            "loss": 0.5,
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )
    return str(ckpt_path), str(tokenizer_path)




@pytest.fixture
def device():
    """Force CPU for these tests — the session-scoped device fixture from
    conftest.py creates models on CUDA, which OOMs on constrained boxes."""
    return torch.device("cpu")

@pytest.fixture
def checkpoint_client(tmp_path, tiny_model, tiny_config) -> Iterator[TestClient]:
    """Start the FastAPI app with a real training checkpoint loaded."""
    ckpt_path, tokenizer_path = _write_serving_checkpoint(tmp_path, tiny_model, tiny_config)
    original_config = api.config

    api.config = ServingConfig(
        model_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        tokenizer_type="simple",
        device="cpu",
        generation_backend="eager",
        max_concurrent_requests=2,
    )

    with TestClient(api.app) as client:
        yield client

    api.config = original_config


@pytest.mark.quick
def test_generate_from_training_checkpoint(checkpoint_client):
    """POST /generate should run inference through a loaded training checkpoint."""
    payload = {
        "prompt": "hello",
        "max_new_tokens": 5,
        "temperature": 0.5,
        "top_k": 5,
    }

    response = checkpoint_client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] != "[generated]"
    assert data["token_count"] >= 1
    assert len(data["generated_text"]) >= len(payload["prompt"])


@pytest.mark.quick
def test_batch_generate_from_training_checkpoint(checkpoint_client):
    """POST /batch_generate should serve multiple prompts from one checkpoint."""
    payload = {
        "prompts": ["hello", "world"],
        "max_new_tokens": 4,
        "temperature": 0.5,
    }

    response = checkpoint_client.post("/batch_generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert result["generated_text"] != "[generated]"
        assert result["token_count"] >= 1


@pytest.mark.quick
def test_health_with_training_checkpoint(checkpoint_client):
    """Health endpoint should stay available when serving from checkpoint."""
    response = checkpoint_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
