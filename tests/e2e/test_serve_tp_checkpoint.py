"""llm-serve serves a TP-trained checkpoint end-to-end (RIL TASK-208).

TASK-204 leg 3 closes the last gap in the TP story: the loader-level proof
(``test_tp_checkpoint_roundtrip``) showed the on-disk full state dict and the
embedded ``model_config`` round-trip at load time, but nobody had booted the
real ``llm-serve`` FastAPI app against a checkpoint produced by an actual
Tensor-Parallel training run and issued a live request.

This test runs the real training loop on 2 GPUs (pure TP, ``tp_size=2``) —
the engine auto-saves via ``CheckpointManager`` with ``model_config``
embedded in the meta sidecar — then starts ``llm.serving.api.app`` via
``TestClient`` pointed at that on-disk checkpoint and verifies:

* ``/health`` stays ``{"status": "ok"}``;
* ``/generate`` returns real generated text (not the error placeholder);
* ``/v1/chat/completions`` returns a non-empty assistant message;
* the served model was rebuilt from the *embedded* ``model_config`` (same
  ``hidden_size`` as the trained model), not the config-less fallback.

Skipped automatically when fewer than 2 free GPUs are available.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastapi.testclient import TestClient

from llm.data.modules.sft import SFTDataModule
from llm.serving import api as serving_api
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    DistributedConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.sft_task import SFTTask
from tests.support.devices import all_gpu_devices

SERVE_MIN_FREE_BYTES = 1 * 1024**3
SERVE_JOIN_TIMEOUT_S = 360


@pytest.fixture(autouse=True)
def _clear_serving_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe ``LLM_SERVING_*`` env vars so the serve stage starts from a
    known baseline (mirrors ``test_serve_main_path.py``)."""
    for key in list(os.environ):
        if key.startswith("LLM_SERVING_"):
            monkeypatch.delenv(key, raising=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _set_master_env(port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _release_parent_cuda_caches() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        with contextlib.suppress(RuntimeError), torch.cuda.device(index):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _prepare_fixtures(tmp_path: Path, ctx: dict) -> None:
    """Write the shared tokenizer + tiny SFT dataset for the TP training run.

    The tokenizer vocabulary is built so that ``tokenizer.vocab_size == model
    vocab``: ``SimpleCharacterTokenizer`` auto-registers PAD as the last id,
    so a corpus of ``vocab - 1`` unique chars yields exactly ``vocab`` tokens.
    This keeps every sampled generation id decodable at serve time.
    """
    # Cover the FULL Alpaca template (lowercase, digits, punctuation:
    # ``.`` ``,`` ``#`` ``:``  plus newline) so ``encode`` never hits an OOV
    # character. ``string.printable`` includes all of those.
    import string

    corpus = [string.printable]
    tokenizer = SimpleCharacterTokenizer(corpus)
    # Model vocab must be EVEN (TP slices the LM head in half) AND <= the
    # tokenizer's vocab (so every sampled generation id is decodable at serve
    # time). ``tokenizer.vocab_size`` = corpus chars + auto PAD; if that is
    # odd, drop the last (a padding char of printable) to make the model
    # vocab even — the corpus still covers every template character.
    model_vocab = tokenizer.vocab_size
    if model_vocab % 2 != 0:
        model_vocab -= 1
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    data_path = tmp_path / "sft_data.jsonl"
    with data_path.open("w") as f:
        for i in range(12):
            f.write(
                json.dumps(
                    {
                        "instruction": f"INST {i}",
                        "input": "",
                        "output": f"OUT {i}",
                    }
                )
                + "\n"
            )

    ctx["tokenizer_path"] = str(tokenizer_path)
    ctx["data_path"] = str(data_path)
    ctx["vocab_size"] = model_vocab
    ctx["tokenizer_vocab"] = tokenizer.vocab_size


def _build_config(ctx: dict) -> Config:
    """Tiny pure-TP training config; the engine saves ``model_config`` into
    the checkpoint meta sidecar so the serve loader rebuilds this exact
    architecture."""
    return Config(
        model=ModelConfig(
            vocab_size=ctx["vocab_size"],
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_seq_len=64,
        ),
        training=TrainingConfig(batch_size=2, epochs=1, lr=5e-3, warmup_epochs=0, log_every_n_steps=1),
        data=DataConfig(
            dataset_path=ctx["data_path"],
            max_seq_len=64,
            tokenizer_type="simple",
            tokenizer_path=ctx["tokenizer_path"],
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(backend="nccl", parallel_strategy="tp", tp_size=2),
        checkpoint=CheckpointConfig(checkpoint_dir=ctx["ckpt_dir"], save_interval=1, keep_last_n=2, save_best=False),
    )


def _train_tp_worker(rank: int, world_size: int, device_indices: list[int], ctx: dict, results) -> None:
    """Real engine run on 2 GPUs; the engine auto-saves ``latest`` with
    ``model_config`` embedded in the meta sidecar."""
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        torch.manual_seed(42 + rank)

        config = _build_config(ctx)
        data_module = SFTDataModule(config)
        data_module.prepare_data()
        data_module.setup()
        task = SFTTask(config, data_module)
        engine = TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        engine.run()
        if rank == 0:
            assert engine.global_step > 0, "no training ran (vacuous TP serve source)"
        results[rank] = {"success": True, "steps": engine.global_step}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_train(tmp_path: Path) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=SERVE_MIN_FREE_BYTES)
    if len(gpu_devices) < 2:
        pytest.skip("need at least 2 free GPUs (TP-trained checkpoint serve)")
    device_indices = [device.index for device in gpu_devices[:2]]
    _release_parent_cuda_caches()

    ctx: dict = {"ckpt_dir": str(tmp_path / "ckpts")}
    _prepare_fixtures(tmp_path, ctx)
    _set_master_env(_free_port())

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _train_tp_worker,
        args=(2, device_indices, ctx, results),
        nprocs=2,
        join=False,
    )
    end_at = time.monotonic() + SERVE_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP serve-train spawn exceeded {SERVE_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(2):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"
    return ctx


@pytest.mark.need_gpu(2)
@pytest.mark.slow
@pytest.mark.e2e
def test_serve_tp_trained_checkpoint_two_gpu(tmp_path):
    """`llm-serve` boots from and serves a real TP-trained checkpoint end-to-end.

    The engine's on-disk ``latest`` checkpoint (full gathered state dict +
    embedded ``model_config``) must drive the real FastAPI app: health is up,
    /generate and /v1/chat/completions return non-trivial output, and the
    rebuilt model honors the trained architecture (not the fallback).
    """
    ctx = _run_train(tmp_path)
    ckpt_stem = str(Path(ctx["ckpt_dir"]) / "latest")
    assert Path(ckpt_stem + ".safetensors").exists()

    original_config = serving_api.config
    serving_api.config = ServingConfig(
        model_path=ckpt_stem,
        tokenizer_path=ctx["tokenizer_path"],
        tokenizer_type="simple",
        device="cpu",  # keep the busy shared box's GPUs free; tiny model
        generation_backend="eager",
        max_concurrent_requests=2,
        api_key=None,
        host="127.0.0.1",
        request_timeout=120.0,
    )

    try:
        with TestClient(serving_api.app) as client:
            # 1. Health stays up.
            hb = client.get("/health")
            assert hb.status_code == 200
            assert hb.json() == {"status": "ok"}

            # 2. /generate returns real text.
            gen = client.post(
                "/generate",
                json={"prompt": "HELLO", "max_new_tokens": 8, "temperature": 0.7, "top_k": 10},
            )
            assert gen.status_code == 200, gen.text
            data = gen.json()
            assert data["token_count"] >= 1
            assert data["generated_text"] != ""
            assert "[generated]" not in data["generated_text"]

            # 3. /v1/chat/completions returns a non-empty assistant message.
            chat = client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "HELLO"}],
                    "max_tokens": 8,
                    "temperature": 0.7,
                    "top_k": 10,
                },
            )
            assert chat.status_code == 200, chat.text
            choice = chat.json()["choices"][0]
            assert choice["message"]["role"] == "assistant"
            assert choice["message"]["content"].strip() != ""

        # 4. The rebuild honored the EMBEDDED model_config: a fresh load via
        # the real loader must reconstruct a model with the trained
        # hidden_size — which the config-less fallback would NOT reproduce
        # (it would use ServingConfig.hidden_size=64 defaults).
        from llm.serving.loader import load_model_and_tokenizer

        served_cfg = serving_api.config  # re-reading the global we set above
        model, _tok = load_model_and_tokenizer(served_cfg)
        assert model.hidden_size == 32  # == ModelConfig above, not fallback 64
    finally:
        serving_api.config = original_config
