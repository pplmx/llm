"""SimPO engine e2e — ``--task simpo`` runs to completion on preference pairs."""

import json
from string import printable

import pytest
import torch

from llm.data.modules.dpo import DPODataModule
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.callbacks import MetricsLogger
from llm.training.core.config import (
    Config,
    DataConfig,
    DistributedConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.simpo_task import SimPOTask


@pytest.mark.e2e
def test_simpo_e2e_flow(tmp_path, device):
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    data = [
        {"prompt": "Q1", "chosen": "Good1", "rejected": "Bad1"},
        {"prompt": "Q2", "chosen": "Good2", "rejected": "Bad2"},
    ] * 5
    data_path = tmp_path / "simpo_data.jsonl"
    with data_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    backend = "nccl" if device.type == "cuda" else "gloo"
    config = Config(
        model=ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, vocab_size=tokenizer.vocab_size + 10, max_seq_len=64
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            lr=1e-3,
            warmup_epochs=0,
            log_every_n_steps=1,
            simpo_beta=2.0,
            simpo_gamma=0.5,
            simpo_lambda=1.0,
        ),
        data=DataConfig(
            dataset_path=str(data_path), max_seq_len=64, tokenizer_type="simple", tokenizer_path=str(tokenizer_path)
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(backend=backend),
    )

    data_module = DPODataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = SimPOTask(config, data_module)
    engine = TrainingEngine(
        config=config, task=task, rank=0, world_size=1, data_module=data_module, callbacks=[MetricsLogger()]
    )
    engine.run()
    assert engine.global_step > 0, "SimPO engine should train at least one step"
