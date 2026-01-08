import json
from string import printable

import pytest
import torch

from llm.data.dpo_data_module import DPODataModule
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
from llm.training.tasks.dpo_task import DPOTask


@pytest.mark.e2e
def test_dpo_e2e_flow(tmp_path):
    # 0. Setup Tokenizer
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    # 1. Create Dummy Data
    data = [
        {"prompt": "Q1", "chosen": "Good1", "rejected": "Bad1"},
        {"prompt": "Q2", "chosen": "Good2", "rejected": "Bad2"},
    ] * 5

    data_path = tmp_path / "dpo_data.jsonl"
    with data_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # 2. Setup Config
    config = Config(
        model=ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, vocab_size=tokenizer.vocab_size + 10, max_seq_len=64
        ),
        training=TrainingConfig(batch_size=2, epochs=1, lr=1e-3, warmup_epochs=0, log_every_n_steps=1, dpo_beta=0.1),
        data=DataConfig(
            dataset_path=str(data_path), max_seq_len=64, tokenizer_type="simple", tokenizer_path=str(tokenizer_path)
        ),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(backend="gloo"),
    )

    # 3. Setup Components
    data_module = DPODataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = DPOTask(config, data_module)

    # 4. Run Engine
    engine = TrainingEngine(
        config=config, task=task, rank=0, world_size=1, data_module=data_module, callbacks=[MetricsLogger()]
    )

    # Run
    engine.run()
    assert True
