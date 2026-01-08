import json
from string import printable

import pytest
import torch

from llm.data.sft_data_module import SFTDataModule
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
from llm.training.tasks.sft_task import SFTTask


@pytest.mark.e2e
def test_sft_e2e_flow(tmp_path):
    # 0. Setup Tokenizer
    tokenizer = SimpleCharacterTokenizer([printable])
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    # 1. Create Dummy Data
    data = [
        {"instruction": "Inst 1", "input": "", "output": "Out 1"},
        {"instruction": "Inst 2", "input": "In 2", "output": "Out 2"},
    ] * 5  # Enough for a batch

    data_path = tmp_path / "sft_data.jsonl"
    with data_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # 2. Setup Config
    config = Config(
        model=ModelConfig(
            hidden_size=32,
            num_layers=2,
            num_heads=4,  # 32/4 = 8 dim per head
            vocab_size=tokenizer.vocab_size + 10,
            max_seq_len=64,
        ),
        training=TrainingConfig(batch_size=2, epochs=1, lr=1e-3, warmup_epochs=0, log_every_n_steps=1),
        data=DataConfig(
            dataset_path=str(data_path), max_seq_len=64, tokenizer_type="simple", tokenizer_path=str(tokenizer_path)
        ),
        optimization=OptimizationConfig(
            use_compile=False,  # Faster for tiny test
            use_amp=False,
            num_workers=0,
        ),
        distributed=DistributedConfig(backend="gloo"),
    )

    # 3. Setup Components
    # We run as rank 0
    data_module = SFTDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = SFTTask(config, data_module)

    # 4. Run Engine
    # We mock world_size=1
    engine = TrainingEngine(
        config=config, task=task, rank=0, world_size=1, data_module=data_module, callbacks=[MetricsLogger()]
    )

    # Run
    engine.run()

    # Check if model updated?
    # Or just that it ran without error.
    # We can check if loss log exists or just pass.
    assert True
