import torch

from llm.models.decoder import DecoderModel
from llm.training.core.config import Config, ModelConfig
from llm.training.tasks.sft_task import SFTTask


def test_sft_task_train_step():
    # Use real configuration for a tiny model
    config = Config(model=ModelConfig(hidden_size=16, num_layers=1, num_heads=2, vocab_size=100, max_seq_len=16))

    # Use a dummy data module or None?
    # SFTTask might access self.tokenizer.
    # Let's create a minimal DataModule class if needed, or pass None if safe.
    # LanguageModelingTask init: self.tokenizer = data_module.tokenizer if data_module else None
    # SFTTask train_step doesn't utilize tokenizer directly.
    task = SFTTask(config, data_module=None)

    # Use real DecoderModel
    model = DecoderModel(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_seq_len=config.model.max_seq_len,
    )

    criterion = task.build_criterion()

    # Batch: 2 sequences, len 4
    # Seq 1: [P, P, R, R] -> Labels [-100, -100, R, R] (P=Prompt, R=Response)
    # Seq 2: [P, R, R, R]
    batch = {
        "input_ids": torch.randint(0, 100, (2, 4)),
        "labels": torch.tensor([[-100, -100, 10, 11], [-100, 20, 21, 22]]),  # 2 examples
        "attention_mask": torch.ones(2, 4),
    }

    loss, metrics = task.train_step(batch, model, criterion)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss" in metrics
    assert "ppl" in metrics
