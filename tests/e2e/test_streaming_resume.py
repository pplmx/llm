"""E2E test for streaming training checkpoint resume."""

import pytest
import torch

from llm.data.modules.streaming import StreamingTextDataModule
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask


@pytest.mark.e2e
def test_streaming_training_resume_preserves_data_cursor(tmp_path, tiny_config, monkeypatch, line_tokenizer):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"line-{idx:03d}\n" for idx in range(200)), encoding="utf-8")

    tiny_config.data.dataset_path = str(corpus)
    tiny_config.data.max_seq_len = 8
    tiny_config.data.steps_per_epoch = 4
    tiny_config.training.batch_size = 1
    tiny_config.training.epochs = 1
    tiny_config.optimization.num_workers = 0
    tiny_config.optimization.use_compile = False
    tiny_config.checkpoint.save_interval = 1
    tiny_config.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")

    data_module = StreamingTextDataModule(tiny_config)
    monkeypatch.setattr(data_module, "_load_tokenizer", lambda: line_tokenizer)
    data_module.prepare_data()
    data_module.setup()

    task = LanguageModelingTask(tiny_config, data_module)
    engine = TrainingEngine(
        config=tiny_config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
    )
    engine.run()

    checkpoint_path = tmp_path / "checkpoints" / "epoch_1.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "extra_state" in checkpoint
    assert checkpoint["extra_state"]["stream_data"]["0"]["line_index"] > 0

    resumed_module = StreamingTextDataModule(tiny_config)
    monkeypatch.setattr(resumed_module, "_load_tokenizer", lambda: line_tokenizer)
    resumed_module.prepare_data()
    resumed_module.setup()

    tiny_config.checkpoint.resume_from_checkpoint = str(checkpoint_path)
    resumed_task = LanguageModelingTask(tiny_config, resumed_module)
    resumed_engine = TrainingEngine(
        config=tiny_config,
        task=resumed_task,
        rank=0,
        world_size=1,
        data_module=resumed_module,
    )

    assert resumed_engine.start_epoch == 1
    assert resumed_module.stream_data_state.shards["0"].line_index > 0
