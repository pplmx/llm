import pytest

from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask
from tests.dummies import DummyLMDataModule


@pytest.mark.e2e
def test_resume_training(tmp_path, tiny_config):
    # Setup
    tiny_config.training.batch_size = 2
    tiny_config.training.num_samples = 8
    tiny_config.training.epochs = 2
    tiny_config.checkpoint.save_interval = 1
    tiny_config.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")

    # 1. First Run: Train 1 epoch
    tiny_config.training.epochs = 1

    data_module = DummyLMDataModule(tiny_config)
    task = LanguageModelingTask(tiny_config, data_module)

    engine = TrainingEngine(config=tiny_config, task=task, rank=0, world_size=1, data_module=data_module)

    engine.run()

    assert (tmp_path / "checkpoints" / "epoch_1.pt").exists()

    # 2. Resume Run: Train up to epoch 2
    tiny_config.training.epochs = 2
    tiny_config.checkpoint.resume_from_checkpoint = str(tmp_path / "checkpoints" / "epoch_1.pt")

    # Re-init engine (simulating restart)
    engine_resumed = TrainingEngine(config=tiny_config, task=task, rank=0, world_size=1, data_module=data_module)

    # Verify it sensed start epoch
    assert engine_resumed.start_epoch == 1

    engine_resumed.run()

    # Check that it ran for epoch 2
    # Since epoch 1 is skipped, it runs epoch 1 (0-indexed logic: range(start_epoch, epochs))
    # range(1, 2) -> runs epoch index 1 (which is "Epoch 2").

    assert (tmp_path / "checkpoints" / "epoch_2.pt").exists()
