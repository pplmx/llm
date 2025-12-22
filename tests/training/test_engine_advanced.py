import pytest
import torch

from llm.data.synthetic_data_module import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask


@pytest.fixture
def mock_config():
    return Config(
        model=ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=64),
        training=TrainingConfig(batch_size=2, num_samples=10),
        optimization=OptimizationConfig(amp_dtype="auto"),
    )


def test_engine_auto_amp_dtype(mock_config):
    """Test that amp_dtype='auto' resolves to float32 on CPU."""
    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    # On CPU, 'auto' should now resolve to 'float32'
    expected_device = "cpu"
    expected_amp_dtype = "float32"
    if torch.cuda.is_available():
        expected_device = "cuda"
        expected_amp_dtype = "float16"
        if torch.cuda.is_bf16_supported():
            expected_amp_dtype = "bfloat16"

    assert engine.device.type == expected_device
    assert engine.resolved_amp_dtype == expected_amp_dtype


def test_engine_explicit_amp_dtype(mock_config):
    """Test that explicit amp_dtype is respected."""
    mock_config.optimization.amp_dtype = "float32"
    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)
    assert engine.resolved_amp_dtype == "float32"
