import pytest

from llm.models.decoder import DecoderModel
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig


@pytest.fixture
def tiny_config():
    """Provides a minimal configuration for fast unit testing."""
    return Config(
        model=ModelConfig(vocab_size=100, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16),
        training=TrainingConfig(batch_size=2, epochs=1, num_samples=10),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Provides a minimal DecoderModel instance."""
    return DecoderModel(
        vocab_size=tiny_config.model.vocab_size,
        hidden_size=tiny_config.model.hidden_size,
        num_layers=tiny_config.model.num_layers,
        num_heads=tiny_config.model.num_heads,
        max_seq_len=tiny_config.model.max_seq_len,
    )
