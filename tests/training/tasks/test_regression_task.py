"""Integration tests for RegressionTask."""

from llm.core.mlp import MLP
from llm.data.modules.synthetic import SyntheticDataModule
from llm.runtime.model_factory import MODEL_REGISTRY
from llm.training.core.config import Config, ModelConfig, TrainingConfig
from llm.training.tasks.regression_task import RegressionTask


def test_regression_task_build_model_uses_model_registry():
    config = Config(
        training=TrainingConfig(task="regression", num_samples=10),
        model=ModelConfig(hidden_size=32, intermediate_size=64),
    )
    task = RegressionTask(config, data_module=SyntheticDataModule(config))

    model = task.build_model()

    assert isinstance(model, MLP)
    assert "regression_mlp" in MODEL_REGISTRY.names()
