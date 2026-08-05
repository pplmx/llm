import pytest
import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine
from llm.training.tasks.lm_task import LanguageModelingTask
from tests.support.devices import cuda_usable


@pytest.fixture
def mock_config():
    return Config(
        model=ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=64),
        training=TrainingConfig(batch_size=2, num_samples=10),
        optimization=OptimizationConfig(amp_dtype="auto"),
    )


@pytest.mark.heavy
def test_engine_auto_amp_dtype(mock_config):
    """Test that amp_dtype='auto' resolves to float32 on CPU."""
    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    # On CPU, 'auto' should now resolve to 'float32'
    expected_device = "cpu"
    expected_amp_dtype = "float32"
    if cuda_usable():
        expected_device = "cuda"
        expected_amp_dtype = "float16"
        if torch.cuda.is_bf16_supported():
            expected_amp_dtype = "bfloat16"

    assert engine.device.type == expected_device
    assert engine.resolved_amp_dtype == expected_amp_dtype


@pytest.mark.heavy
def test_engine_explicit_amp_dtype(mock_config):
    """Test that explicit amp_dtype is respected."""
    mock_config.optimization.amp_dtype = "float32"
    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)
    assert engine.resolved_amp_dtype == "float32"


@pytest.mark.heavy
def test_engine_stops_at_max_steps(mock_config):
    """Regression: ``training.max_steps`` must cap total optimizer steps.

    Previously the engine never read max_steps, so a config with
    epochs * steps_per_epoch > max_steps trained past the documented cap
    (observed: max_steps=600 config ran 1500 steps).
    """
    mock_config.training.epochs = 5
    mock_config.training.max_steps = 3
    mock_config.training.run_validation = False
    mock_config.optimization.use_compile = False  # isolate the step-cap behavior
    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)
    # SyntheticDataModule yields float activations (regression-style), which
    # the LM embedding rejects — drive the engine with a long-token loader.
    from torch.utils.data import DataLoader, TensorDataset

    ids = torch.randint(0, mock_config.model.vocab_size, (8, 16), dtype=torch.long)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=2)

    engine.run()

    assert engine.global_step == 3


def test_engine_validation_empty_dataloader_skips(mock_config):
    """An empty validation split must skip validation instead of raising
    ZeroDivisionError."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from llm.training.core.engine import TrainingEngine

    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    engine.val_dataloader = DataLoader(TensorDataset(torch.empty(0, 2, dtype=torch.long)))
    engine.val_sampler = None

    result = engine._run_validation_epoch(0)
    assert result is None


def test_engine_log_metrics_logs_at_rank_zero(mock_config, caplog):
    """EvaluationCallback calls engine.log_metrics on its eval interval; the
    method must exist and log at rank 0 (regression: it only existed in a
    type-checker stub, so the callback raised AttributeError at runtime)."""
    from llm.training.core.engine import TrainingEngine

    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    with caplog.at_level("INFO"):
        engine.log_metrics({"accuracy": 0.875, "num_samples": 32})
    assert "accuracy: 0.8750" in caplog.text
    assert "num_samples: 32" in caplog.text


def test_evaluation_callback_fires_without_attribute_error(mock_config):
    """Regression: EvaluationCallback.on_train_step_end called the missing
    engine.log_metrics and crashed. It must run end-to-end on a real engine."""
    from llm.training.core.callbacks import EvaluationCallback
    from llm.training.core.engine import TrainingEngine

    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    class _Runner:
        def run(self, model):
            return {"accuracy": 1.0}

    engine.callbacks = [EvaluationCallback(eval_runner=_Runner(), eval_interval=1)]
    for cb in engine.callbacks:
        cb.set_engine(engine)
    engine._run_callbacks("on_train_step_end", epoch=0, batch_idx=0, loss=torch.tensor(1.0), metrics={"loss": 1.0})
