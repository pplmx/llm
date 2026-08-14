import pytest
import torch

from llm.data.modules.synthetic import SyntheticDataModule
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.engine import TrainingEngine, _cuda_usable
from llm.training.tasks.lm_task import LanguageModelingTask


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
    if _cuda_usable(0):
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


@pytest.mark.heavy
def test_engine_restores_global_step_on_resume(tmp_path, mock_config):
    """Regression (RIL round-47): ``global_step`` must survive a resume.

    Previously every resume restarted the step counter at 0, silently
    re-arming ``max_steps`` (a run stopped at 3 steps resumed and trained 3
    more — and worse, an AdaLoRA resumed run could crash on the next prune
    cadence when the budget rose above the already-reduced effective rank).
    The resumed engine must start from the checkpointed counter.
    """
    mock_config.training.epochs = 5
    mock_config.training.max_steps = 3
    mock_config.training.run_validation = False
    mock_config.optimization.use_compile = False
    mock_config.checkpoint.save_interval = 1
    mock_config.checkpoint.checkpoint_dir = str(tmp_path / "ckpt")
    mock_config.checkpoint.resume_from_checkpoint = None

    dm = SyntheticDataModule(mock_config)
    dm.setup()
    task = LanguageModelingTask(mock_config, dm)

    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)
    # SyntheticDataModule yields float activations, which the LM embedding
    # rejects — drive the engine with a long-token loader (same as
    # test_engine_stops_at_max_steps).
    from torch.utils.data import DataLoader, TensorDataset

    ids = torch.randint(0, mock_config.model.vocab_size, (8, 16), dtype=torch.long)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=2)

    engine.run()
    assert engine.global_step == 3

    # Simulate a restart: fresh engine, same config + task, resume from the
    # checkpoint written at the max_steps cap.
    mock_config.checkpoint.resume_from_checkpoint = str(tmp_path / "ckpt" / "epoch_1.pt")
    engine_resumed = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)

    # start_epoch AND global_step must be restored (not re-armed at 0).
    assert engine_resumed.start_epoch == 1
    assert engine_resumed.global_step == 3, (
        f"resumed engine must continue from step 3, got {engine_resumed.global_step} "
        "(global_step was not persisted across resume)"
    )


@pytest.mark.heavy
def test_engine_epoch_avg_not_deflated_by_max_steps(mock_config):
    """Regression (RIL round-47): a ``max_steps`` cap that truncates the
    final epoch must not deflate the epoch average.

    Previously ``batch_count`` was set to the FULL epoch length even when
    only some micro-batches were processed, so the partial epoch's avg_loss
    was divided by the full denominator — corrupting save_best /
    EarlyStopping / ReduceLROnPlateau with an artificially good metric.
    """
    mock_config.training.epochs = 1
    mock_config.training.max_steps = 2  # truncate an 8-sample/4-batch epoch at 2 batches
    mock_config.training.run_validation = False
    mock_config.optimization.use_compile = False
    mock_config.optimization.gradient_accumulation_steps = 1

    dm = SyntheticDataModule(mock_config)
    dm.setup()

    class _ConstLossTask(LanguageModelingTask):
        """Deterministic 2.0 loss per batch — the average must be exactly 2.0
        regardless of how many batches actually ran."""

        def train_step(self, batch, model, criterion):
            return torch.tensor(2.0, requires_grad=True), {"loss": 2.0}

    task = _ConstLossTask(mock_config, dm)
    engine = TrainingEngine(mock_config, task, rank=0, world_size=1, data_module=dm)
    from torch.utils.data import DataLoader, TensorDataset

    ids = torch.randint(0, mock_config.model.vocab_size, (8, 16), dtype=torch.long)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=2)  # 4 batches/epoch
    engine.val_dataloader = None

    avg = engine._run_epoch(0)
    # 2 processed batches x 2.0 / 2 processed batches == 2.0 (not 4.0/4 == 1.0).
    assert avg == pytest.approx(2.0), f"partial-epoch average must divide by processed batch count, got {avg}"


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
