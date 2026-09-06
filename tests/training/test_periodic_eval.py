"""Tests for periodic in-training evaluation (RIL TASK-329).

Covers the ``build_periodic_eval_callback`` wiring — disabled-by-default,
corpus resolution, missing-corpus and collective-strategy refusal — plus the
``EvaluationCallback`` rank/eval-mode guards on a real engine stub.
"""

from __future__ import annotations

import pytest
import torch

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.callbacks import EvaluationCallback
from llm.training.core.config import Config, ModelConfig, OptimizationConfig, TrainingConfig
from llm.training.core.periodic_eval import build_periodic_eval_callback


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "eval.txt"
    p.write_text("hello world\n" * 8, encoding="utf-8")
    return str(p)


@pytest.fixture
def tokenizer():
    return SimpleCharacterTokenizer(["hello world"])


def test_eval_interval_zero_returns_none(tokenizer, corpus):
    config = Config()
    config.training.eval_interval = 0
    config.data.eval_dataset_path = corpus
    assert build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=1) is None, (
        "default eval_interval=0 must disable the callback"
    )


def test_enabled_builds_callback(tokenizer, corpus):
    config = Config()
    config.training.eval_interval = 25
    config.data.eval_dataset_path = corpus
    cb = build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=1)
    assert isinstance(cb, EvaluationCallback)
    assert cb.eval_interval == 25


def test_corpus_resolution_falls_back_to_dataset_path(tokenizer, tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("the quick brown fox\n" * 4, encoding="utf-8")
    config = Config()
    config.training.eval_interval = 10
    config.data.dataset_path = str(p)  # no dedicated eval/val path
    cb = build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=1)
    assert isinstance(cb, EvaluationCallback)


def test_enabled_without_corpus_raises(tokenizer):
    config = Config()
    config.training.eval_interval = 10
    # no dataset_path / val_dataset_path / eval_dataset_path
    with pytest.raises(ValueError, match="requires a text corpus"):
        build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=1)


@pytest.mark.parametrize("strategy", ["tp", "fsdp", "pp", "3d"])
def test_collective_forward_strategies_refused(tokenizer, corpus, strategy):
    config = Config()
    config.training.eval_interval = 10
    config.data.eval_dataset_path = corpus
    config.distributed.parallel_strategy = strategy
    with pytest.raises(ValueError, match="not supported with"):
        build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=2)


def test_ddp_and_zero_strategies_allowed(tokenizer, corpus):
    for strategy in ("ddp", "zero"):
        config = Config()
        config.training.eval_interval = 10
        config.data.eval_dataset_path = corpus
        config.distributed.parallel_strategy = strategy
        cb = build_periodic_eval_callback(config, model_max_seq_len=16, tokenizer=tokenizer, world_size=2)
        assert isinstance(cb, EvaluationCallback), f"{strategy} must allow periodic eval"


# --------------------------------------------------------------------------- #
# EvaluationCallback guards
# --------------------------------------------------------------------------- #


class _Recorder:
    def __init__(self):
        self.logged: list[dict] = []

    def log_metrics(self, metrics: dict) -> None:
        self.logged.append(metrics)


class _FakeRunner:
    def __init__(self):
        self.calls = 0
        self.results = {"perplexity": 2.5}

    def run(self, model):
        self.calls += 1
        return self.results


def test_callback_runs_only_on_rank_zero_and_restores_train_mode():
    class _StubModel:
        """Plain object (not nn.Module) exposing the ``training``/``train``/
        ``eval`` surface the callback toggles."""

        def __init__(self):
            self.training = True

        def train(self, mode: bool = True):
            self.training = mode
            return self

        def eval(self):
            self.training = False
            return self

    runner = _FakeRunner()
    recorder = _Recorder()
    model = _StubModel()

    class _StubEngine:
        rank = 0

    engine = _StubEngine()
    engine.model = model
    engine.log_metrics = recorder.log_metrics

    cb = EvaluationCallback(runner, eval_interval=2)
    cb.set_engine(engine)

    assert model.training is True, "stub starts in train mode"
    cb.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(1.0), metrics={})
    assert runner.calls == 0, "non-interval steps must not trigger eval"
    cb.on_train_step_end(epoch=0, batch_idx=1, loss=torch.tensor(1.0), metrics={})
    assert runner.calls == 1, "eval must fire on the interval step"
    assert model.training is True, "callback must restore train mode after eval"
    assert len(recorder.logged) == 1

    # Non-rank-0 engine: no eval, no logs.
    engine.rank = 1
    engine2 = _StubEngine()
    engine2.rank = 1
    engine2.model = model
    engine2.log_metrics = recorder.log_metrics
    cb2 = EvaluationCallback(runner, eval_interval=1)
    cb2.set_engine(engine2)
    cb2.on_train_step_end(epoch=0, batch_idx=0, loss=torch.tensor(1.0), metrics={})
    assert runner.calls == 1, "rank>0 must not run periodic eval"
    assert len(recorder.logged) == 1, "rank>0 must not log metrics"


def test_engine_runs_periodic_eval_end_to_end(tmp_path):
    """End-to-end (RIL TASK-329): with ``training.eval_interval > 0`` the
    engine's callback runs a real LM evaluation on the model mid-training and
    logs the metrics at rank 0."""
    import logging

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask

    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 8, encoding="utf-8")

    config = Config(
        model=ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=64),
        training=TrainingConfig(
            batch_size=2,
            num_samples=10,
            epochs=1,
            run_validation=False,
            eval_interval=2,  # fire on batch_idx 1 and 3
        ),
        optimization=OptimizationConfig(amp_dtype="auto", use_compile=False),
    )
    config.data.eval_dataset_path = str(corpus)
    tokenizer = SimpleCharacterTokenizer(["hello world"])

    dm = SyntheticDataModule(config)
    dm.setup()
    task = LanguageModelingTask(config, dm)

    cb = build_periodic_eval_callback(config, model_max_seq_len=64, tokenizer=tokenizer, world_size=1)
    assert isinstance(cb, EvaluationCallback), "eval_interval>0 must build the callback"

    engine = TrainingEngine(config, task, rank=0, world_size=1, data_module=dm, callbacks=[cb])
    ids = torch.randint(0, config.model.vocab_size, (8, 16), dtype=torch.long)
    engine.is_streaming = False
    engine.dataloader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=2)  # 4 batches/epoch
    engine.val_dataloader = None

    # The engine's Logger proxies to stdlib ``rank_0`` with propagate=False,
    # so caplog can't see it — capture via our own handler on that logger.
    emitted: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            emitted.append(record.getMessage())

    engine_logger = logging.getLogger("rank_0")
    engine_logger.setLevel(logging.INFO)
    capture = _Capture()
    engine_logger.addHandler(capture)
    try:
        engine.run()
    finally:
        engine_logger.removeHandler(capture)

    assert any("Evaluation metrics" in msg for msg in emitted), "periodic eval results must be logged during training"
    assert any("perplexity" in msg for msg in emitted), "the eval must report its metric values"
