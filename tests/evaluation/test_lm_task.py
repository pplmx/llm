import pytest
import torch

from llm.evaluation.eval_tasks.lm_task import LMTask
from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.evaluation.runner import EvaluationRunner


def test_perplexity_metric_perfect_prediction():
    metric = PerplexityMetric()
    logits = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])
    labels = torch.tensor([[1, 1]])

    result = metric.compute(logits, labels)
    assert result["perplexity"] < 1.5


def test_perplexity_metric_empty_batch_returns_inf():
    """An empty batch must not crash — return inf (undefined perplexity)."""
    metric = PerplexityMetric()
    logits = torch.empty(0, 4, 10, dtype=torch.float)
    labels = torch.empty(0, 4, dtype=torch.long)
    result = metric.compute(logits, labels)
    assert result == {"perplexity": float("inf")}


def test_perplexity_metric_single_token_returns_finite():
    """A single-token sequence (no shift targets) must not return NaN."""
    metric = PerplexityMetric()
    logits = torch.zeros(1, 1, 10)
    labels = torch.tensor([[1]])
    result = metric.compute(logits, labels)
    assert result["perplexity"] == float("inf")


def test_lm_task_prepare_data_returns_token_tensors(tmp_path):
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 4, encoding="utf-8")

    task = LMTask(dataset_path=str(corpus))
    inputs, refs = task.prepare_data("val")

    assert len(inputs) == 1
    assert len(refs) == 1
    assert inputs[0].dtype == torch.long
    assert refs[0].dtype == torch.long
    assert inputs[0].shape == refs[0].shape


def test_evaluation_runner_reports_perplexity(tmp_path):
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=1)
    runner = EvaluationRunner(task)

    class MockModel:
        def __call__(self, input_ids):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    results = runner.evaluate(MockModel())
    assert results["perplexity"] == pytest.approx(task.tokenizer.vocab_size, rel=1e-4)
