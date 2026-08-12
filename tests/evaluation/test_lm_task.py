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


def test_perplexity_metric_accepts_list_references():
    """PerplexityMetric must accept raw list references (the ``runner.run``
    path) as well as tensors (the ``runner.evaluate`` path)."""
    metric = PerplexityMetric()
    logits = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])
    references = [[1, 1]]

    result = metric.compute(logits, references)
    assert result["perplexity"] < 1.5


def test_perplexity_metric_ignores_minus_100_textdataset_mask():
    """Regression (RIL ISS-041): TextDataset marks padded label slots with
    -100, so the LM eval metric must ignore -100. Using the tokenizer's
    pad_token_id instead made cross_entropy raise "Target -100 is out of
    bounds" on any corpus shorter than max_seq_len."""
    metric = PerplexityMetric(ignore_index=-100)
    logits = torch.tensor(
        [
            [
                [0.0, 10.0],  # perfect prediction of token 1
                [0.0, 10.0],  # perfect prediction of token 1
                [0.0, 0.0],  # padded (label -100) — must be ignored
            ]
        ]
    )
    labels = torch.tensor([[1, 1, -100]])

    result = metric.compute(logits, labels)

    assert result["perplexity"] < 1.5


def test_lm_task_builds_metric_with_minus_100_ignore(tmp_path):
    """The LM eval task must wire the metric to the label mask (-100), not
    the tokenizer's pad id, matching how TextDataset builds its labels."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus))

    assert task.metrics[0].ignore_index == -100


def test_perplexity_metric_ignore_index_skips_pads():
    """``ignore_index`` masks pad positions: garbage logits on pad tokens
    must not affect the score."""
    metric = PerplexityMetric(ignore_index=0)
    logits = torch.tensor(
        [
            [
                [0.0, 10.0],  # perfect prediction of token 1
                [0.0, 10.0],  # perfect prediction of token 1
                [0.0, 0.0],  # pad position (label 0) — must be ignored
            ]
        ]
    )
    labels = torch.tensor([[1, 1, 0]])

    result = metric.compute(logits, labels)
    assert result["perplexity"] < 1.5


def test_evaluation_runner_run_reports_perplexity(tmp_path):
    """Regression: ``EvaluationRunner.run`` (the training-callback path)
    crashed with ``TypeError: list indices must be integers`` because
    ``PerplexityMetric`` indexed raw list references."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=1)
    runner = EvaluationRunner(task)

    class MockModel:
        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    results = runner.run(MockModel())
    assert results["perplexity"] == pytest.approx(task.tokenizer.vocab_size, rel=1e-4)


def test_evaluation_runner_reports_perplexity(tmp_path):
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=1)
    runner = EvaluationRunner(task)

    class MockModel:
        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    results = runner.evaluate(MockModel())
    assert results["perplexity"] == pytest.approx(task.tokenizer.vocab_size, rel=1e-4)


def test_perplexity_metric_all_labels_ignored_returns_inf():
    """Regression (RIL ISS-055): when every shift-target label is the
    ``ignore_index`` (e.g. a 1-token corpus whose shifted labels are all
    ``-100``), ``cross_entropy(..., reduction='mean', ignore_index=-100)``
    averages over zero valid elements and returns NaN. The metric's
    documented convention for undefined perplexity is ``inf`` — NaN would
    then serialize to JSON ``null`` and poison the report."""
    metric = PerplexityMetric(ignore_index=-100)
    logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])  # 2 positions, 1 shift target
    labels = torch.tensor([[-100, -100]])  # the shift target is ignored

    result = metric.compute(logits, labels)

    assert result["perplexity"] == float("inf")


def test_evaluation_runner_run_empty_corpus_no_crash(tmp_path):
    """Regression (RIL ISS-045): an empty eval corpus must not crash
    ``EvaluationRunner.run`` (the training-callback path). ``predict``
    returned ``torch.cat([])`` which raises ``ValueError``; instead it must
    yield an empty prediction and the metric layer reports ``inf``."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=2)
    runner = EvaluationRunner(task)

    class MockModel:
        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    results = runner.run(MockModel())
    assert results["num_samples"] == 0
    assert results["perplexity"] == float("inf")


def test_evaluation_runner_evaluate_empty_corpus_no_crash(tmp_path):
    """Regression (RIL ISS-045): the tensor-coercing ``evaluate`` path must
    also handle an empty corpus without ``torch.stack([])`` / ``torch.cat``
    crashing."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=2)
    runner = EvaluationRunner(task)

    class MockModel:
        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    results = runner.evaluate(MockModel())
    assert results["perplexity"] == float("inf")
