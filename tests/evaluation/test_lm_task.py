import pytest
import torch

from llm.evaluation.eval_tasks.lm_task import LMTask
from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.evaluation.runner import EvaluationRunner
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


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


def test_perplexity_metric_accepts_1d_references():
    """A single-sequence (1-D) reference must broadcast to one batch row, not
    crash with an opaque IndexError on ``references.shape[1]`` (eval deep-dive
    F1)."""
    metric = PerplexityMetric()
    logits = torch.zeros(1, 3, 10)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    labels_2d = metric.compute(logits, torch.tensor([[1, 2, 0]]))["perplexity"]
    labels_1d = metric.compute(logits, [1, 2, 0])["perplexity"]
    labels_1d_tensor = metric.compute(logits, torch.tensor([1, 2, 0]))["perplexity"]
    for value in (labels_1d, labels_1d_tensor):
        assert value == pytest.approx(labels_2d)


def test_perplexity_metric_ragged_references_clear_error():
    """Ragged reference lists raise a metric-level ValueError (not torch's raw
    ValueError deep inside cross_entropy) — eval deep-dive F1."""
    metric = PerplexityMetric()
    logits = torch.zeros(1, 3, 10)
    with pytest.raises(ValueError, match="rectangular"):
        metric.compute(logits, [[1, 2, 0], [1, 2]])


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


def test_lm_task_predict_variable_length_batches(tmp_path):
    """``LMTask.predict`` must handle variable-length inputs split across
    batches. The previous per-batch ``max_len`` padding made batches differ
    in seq dim, so the final ``torch.cat(results, dim=0)`` raised
    ``RuntimeError: Sizes of tensors must match except in dimension 0``
    (it only escaped today because ``TextDataset`` pre-pads to 128)."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello\n", encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=2)

    class MockModel:
        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, task.tokenizer.vocab_size)

    inputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8]), torch.tensor([9]), torch.tensor([10, 11])]
    out = task.predict(MockModel(), inputs)

    # Padded to the global max sequence length (5), one row per input.
    assert out.shape == (4, 5, task.tokenizer.vocab_size)


def test_lm_task_predict_clamps_to_model_capacity(tmp_path):
    """RIL ISS-130: ``predict`` must clamp the batch to the model's OWN
    context window.

    The old code hardcoded ``TextDataset(max_seq_len=128)`` and padded every
    batch to the global max — a model with ``max_seq_len < 128`` then received
    128-token rows and crashed against its positional-encoding table
    ("Sequence endpoint 128 exceeds maximum sequence length"). The batch's seq
    dim must never exceed what the model can attend to.
    """
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 8, encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=4, max_seq_len=128)

    seq_len_seen: list[int] = []

    class SmallModel:
        max_seq_len = 32

        def __init__(self):
            self.vocab_size = task.tokenizer.vocab_size

        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            seq_len_seen.append(seq)
            return torch.zeros(batch, seq, self.vocab_size)

    task.predict(SmallModel(), [torch.tensor([1] * 128, dtype=torch.long)])

    assert seq_len_seen, "model forward must have been called"
    assert all(seq <= 32 for seq in seq_len_seen), f"batch exceeded model capacity: {seq_len_seen}"


def test_lm_task_metric_aligns_clamped_predictions(tmp_path):
    """RIL ISS-192: with a small-context model, the clamped predictions are
    narrower than the references — the perplexity metric must slice the
    labels to the prediction horizon instead of crashing with a cross_entropy
    shape error.

    ``predict`` truncates inputs to ``model.max_seq_len`` (32) while
    ``prepare_data`` yields references padded to the dataset ``max_seq_len``
    (128). Before the fix ``cross_entropy(logits[B,31,V], labels[B,127])``
    raised a RuntimeError, replacing the ISS-130 positional-encoding crash
    with a metric crash one call later."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 8, encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=4, max_seq_len=128)
    inputs, references = task.prepare_data("val")
    assert all(len(r) == 128 for r in references), "references must be padded to dataset max_seq_len"

    class SmallModel:
        max_seq_len = 32

        def __init__(self):
            self.vocab_size = task.tokenizer.vocab_size

        def __call__(self, input_ids, attn_mask=None):
            batch, seq = input_ids.shape
            return torch.zeros(batch, seq, self.vocab_size)

    predictions = task.predict(SmallModel(), inputs)
    assert predictions.shape == (len(inputs), 32, task.tokenizer.vocab_size)

    result = PerplexityMetric(ignore_index=-100).compute(predictions, references)
    assert "perplexity" in result
    assert result["perplexity"] != float("inf")  # scored over real tokens, not a shape crash


def test_lm_task_respects_explicit_max_seq_len(tmp_path):
    """RIL ISS-130: the caller-settable ``max_seq_len`` (not a hardcoded 128)
    is passed through to the dataset truncation."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world hello world\n" * 8, encoding="utf-8")

    task = LMTask(dataset_path=str(corpus), batch_size=2, max_seq_len=16)

    inputs, _ = task.prepare_data("val")
    assert all(len(x) <= 16 for x in inputs), "dataset must be truncated to max_seq_len"


def test_lm_task_uses_explicit_tokenizer_not_corpus(tmp_path):
    """RIL ISS-195: a caller-supplied tokenizer (bound to the model's vocab
    during training) must win over the corpus-derived one.

    Without this, evaluating a model trained with an arbitrary tokenizer
    re-derives a character tokenizer from the eval corpus, so the vocab ids
    fed to the model never match its trained vocabulary — the reported
    perplexity is meaningful only by accident.
    """
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 4, encoding="utf-8")

    model_tok = TokenizerFactory.from_printable_corpus()
    task = LMTask(dataset_path=str(corpus), batch_size=2, tokenizer=model_tok)

    assert task.tokenizer is model_tok
    assert task.pad_token_id == model_tok.pad_token_id
    # The eval dataset must be encoded with the model's vocab, not the corpus.
    inputs, _ = task.prepare_data("val")
    assert all(x.max().item() < model_tok.vocab_size for x in inputs)


def test_lm_task_tokenizer_defaults_to_corpus_derived(tmp_path):
    """RIL ISS-195: without an explicit tokenizer, behavior is unchanged —
    ``LMTask`` derives a character tokenizer from the eval corpus (backward
    compatibility with all existing callers)."""
    corpus = tmp_path / "eval.txt"
    corpus.write_text("hello world\n" * 2, encoding="utf-8")

    task = LMTask(dataset_path=str(corpus))
    assert isinstance(task.tokenizer, SimpleCharacterTokenizer)
    # Corpus-derived vocab covers the corpus characters.
    assert task.tokenizer.encode("h")[0] is not None  # encodes without KeyError
