import sys

import pytest

from llm.evaluation.metrics.generation import BleuMetric, ChrFMetric, RougeMetric


def test_rouge_basic():
    pytest.importorskip("rouge_score", reason="rouge_score is an optional eval dependency")
    pytest.importorskip("nltk", reason="nltk is an optional eval dependency (rouge stemmer)")
    metric = RougeMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the rug"]

    result = metric.compute(predictions, references)
    assert result["rouge-l"] > 0.5


def test_rouge_custom_types():
    pytest.importorskip("rouge_score", reason="rouge_score is an optional eval dependency")
    pytest.importorskip("nltk", reason="nltk is an optional eval dependency (rouge stemmer)")
    metric = RougeMetric(rouge_types=["rouge1"])
    predictions = ["a b c"]
    references = ["a b c"]

    result = metric.compute(predictions, references)
    assert "rouge-1" in result
    assert result["rouge-1"] == pytest.approx(1.0)


def test_bleu_basic():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = BleuMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert result["bleu"] == pytest.approx(100.0, rel=0.1)


def test_bleu_partial_overlap():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = BleuMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the dog sat on the rug"]

    result = metric.compute(predictions, references)
    assert 0 < result["bleu"] <= 100.0


def test_chrf_basic():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = ChrFMetric()
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    result = metric.compute(predictions, references)
    assert "chrf" in result


def test_chrf_no_overlap():
    pytest.importorskip("sacrebleu", reason="sacrebleu is an optional eval dependency")
    metric = ChrFMetric()
    predictions = ["cat"]
    references = ["dog"]

    result = metric.compute(predictions, references)
    assert "chrf" in result
    assert result["chrf"] < 30.0


@pytest.mark.parametrize(("metric_cls", "key"), [(BleuMetric, "bleu"), (ChrFMetric, "chrf")])
def test_generation_metric_empty_references_returns_zero(metric_cls, key):
    """Empty references with non-empty predictions must return the same 0.0
    empty contract as empty predictions — sacrebleu raises a
    raw IndexError on an empty reference corpus, so the metric must
    short-circuit before calling it."""
    metric = metric_cls()
    result = metric.compute(["the cat sat on the mat"], [])
    assert result[key] == 0.0


# --------------------------------------------------------------------------- #
# Soft-dependency contract for RougeMetric
#
# ``RougeMetric`` must be constructible without ``rouge-score`` installed
# (same contract as ``BleuMetric`` / ``ChrFMetric`` which import sacrebleu
# lazily inside ``compute``). We monkey-patch ``sys.modules`` so the test
# runs in *every* environment — with or without rouge_score present.
# --------------------------------------------------------------------------- #


def test_rouge_metric_constructible_without_rouge_score(monkeypatch):
    """``RougeMetric()`` must not import ``rouge_score`` eagerly."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    assert metric.rouge_types == ["rouge1", "rouge2", "rougeL"]


def test_rouge_metric_custom_types_without_rouge_score(monkeypatch):
    """Custom ``rouge_types`` are stored at init without triggering import."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric(rouge_types=["rouge1"])
    assert metric.rouge_types == ["rouge1"]


def test_rouge_metric_compute_raises_without_rouge_score(monkeypatch):
    """``compute`` on a host without rouge_score raises a clear ImportError."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    with pytest.raises(ImportError, match="rouge-score"):
        metric.compute(["pred"], ["ref"])


def test_rouge_metric_empty_inputs_without_rouge_score(monkeypatch):
    """Empty predictions/references report ``0.0`` per dimension WITHOUT
    importing rouge_score (the short-circuit precedes the import). Matches the
    sibling-metric convention; an empty ``{}`` made rouge keys vanish from eval
    output (FINDING 5)."""
    monkeypatch.setitem(sys.modules, "rouge_score", None)

    metric = RougeMetric()
    result = metric.compute([], [])
    assert result == {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


# --------------------------------------------------------------------------- #
# Soft-dependency contract for BleuMetric / ChrFMetric
#
# Both must be constructible without ``sacrebleu`` installed, and their
# ``compute`` must raise a clear ``ImportError`` (not a bare ModuleNotFoundError)
# when sacrebleu is missing — same contract as RougeMetric above.
# --------------------------------------------------------------------------- #


def test_bleu_constructible_without_sacrebleu(monkeypatch):
    """BleuMetric() must not import sacrebleu eagerly."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = BleuMetric()
    assert metric.name == "bleu"


def test_bleu_compute_raises_without_sacrebleu(monkeypatch):
    """compute without sacrebleu raises a clear ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = BleuMetric()
    with pytest.raises(ImportError, match=r"pip install 'llm\[eval\]'"):
        metric.compute(["pred"], ["ref"])


def test_chrf_constructible_without_sacrebleu(monkeypatch):
    """ChrFMetric() must not import sacrebleu eagerly."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = ChrFMetric()
    assert metric.name == "chrf"


def test_chrf_compute_raises_without_sacrebleu(monkeypatch):
    """compute without sacrebleu raises a clear ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = ChrFMetric()
    with pytest.raises(ImportError, match=r"pip install 'llm\[eval\]'"):
        metric.compute(["pred"], ["ref"])


def test_bleu_metric_empty_inputs_without_sacrebleu(monkeypatch):
    """Empty predictions return ``{"bleu": 0.0}`` without importing sacrebleu,
    matching RougeMetric's empty-input contract (and F1Metric's 0.0)."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = BleuMetric()
    result = metric.compute([], [])
    assert result == {"bleu": 0.0}


def test_chrf_metric_empty_inputs_without_sacrebleu(monkeypatch):
    """Empty predictions return ``{"chrf": 0.0}`` without importing sacrebleu."""
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    metric = ChrFMetric()
    result = metric.compute([], [])
    assert result == {"chrf": 0.0}


# --------------------------------------------------------------------------- #
# Predictions/references alignment
# --------------------------------------------------------------------------- #


class _FakeScorer:
    """Stand-in for ``rouge_score.RougeScorer``."""

    def __init__(self, rouge_types, use_stemmer=True):
        self._types = rouge_types

    def score(self, reference, prediction):
        class _S:
            fmeasure = 0.5

        return {t: _S() for t in self._types}


class _FakeSacrebleu:
    """Stand-in for ``sacrebleu`` recording what reached the scorer."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def corpus_bleu(self, predictions, references):
        self.calls.append((len(predictions), len(references)))
        return type("_R", (), {"score": 50.0})()

    def corpus_chrf(self, predictions, references):
        self.calls.append((len(predictions), len(references)))
        return type("_R", (), {"score": 50.0})()


@pytest.fixture
def fake_metric_deps(monkeypatch):
    """Install fake ``rouge_score`` / ``sacrebleu`` so the REAL compute path runs.

    The rest of this file monkeypatches ``sys.modules[name] = None`` to model a
    host WITHOUT the optional deps, which means every assertion about scoring
    stops at the ImportError. These fakes let the alignment logic below be
    exercised for real; a scoring library is not what is under test here.
    """
    import types

    rouge_pkg = types.ModuleType("rouge_score")
    rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
    rouge_scorer.RougeScorer = _FakeScorer
    rouge_pkg.rouge_scorer = rouge_scorer

    sacrebleu = _FakeSacrebleu()
    monkeypatch.setitem(sys.modules, "rouge_score", rouge_pkg)
    monkeypatch.setitem(sys.modules, "rouge_score.rouge_scorer", rouge_scorer)
    monkeypatch.setitem(sys.modules, "sacrebleu", sacrebleu)
    return sacrebleu


class TestMetricsRejectMisalignedInputs:
    """A length mismatch must fail loudly for EVERY generation metric.

    Before this, the three metrics disagreed: ``RougeMetric`` used
    ``zip(..., strict=True)`` and raised ``zip() argument 2 is shorter than
    argument 1``, while ``BleuMetric`` / ``ChrFMetric`` forwarded the ragged
    pair to sacrebleu, which scored the truncated set and returned a
    plausible-looking number. A silently wrong BLEU is worse than a crash — a
    task whose ``predict`` drops rows would report a score computed from the
    wrong pairs, with nothing in the output saying so.

    The runner calls ``metric.compute(predictions, references)`` straight from
    ``task.predict`` / ``task.prepare_data`` with no length check of its own, so
    the metric is the last place the mismatch can be caught.
    """

    @pytest.mark.parametrize("metric_cls", [RougeMetric, BleuMetric, ChrFMetric])
    def test_mismatch_raises_with_a_usable_message(self, metric_cls, fake_metric_deps):
        metric = metric_cls()
        with pytest.raises(ValueError, match="must be aligned") as excinfo:
            metric.compute(["a", "b", "c"], ["x"])

        message = str(excinfo.value)
        assert metric.name in message, "the error must name the metric that rejected the input"
        assert "3" in message, "the prediction count must appear"
        assert "1" in message, "the reference count must appear"

    @pytest.mark.parametrize("metric_cls", [RougeMetric, BleuMetric, ChrFMetric])
    def test_aligned_inputs_still_score(self, metric_cls, fake_metric_deps):
        """The guard must not reject a normal evaluation."""
        out = metric_cls().compute(["a", "b"], ["x", "y"])
        assert isinstance(out, dict), "a metric must return a dict of scores"
        assert out, "an aligned batch must still produce scores"

    def test_bleu_never_forwards_a_ragged_pair_to_sacrebleu(self, fake_metric_deps):
        """The specific regression: sacrebleu used to receive the mismatch.

        Asserting the raise alone would not prove the call was prevented; this
        checks the library saw no call at all.
        """
        with pytest.raises(ValueError, match="must be aligned"):
            BleuMetric().compute(["a", "b", "c"], ["x"])
        assert fake_metric_deps.calls == [], "sacrebleu must not be invoked with misaligned inputs"

    def test_rouge_message_replaces_the_bare_zip_error(self, fake_metric_deps):
        """``RougeMetric`` raised a message naming neither metric nor arrays."""
        with pytest.raises(ValueError, match="must be aligned") as excinfo:
            RougeMetric().compute(["a", "b"], ["x"])
        assert "zip()" not in str(excinfo.value), "the bare zip() message gives the user nothing to act on"
