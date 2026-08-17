from importlib import import_module

from llm.evaluation.metrics.base import BaseMetric


class RougeMetric(BaseMetric):
    """ROUGE metric for generation tasks.

    Requires the ``rouge-score`` package, available via the ``[eval]`` extra
    (``pip install llm[eval]``).

    The ``rouge_score`` import is deferred to :meth:`compute` (and the
    scorer is built lazily on first use) so the class can be instantiated
    on hosts without ``rouge-score`` installed — the same soft-dependency
    contract as :class:`BleuMetric` and :class:`ChrFMetric`.
    """

    name = "rouge"

    def __init__(self, rouge_types=None):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self._scorer = None

    @staticmethod
    def _build_scorer(rouge_types: list[str]):
        """Import ``rouge_score`` lazily and build a ``RougeScorer``.

        Raises:
            ImportError: with an actionable install hint if
                ``rouge-score`` is not installed.
        """
        try:
            # Read the submodule off the parent package first so callers
            # can patch ``rouge_score`` in sys.modules (test contract),
            # falling back to a real submodule import on first use.
            rouge_module = import_module("rouge_score")
            rouge_scorer = rouge_module.__dict__.get("rouge_scorer") or import_module("rouge_score.rouge_scorer")
        except ImportError as exc:
            raise ImportError(
                "rouge-score is an optional evaluation dependency. Install with `pip install 'llm[eval]'`."
            ) from exc
        return rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def compute(self, predictions: list, references: list) -> dict:
        # Empty inputs — nothing to score, and we shouldn't require the
        # optional dependency just to short-circuit. Every sibling metric
        # reports ``0.0`` on empty input (``BleuMetric`` -> ``{"bleu": 0.0}``,
        # ``AccuracyMetric``/``F1Metric`` -> 0.0); an empty ``{}`` made the
        # per-dimension keys silently vanish from eval output and consumers
        # doing ``results["rouge-1"]`` hit a KeyError (round-73 FINDING 5).
        if not predictions:
            return {t.replace("rouge", "rouge-").lower(): 0.0 for t in self.rouge_types}

        if self._scorer is None:
            self._scorer = self._build_scorer(self.rouge_types)

        results = {}
        for pred, ref in zip(predictions, references, strict=True):
            scores = self._scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                key = rouge_type.replace("rouge", "rouge-").lower()
                if key not in results:
                    results[key] = []
                results[key].append(scores[rouge_type].fmeasure)

        return {k: sum(v) / len(v) for k, v in results.items()}


class BleuMetric(BaseMetric):
    """BLEU metric for generation tasks.

    Requires the ``sacrebleu`` package, available via the ``[eval]`` extra
    (``pip install llm[eval]``).  The import is deferred to :meth:`compute`
    so the class can be instantiated on hosts without ``sacrebleu``
    installed — the same soft-dependency contract as
    :class:`RougeMetric`.
    """

    name = "bleu"

    def compute(self, predictions: list, references: list) -> dict:
        # Empty inputs — nothing to score, and we shouldn't require the
        # optional dependency just to short-circuit (sacrebleu raises on an
        # empty corpus). Matches the ``0.0`` convention of
        # :class:`AccuracyMetric` / :class:`F1Metric`.
        if not predictions:
            return {"bleu": 0.0}

        try:
            sacrebleu = import_module("sacrebleu")
        except ImportError as exc:
            raise ImportError(
                "sacrebleu is an optional evaluation dependency. Install with `pip install 'llm[eval]'`."
            ) from exc

        refs = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        return {"bleu": bleu.score}


class ChrFMetric(BaseMetric):
    """chrF metric for generation tasks.

    Requires the ``sacrebleu`` package, available via the ``[eval]`` extra
    (``pip install llm[eval]``).  The import is deferred to :meth:`compute`
    so the class can be instantiated on hosts without ``sacrebleu``
    installed — the same soft-dependency contract as
    :class:`RougeMetric`.
    """

    name = "chrf"

    def compute(self, predictions: list, references: list) -> dict:
        # Empty inputs — nothing to score, and we shouldn't require the
        # optional dependency just to short-circuit (sacrebleu raises on an
        # empty corpus). Matches the ``0.0`` convention of
        # :class:`AccuracyMetric` / :class:`F1Metric`.
        if not predictions:
            return {"chrf": 0.0}

        try:
            sacrebleu = import_module("sacrebleu")
        except ImportError as exc:
            raise ImportError(
                "sacrebleu is an optional evaluation dependency. Install with `pip install 'llm[eval]'`."
            ) from exc

        refs = [[r] for r in references]
        chrf = sacrebleu.corpus_chrf(predictions, refs)
        return {"chrf": chrf.score}
