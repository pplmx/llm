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
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise ImportError(
                "rouge-score is an optional evaluation dependency. Install with `pip install 'llm[eval]'`."
            ) from exc
        return rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def compute(self, predictions: list, references: list) -> dict:
        # Empty inputs — nothing to score, and we shouldn't require the
        # optional dependency just to short-circuit.
        if not predictions:
            return {}

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

    Requires the ``sacrebleu`` package, available via the ``[eval]`` extra.
    """

    name = "bleu"

    def compute(self, predictions: list, references: list) -> dict:
        import sacrebleu

        refs = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        return {"bleu": bleu.score}


class ChrFMetric(BaseMetric):
    """chrF metric for generation tasks.

    Requires the ``sacrebleu`` package, available via the ``[eval]`` extra.
    """

    name = "chrf"

    def compute(self, predictions: list, references: list) -> dict:
        import sacrebleu

        refs = [[r] for r in references]
        chrf = sacrebleu.corpus_chrf(predictions, refs)
        return {"chrf": chrf.score}
