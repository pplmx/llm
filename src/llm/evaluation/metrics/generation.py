import sacrebleu
from rouge_score import rouge_scorer

from llm.evaluation.metrics.base import BaseMetric


class RougeMetric(BaseMetric):
    """ROUGE metric for generation tasks."""

    name = "rouge"

    def __init__(self, rouge_types=None):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)

    def compute(self, predictions: list, references: list) -> dict:
        results = {}

        for pred, ref in zip(predictions, references, strict=True):
            scores = self.scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                key = rouge_type.replace("rouge", "rouge-").lower()
                if key not in results:
                    results[key] = []
                results[key].append(scores[rouge_type].fmeasure)

        return {k: sum(v) / len(v) for k, v in results.items()}


class BleuMetric(BaseMetric):
    """BLEU metric for generation tasks."""

    name = "bleu"

    def compute(self, predictions: list, references: list) -> dict:
        refs = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        return {"bleu": bleu.score}


class ChrFMetric(BaseMetric):
    """chrF metric for generation tasks."""

    name = "chrf"

    def compute(self, predictions: list, references: list) -> dict:
        refs = [[r] for r in references]
        chrf = sacrebleu.corpus_chrf(predictions, refs)
        return {"chrf": chrf.score}
