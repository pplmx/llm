from llm.evaluation.metrics.base import BaseMetric


class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks."""

    name = "accuracy"

    def compute(self, predictions: list, references: list) -> dict:
        """Compute accuracy."""
        correct = sum(p == r for p, r in zip(predictions, references, strict=True))
        acc = correct / len(predictions) if predictions else 0.0
        return {"accuracy": acc}


class F1Metric(BaseMetric):
    """F1 score metric for classification tasks."""

    name = "f1"

    def __init__(self, average: str = "macro"):
        self.average = average

    def compute(self, predictions: list, references: list) -> dict:
        """Compute F1 score using sklearn."""
        from sklearn.metrics import f1_score

        f1 = f1_score(references, predictions, average=self.average, zero_division=0)
        return {"f1": f1}
