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
        """Compute F1 score."""
        labels = list(set(predictions + references))
        label_to_idx = {label: i for i, label in enumerate(labels)}

        pred_indices = [label_to_idx[p] for p in predictions]
        ref_indices = [label_to_idx[r] for r in references]

        tp = sum(1 for p, r in zip(pred_indices, ref_indices, strict=True) if p == r)
        fp = sum(1 for p, r in zip(pred_indices, ref_indices, strict=True) if p != r)
        fn = sum(1 for p, r in zip(pred_indices, ref_indices, strict=True) if p != r)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"f1": f1}
