import torch

from llm.evaluation.metrics.perplexity import PerplexityMetric


def test_perplexity_basic():
    metric = PerplexityMetric()

    # Simple case: perfect prediction
    logits = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])  # Model predicts class 1
    labels = torch.tensor([[1, 1]])

    result = metric.compute(logits, labels)
    assert "perplexity" in result
    assert result["perplexity"] < 1.5  # Should be very low for near-perfect prediction
