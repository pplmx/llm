import torch
from torch.nn import functional

from llm.evaluation.metrics.base import BaseMetric


class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling evaluation."""

    name = "perplexity"

    def compute(self, predictions: torch.Tensor, references: torch.Tensor) -> dict:
        """Compute perplexity.

        Args:
            predictions: Logits tensor of shape (batch, seq, vocab)
            references: Target token IDs of shape (batch, seq)

        Returns:
            Dictionary with perplexity score
        """
        _batch_size, _seq_len, vocab_size = predictions.shape

        logits = predictions[:, :-1, :].contiguous().view(-1, vocab_size)
        labels = references[:, 1:].contiguous().view(-1)

        loss = functional.cross_entropy(logits, labels, reduction="mean")
        perplexity = torch.exp(loss).item()

        return {"perplexity": perplexity}
