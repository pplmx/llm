import torch
from torch.nn import functional

from llm.evaluation.metrics.base import BaseMetric


class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling evaluation."""

    name = "perplexity"

    def __init__(self, ignore_index: int | None = None) -> None:
        """Create the metric.

        Args:
            ignore_index: Token id to exclude from the loss (typically
                the tokenizer's ``pad_token_id``). When set, positions
                whose label equals ``ignore_index`` are skipped so padded
                sequences are scored only over real tokens. ``None``
                (the default) scores every position.
        """
        self.ignore_index = ignore_index

    def compute(self, predictions: torch.Tensor, references: torch.Tensor | list) -> dict:
        """Compute perplexity.

        Args:
            predictions: Logits tensor of shape (batch, seq, vocab)
            references: Target token IDs of shape (batch, seq). A list
                of equal-length token sequences is coerced to a tensor
                so the metric works through both :meth:`EvaluationRunner.run`
                (raw path) and :meth:`EvaluationRunner.evaluate`.

        Returns:
            Dictionary with perplexity score.  ``inf`` is returned when
            the batch is empty or no shift-targets are available (e.g.
            ``seq == 1``), since perplexity is undefined in those cases.
        """
        if not isinstance(references, torch.Tensor):
            # ``run()`` passes raw (non-tensor) references: LMTask yields
            # equal-length padded sequences, so a list of tensors stacks
            # cleanly; plain lists are coerced elementwise.
            if references and isinstance(references[0], torch.Tensor):
                references = torch.stack(references)
            else:
                references = torch.as_tensor(references, dtype=torch.long)

        batch_size = predictions.shape[0]
        if batch_size == 0:
            return {"perplexity": float("inf")}

        _batch, _seq_len, vocab_size = predictions.shape

        logits = predictions[:, :-1, :].contiguous().view(-1, vocab_size)
        labels = references[:, 1:].contiguous().view(-1)

        if logits.shape[0] == 0:
            return {"perplexity": float("inf")}

        kwargs = {"ignore_index": self.ignore_index} if self.ignore_index is not None else {}
        loss = functional.cross_entropy(logits, labels, reduction="mean", **kwargs)
        perplexity = torch.exp(loss).item()

        return {"perplexity": perplexity}
