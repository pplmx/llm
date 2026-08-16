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

        # RIL ISS-192: ``LMTask.predict`` clamps inputs to the model's
        # context window (``min(max_seq_len, model.max_seq_len)``) while the
        # references are padded to the dataset's ``max_seq_len`` — for a
        # small-context model the predictions are *narrower* than the
        # references, so the naive ``references[:, 1:]`` yields more label
        # positions than logits rows and ``cross_entropy`` raises a shape
        # error mid-evaluation. Slice the labels to the prediction horizon;
        # the truncated tail was never scored anyway.
        label_width = min(references.shape[1] - 1, _seq_len - 1)
        labels = references[:, 1 : 1 + label_width].contiguous().view(-1)

        if logits.shape[0] == 0 or labels.numel() == 0:
            return {"perplexity": float("inf")}

        if self.ignore_index is not None and labels.numel() > 0 and bool((labels == self.ignore_index).all().item()):
            # Every shift-target is ignored (e.g. a 1-token corpus whose
            # shifted labels are all -100): ``cross_entropy`` with
            # ``reduction='mean'`` and ``ignore_index`` averages over ZERO
            # valid elements and returns NaN. Return the documented
            # ``inf`` (undefined perplexity) instead — NaN would serialize
            # to JSON ``null`` and poison the report (RIL ISS-055).
            return {"perplexity": float("inf")}

        kwargs = {"ignore_index": self.ignore_index} if self.ignore_index is not None else {}
        loss = functional.cross_entropy(logits, labels, reduction="mean", **kwargs)
        perplexity = torch.exp(loss).item()

        return {"perplexity": perplexity}
