from typing import cast

import torch

from llm.data.datasets.text import TextDataset
from llm.evaluation.eval_tasks.base import BaseTask
from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.tokenizer import BaseTokenizer


class LMTask(BaseTask):
    name = "lm"

    def __init__(self, dataset_path: str, batch_size: int = 8, max_seq_len: int | None = None):
        """Perplexity evaluation on a text corpus.

        Args:
            dataset_path: Text corpus file to evaluate.
            batch_size: Batch size for the forward pass.
            max_seq_len: Context window the sequences are truncated to. MUST
                be at least the model's ``max_seq_len`` when the caller knows
                it — the old hardcoded 128 silently crashed any model with a
                smaller context ("Sequence endpoint 128 exceeds maximum
                sequence length", RIL ISS-130). Defaults to 128 for backward
                compatibility with callers that never tuned it.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len or 128
        self.tokenizer = TokenizerFactory.from_dataset_text(dataset_path)
        # Mask padded positions so short trailing sequences are scored
        # only over real tokens. TextDataset marks padded label slots with
        # the standard ignore index -100 (never a real token id), so the
        # metric must ignore -100 — passing the tokenizer's pad_token_id
        # instead crashes cross_entropy with "Target -100 is out of bounds"
        # and silently scores pad tokens when the pad id collides (RIL
        # ISS-041, regression from the ISS-040 label-masking fix).
        self.metrics = [PerplexityMetric(ignore_index=-100)]
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        self.val_dataset = TextDataset(
            file_path=dataset_path,
            tokenizer=cast(BaseTokenizer, self.tokenizer),
            max_seq_len=self.max_seq_len,
        )

    def prepare_data(self, split: str):
        # ``split`` is part of the BaseTask contract but a single-corpus LM
        # task has exactly one (validation) dataset, so train/val/test are
        # the same file by design. Deliberately not filtered.
        inputs = []
        references = []

        for item in self.val_dataset:
            inputs.append(item["input_ids"])
            references.append(item["labels"])

        return inputs, references

    def predict(self, model, inputs: list):
        results = []
        # Pad every batch to the GLOBAL max sequence length (not the
        # batch-local max): the final ``torch.cat(results, dim=0)`` requires
        # a uniform seq dim, and per-batch padding previously crashed on any
        # input whose lengths differ across batches (the standard flow only
        # escaped because TextDataset pre-pads to 128).
        global_max_len = max((len(x) for x in inputs), default=0)
        # Clamp to the model's own context window (RIL ISS-130). The dataset
        # is pre-truncated to ``self.max_seq_len``, but a caller that tuned
        # ``LMTask(max_seq_len=128)`` while handing it a smaller model would
        # still forward 128-token rows into a smaller positional-encoding
        # table and crash with "Sequence endpoint N exceeds maximum sequence
        # length". Never let the batch exceed what the model can attend to.
        model_capacity = getattr(model, "max_seq_len", None)
        if model_capacity is not None:
            global_max_len = min(global_max_len, int(model_capacity))
            # Truncate sequences longer than the model can attend to (rather
            # than letting them overrun the table): the tail beyond capacity
            # simply cannot be scored by this model.
            inputs = [x[:global_max_len] for x in inputs]
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0

        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            lengths = [len(x) for x in batch]
            max_len = global_max_len
            padded = torch.stack(
                [
                    torch.cat(
                        [
                            torch.as_tensor(x, dtype=torch.long),
                            torch.full((max_len - len(x),), pad_id, dtype=torch.long),
                        ]
                    )
                    for x in batch
                ]
            )

            # Padding mask (True = mask out, the ``sdpa`` wrapper's
            # convention). Only built when the tokenizer has a dedicated
            # pad id and the batch actually contains padding; a literal
            # pad token in the text is indistinguishable, so masking is
            # disabled for tokenizers without a pad id.
            attn_mask = None
            if self.pad_token_id is not None and any(length < max_len for length in lengths):
                attn_mask = (padded == self.pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]

            with torch.no_grad():
                logits = model(padded, attn_mask=attn_mask)
            results.append(logits)

        if not results:
            # Empty eval set (e.g. an empty corpus file): return an
            # empty 3-D prediction instead of crashing on
            # ``torch.cat([])``. The metric layer already reports ``inf``
            # for a zero-size batch (perplexity is undefined), so an
            # appropriately-shaped empty tensor keeps the whole
            # evaluation pipeline (run() + evaluate()) alive.
            return torch.empty(0, 0, 0)
        return torch.cat(results, dim=0)
