from typing import cast

import torch

from llm.data.datasets.text import TextDataset
from llm.evaluation.eval_tasks.base import BaseTask
from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.tokenizer import BaseTokenizer


class LMTask(BaseTask):
    name = "lm"

    def __init__(self, dataset_path: str, batch_size: int = 8):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
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
            max_seq_len=128,
        )

    def prepare_data(self, split: str):
        inputs = []
        references = []

        for item in self.val_dataset:
            inputs.append(item["input_ids"])
            references.append(item["labels"])

        return inputs, references

    def predict(self, model, inputs: list):
        results = []

        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            lengths = [len(x) for x in batch]
            max_len = max(lengths)
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
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
