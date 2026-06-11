import torch

from llm.data.datasets.text import TextDataset
from llm.evaluation.eval_tasks.base import BaseTask
from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.runtime.tokenizer_factory import TokenizerFactory


class LMTask(BaseTask):
    name = "lm"

    def __init__(self, dataset_path: str, batch_size: int = 8):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.metrics = [PerplexityMetric()]
        self.tokenizer = TokenizerFactory.from_dataset_text(dataset_path)

        self.val_dataset = TextDataset(
            file_path=dataset_path,
            tokenizer=self.tokenizer,
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
            max_len = max(len(x) for x in batch)
            padded = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch])

            with torch.no_grad():
                logits = model(padded)
            results.append(logits)

        return torch.cat(results, dim=0)
