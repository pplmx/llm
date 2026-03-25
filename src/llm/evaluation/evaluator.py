import torch

from llm.evaluation.tasks.base import BaseTask


class Evaluator:
    def __init__(self, task: BaseTask):
        self.task = task

    def evaluate(self, model) -> dict:
        inputs, references = self.task.prepare_data("val")

        input_ids = [torch.as_tensor(x, dtype=torch.long) for x in inputs]

        predictions = self.task.predict(model, input_ids)

        refs = torch.stack([torch.as_tensor(x, dtype=torch.long) for x in references])

        results = {}
        for metric in self.task.metrics:
            result = metric.compute(predictions, refs)
            results.update(result)

        return results
