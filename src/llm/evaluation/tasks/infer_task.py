from llm.evaluation.metrics.generation import BleuMetric, RougeMetric
from llm.evaluation.tasks.base import BaseTask


class InferTask(BaseTask):
    name = "inference"

    def __init__(self, dataset_path: str, batch_size: int = 8):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.metrics = [RougeMetric(), BleuMetric()]

    def prepare_data(self, split: str) -> tuple[list[str], list[str]]:
        inputs = ["hello world", "what is AI"]
        references = ["hello world", "artificial intelligence"]

        return inputs, references

    def predict(self, model, inputs: list[str]) -> list[str]:
        if callable(model) and not hasattr(model, "generate"):
            return model(inputs)

        return [model.generate(text) for text in inputs]
