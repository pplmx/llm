from llm.evaluation.metrics.base import BaseMetric
from llm.evaluation.tasks.base import BaseTask


class MockMetric(BaseMetric):
    name = "mock"

    def compute(self, predictions, references):
        return {"score": len(predictions)}


class MockTask(BaseTask):
    name = "mock"
    metrics = []

    def prepare_data(self, split: str):
        return (["input1", "input2"], ["ref1", "ref2"])

    def predict(self, model, inputs):
        return ["pred1", "pred2"]


def test_metric_compute():
    metric = MockMetric()
    result = metric.compute(["a", "b"], ["a", "b"])
    assert result == {"score": 2}


def test_task_prepare_data():
    task = MockTask()
    inputs, refs = task.prepare_data("val")
    assert inputs == ["input1", "input2"]
    assert refs == ["ref1", "ref2"]


def test_task_predict():
    task = MockTask()
    preds = task.predict(None, ["hello", "world"])
    assert preds == ["pred1", "pred2"]
