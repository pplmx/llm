import torch

from llm.evaluation.evaluator import Evaluator
from llm.evaluation.tasks.lm_task import LMTask


def test_lm_task_prepare_data():
    task = LMTask(dataset_path="tests/dummies.py")
    inputs, refs = task.prepare_data("val")
    assert isinstance(inputs, list)
    assert isinstance(refs, list)


def test_evaluator_evaluate():
    task = LMTask(dataset_path="tests/dummies.py", batch_size=2)
    evaluator = Evaluator(task)

    class MockModel:
        def __call__(self, input_ids):
            batch, seq = input_ids.shape
            return torch.randn(batch, seq, 1000)

    results = evaluator.evaluate(MockModel())
    assert "perplexity" in results
