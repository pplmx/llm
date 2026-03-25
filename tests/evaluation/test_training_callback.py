import pytest
from llm.training.core.callbacks import Callback
from llm.evaluation.runner import EvaluationRunner


def test_evaluation_callback():
    from llm.training.core.callbacks import EvaluationCallback

    callback = EvaluationCallback(eval_runner=EvaluationRunner, eval_interval=100)

    assert callback is not None
