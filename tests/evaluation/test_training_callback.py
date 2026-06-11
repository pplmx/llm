from llm.evaluation.runner import EvaluationRunner
from llm.training.core.callbacks import EvaluationCallback


def test_evaluation_callback_stores_eval_interval():
    callback = EvaluationCallback(eval_runner=EvaluationRunner, eval_interval=100)
    assert callback.eval_interval == 100
