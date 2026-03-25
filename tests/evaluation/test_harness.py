from llm.evaluation.harness.adapter import LmEvalAdapter


def test_lm_eval_adapter_init():
    adapter = LmEvalAdapter()
    assert adapter is not None


def test_lm_eval_adapter_list_tasks():
    adapter = LmEvalAdapter()
    tasks = adapter.list_tasks()
    assert "mmlu" in tasks or "arc" in tasks
