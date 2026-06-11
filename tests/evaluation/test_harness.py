from llm.evaluation.harness.adapter import LmEvalAdapter


def test_lm_eval_adapter_lists_known_benchmarks():
    adapter = LmEvalAdapter()
    tasks = adapter.list_tasks()
    assert "mmlu" in tasks or "arc" in tasks
