from lm_eval import evaluator
from lm_eval.tasks import TaskManager


class LmEvalAdapter:
    """Adapter for lm-evaluation-harness."""

    def __init__(self):
        self._task_manager = TaskManager()

    def list_tasks(self):
        """List available benchmark tasks."""
        return sorted(self._task_manager.all_tasks)

    def evaluate(self, model, tasks: list[str] = None, **kwargs):
        """Run evaluation on specified tasks."""
        results = evaluator.evaluate(model=model, tasks=tasks or ["mmlu"], **kwargs)

        return results

    def run_benchmark(self, model, benchmark: str, **kwargs) -> dict:
        """Run a specific benchmark."""
        return self.evaluate(model, tasks=[benchmark], **kwargs)
