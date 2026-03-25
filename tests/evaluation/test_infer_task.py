from llm.evaluation.tasks.infer_task import InferTask


def test_infer_task_prepare_data():
    task = InferTask(dataset_path="tests/dummies.py")
    inputs, refs = task.prepare_data("test")
    assert isinstance(inputs, list)
    assert len(inputs) > 0


def test_infer_task_predict():
    task = InferTask(dataset_path="tests/dummies.py", batch_size=2)

    def mock_generate(texts):
        return [f"generated: {t}" for t in texts]

    preds = task.predict(mock_generate, ["hello", "world"])
    assert preds == ["generated: hello", "generated: world"]
