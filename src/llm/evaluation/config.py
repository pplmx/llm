from pydantic import BaseModel


class EvalConfig(BaseModel):
    eval_interval: int = 1000
    eval_batch_size: int = 8
    metrics: list[str] = ["perplexity", "accuracy"]
