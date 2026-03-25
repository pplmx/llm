# Evaluation Framework Design

**Date**: 2026-03-25  
**Status**: Draft  
**Owner**: LLM Project

## Overview

A unified evaluation framework supporting both training and inference stages, with benchmark integration for comprehensive model assessment.

## Goals

- Support training-time periodic evaluation (per-step/epoch)
- Support batch inference evaluation with benchmark tasks
- Integrate with `lm-eval` for standardized benchmarks
- Provide clean metrics API for extensibility

## Architecture

```
src/llm/evaluation/
├── __init__.py
├── metrics/              # Metric implementations
│   ├── __init__.py
│   ├── base.py           # BaseMetric abstract class
│   ├── perplexity.py     # PPL calculation
│   ├── accuracy.py       # Classification metrics
│   └── generation.py     # ROUGE, BLEU, chrF
├── tasks/                # Task definitions
│   ├── __init__.py
│   ├── base.py           # BaseTask abstract class
│   ├── lm_task.py        # Language modeling task
│   └── infer_task.py     # Inference generation task
├── harness/              # lm-eval integration
│   ├── __init__.py
│   └── adapter.py        # lm-eval wrapper
├── evaluator.py          # Unified evaluator
├── runner.py             # Periodic/batch runner
└── config.py             # Evaluation config
```

## Components

### 1. Metrics Layer

**BaseMetric** (base.py):
```python
class BaseMetric(ABC):
    name: str
    
    @abstractmethod
    def compute(self, predictions: Any, references: Any) -> dict:
        pass
```

**Implemented Metrics**:
- `PerplexityMetric`: Cross-entropy based, for training eval
- `AccuracyMetric`: Top-1 / Top-k accuracy
- `F1Metric`: Precision, recall, F1 for classification
- `RougeMetric`: ROUGE-1/2/L for generation
- `BleuMetric`: BLEU score
- `ChrFMetric`: Character n-gram F-score

### 2. Tasks Layer

**BaseTask** (base.py):
```python
class BaseTask(ABC):
    name: str
    metrics: list[BaseMetric]
    
    @abstractmethod
    def prepare_data(self, split: str) -> tuple[list, list]:
        pass
    
    @abstractmethod
    def predict(self, model, inputs: list) -> list:
        pass
```

**LMTask**: Training-time evaluation, uses validation set
**InferTask**: Inference evaluation, generates and computes metrics

### 3. Harness Layer

**LmEvalAdapter** (adapter.py):
- Wraps `lm-eval` library
- Provides standardized interface
- Supports: MMLU, ARC, BoolQ, HumanEval, MBPP

### 4. Evaluator

**Evaluator** (evaluator.py):
```python
class Evaluator:
    def __init__(self, task: BaseTask):
        self.task = task
    
    def evaluate(self, model) -> dict:
        # 1. Get predictions
        # 2. Compute all metrics
        # 3. Return aggregated results
```

### 5. Runner

**EvaluationRunner** (runner.py):

**Training Mode**:
- Integrates with `TrainingEngine` via callback
- Runs every N steps/epochs
- Logs to same metrics backend as training

**Inference Mode**:
- Batch evaluation on test set
- Supports scheduled runs (cron-like)
- Generates JSON/Markdown reports

## Integration Points

### Training Integration
```python
# In config
evaluation:
  eval_interval: 1000  # steps
  eval_batch_size: 8
  
# Evaluator registered as callback
engine = TrainingEngine(..., evaluators=[evaluator])
```

### Inference Integration
```python
# Batch evaluation
runner = InferenceEvaluationRunner(task=infer_task)
results = runner.run(model, dataset=test_data)
runner.save_report(results, format="markdown")
```

## Output Format

**Training Eval**:
```json
{
  "step": 1000,
  "metrics": {
    "perplexity": 12.5,
    "accuracy": 0.85
  }
}
```

**Benchmark Eval**:
```json
{
  "task": "mmlu",
  "metrics": {
    "acc": 0.72,
    "acc_norm": 0.70
  }
}
```

**Report** (Markdown):
```markdown
# Evaluation Report

## MMLU
| Subject | Accuracy |
|---------|----------|
| STEM    | 0.68     |
| Social  | 0.75     |

## Generation
| Metric | Score |
|--------|-------|
| ROUGE-L| 0.42  |
| BLEU   | 0.28  |
```

## Implementation Order

1. Base classes (BaseMetric, BaseTask)
2. Core metrics (Perplexity, Accuracy, F1)
3. LMTask + Evaluator (training eval)
4. Generation metrics (ROUGE, BLEU)
5. InferTask + Runner (inference eval)
6. lm-eval integration (benchmarks)
7. Report generation

## Dependencies

- `lm-eval` (for benchmarks)
- `rouge-score` (ROUGE)
- `sacrebleu` (BLEU, chrF)
- `scikit-learn` (F1, accuracy)

## Testing Strategy

- Unit tests for each metric
- Integration tests for task + evaluator
- E2E tests with dummy model