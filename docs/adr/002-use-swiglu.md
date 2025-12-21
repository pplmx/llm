# 002. Use SwiGLU Activation Function

Date: 2024-12

## Status

Accepted

## Context

The choice of activation function in feedforward networks significantly impacts model performance. Traditional Transformer models use GELU, but recent research has shown that gated activation functions can provide better performance.

Key considerations:
- **Model performance**: Need to maximize model quality
- **Computational cost**: Must remain practical for training and inference
- **Industry adoption**: Learn from successful large-scale models
- **Implementation complexity**: Balance benefits against complexity

Alternatives considered:
- **GELU**: Standard choice, simple and effective
- **ReLU**: Fast but potentially suboptimal
- **GLU variants**: Better performance but higher computational cost

## Decision

We adopt **SwiGLU (Swish-Gated Linear Unit)** as an optional activation function in our MLP layers.

**Implementation details**:
- Add `use_glu` parameter to MLP class
- When `use_glu=True`, use SwiGLU; otherwise use standard activation (GELU)
- SwiGLU formula: `SwiGLU(x, W, V) = Swish(xW) ⊗ xV`
- Requires 3x parameter count in intermediate layer but provides better performance

**Usage**:
```python
mlp = MLP(
    hidden_size=2048,
    intermediate_size=8192,
    use_glu=True,  # Enable SwiGLU
)
```

## Consequences

### Positive

- **Better performance**: 1-2% improvement in model quality compared to GELU
- **Used in SOTA models**: GLU variants used in PaLM, LLaMA, and other leading models
- **Smooth activations**: Swish provides smooth, non-monotonic activation
- **Empirically validated**: Strong performance across various benchmarks
- **Optional feature**: Can be disabled for simpler baseline experiments

### Negative

- **3x parameters**: Intermediate layer needs 3x parameters compared to standard FFN
- **Slower training**: Approximately 10-15% slower due to additional computation
- **More memory**: Higher memory footprint during training

### Neutral

- **Backward compatible**: Default is `use_glu=False` to maintain compatibility
- **Configuration flexibility**: Easy to A/B test against standard activations

## References

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Implementation: `src/llm/core/mlp.py`
