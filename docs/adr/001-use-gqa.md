# 001. Use Grouped Query Attention (GQA)

Date: 2024-12

## Status

Accepted

## Context

Multi-Head Attention (MHA) is the standard attention mechanism in Transformers, but it has significant memory overhead during inference due to the KV Cache. For each attention head, we need to cache both Key and Value tensors, which grows linearly with the number of heads and sequence length.

Key considerations:

- **Memory constraints**: KV Cache can consume 40-60% of GPU memory during inference
- **Inference speed**: Memory bandwidth becomes a bottleneck for large models
- **Model quality**: We need to maintain model performance while reducing memory
- **Industry trends**: Models like Llama 2, Mistral use GQA successfully

## Decision

We adopt **Grouped Query Attention (GQA)** as the default attention mechanism in our DecoderModel.

**Implementation details**:

- Add `num_kv_heads` parameter to `MultiHeadAttention`
- When `num_kv_heads < num_heads`, multiple Query heads share the same Key/Value heads
- Example: 32 Q heads with 8 KV heads means 4 Q heads share 1 KV head group
- Backward compatible: setting `num_kv_heads = num_heads` gives standard MHA

**Key features**:

```python
mha = MultiHeadAttention(
    hidden_size=2048,
    num_heads=32,        # 32 Query heads
    num_kv_heads=8,      # 8 Key/Value head groups (GQA)
)
```

## Consequences

### Positive

- **Memory savings**: 40-60% reduction in KV Cache size
- **Faster inference**: 20-30% speedup due to reduced memory bandwidth requirements
- **Longer sequences**: Can handle longer contexts with same memory budget
- **Proven approach**: Used successfully in state-of-the-art models
- **Minimal quality loss**: Performance degradation < 1% compared to MHA

### Negative

- **Slightly more complex**: Implementation is more complex than standard MHA
- **Hyperparameter tuning**: Need to tune the `num_kv_heads` parameter
- **Training cost**: Very slight increase in training time (< 5%)

### Neutral

- **Configuration flexibility**: Users can choose between MHA and GQA by setting `num_kv_heads`
- **Incremental adoption**: Can start with MHA and migrate to GQA later

## References

- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- Implementation: `src/llm/core/attn/mha.py`
