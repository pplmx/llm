# 5. Use Pre-Allocated KV Cache

Date: 2026-01-08

## Status

Accepted

## Context

在自回归生成过程中，传统的 KV 缓存实现使用 `torch.cat` 在每个 token 生成时动态扩展缓存。这导致：

1. **内存碎片化**: 频繁的内存分配和释放
2. **O(n²) 内存操作**: 每次 cat 都需要复制已有数据
3. **性能下降**: 长序列生成时延迟显著增加

## Decision

实现 `KVCache` 类（`src/llm/core/kv_cache.py`），采用预分配策略：

1. **Pre-allocation**: 在生成开始时一次性分配 `max_seq_len` 大小的缓冲区
2. **In-place Update**: 使用索引赋值而非 cat 操作
3. **View Return**: 返回当前有效缓存的视图，无需复制

```python
# 传统方式 (O(n²))
k_cache = torch.cat([k_cache, new_k], dim=2)

# 新方式 (O(1))
self.k_cache[:, :, pos:pos+seq_len, :] = new_k
return self.k_cache[:, :, :pos+seq_len, :]
```

## Consequences

**优势**:

- 内存分配从 O(n) 次降为 1 次
- 无内存碎片化
- 生成延迟更稳定

**劣势**:

- 需要预先知道 `max_seq_len`
- 可能分配超过实际需要的内存

**兼容性**: 保留 `past_key_value` 元组格式作为向后兼容选项。
