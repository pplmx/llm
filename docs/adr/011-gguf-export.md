# 011. GGUF Export Format Module

Date: 2026-08-04

## Status

Accepted

## Context

ROADMAP §13.3 的 "研究 GGML/GGUF 格式支持" 与 §14.2 的模型导出目标是同一件事的两面：

- **生态侧**：GGUF（GGML Unified Format）是 llama.cpp 生态的开放模型容器格式，
  可被 llama.cpp / llama-cpp-python / ollama 等直接消费；现有导出面只有 ONNX 与
  TorchScript（ADR-005）。
- **技术侧**：GGUF 是"重量优先"的推理格式——自带 F16/F32 与 GGML 块量化
  （Q4_0/Q8_0 等）存储语义，与已有的 GPTQ / AWQ / SmoothQuant（ADR-007/009/010）
  这类训练侧量化路径互补。

此前仓库没有任何 GGUF 相关代码，`EXPORT_REGISTRY` 也只有 `onnx`（代码内置）与
`torchscript`（entry point）两个目标。

## Decision

新增 `src/llm/export/gguf/` 包，分层实现 GGUF v3 容器与导出后端：

```text
export/gguf/
  spec.py       # 格式常量/枚举/header/tensor-info + 尺寸计算（无依赖）
  metadata.py   # 类型化 metadata KV 编解码（GGUFValueType，含 ARRAY）
  quant.py      # Q4_0 / Q8_0 块量化（numpy，镜像 ggml-quants.c 参考数学）
  reader.py     # GGUFReader：header/metadata/tensor info + 反量化读取
  writer.py     # GGUFWriter：组装 + 32 字节对齐 + 原子写入
  exporter.py   # export_to_gguf / build_gguf_exporter（EXPORT_REGISTRY 工厂）
```

### 格式语义（GGUF v3）

1. **布局**：header（magic `0x46554747` + uint32 版本 + uint64 tensor/kv 计数）→
   类型化 metadata → tensor info → 32 字节对齐的 tensor data（每个 tensor 的载荷
   之后补齐到对齐边界）。维度在磁盘上**逆序**存储（PyTorch `(A, B)` ↔ 文件
   `[B, A]`），reader/writer 在边界处翻译。
2. **v1 类型范围**：只实现 `F32` / `F16` / `Q4_0` / `Q8_0`（码值跨版本稳定），
   其余 GGML 类型（Q4_1、K-quants、IQ、整数类型）在 reader/writer 中显式报错。
3. **Q4_0 / Q8_0**：32 元素块；Q4_0 = fp16 scale + 16 字节 nibble（低半区低
   nibble、高半区高 nibble，隐式 offset 8）；Q8_0 = fp16 scale + 32 个 int8。
   量化公式镜像 ggml 参考实现（`d = amax / (2^(bits-1) - 1)`；Q4_0 用
   `(int8_t)(x + 8.5)` 截断并 clamp 到 15；Q8_0 用 round-half-away），保证字节级
   与 llama.cpp 兼容。
4. **格式层 torch-free**：spec/metadata/quant/reader/writer 只依赖 stdlib +
   numpy，exporter 层才接触 torch——格式层可独立测试与复用。
5. **错误处理**：坏 magic、版本越界、截断、越界 offset、未知类型码、异常 rank /
   string 长度统一抛 `GGUFError`（`ValueError` 子类），消息指明具体记录。

### 导出策略

`export_to_gguf(model, path, *, quantize=None, metadata=None, model_name=None)`：

- 默认全部 F16；`quantize="f32"` 全部 F32；`quantize="q4_0"/"q8_0"` 对
  ndim ≥ 2 且最后一维是 32 倍数的浮点张量做块量化，其余保持 F16；
- 元数据写入标准 `general.*` 键（`architecture` / `name` / `file_type` /
  `quantization_version`），用户元数据覆盖默认值；
- 非浮点张量（如 int buffer）显式 `NotImplementedError`（v1 范围）；
- 写入经临时文件 + rename 原子落盘，父目录自动创建。

### Registry 接线

`gguf = "llm.export.gguf.exporter:build_gguf_exporter"` 注册进
`pyproject.toml` 的 `llm.export_backends` entry-point group（与 `torchscript`
同模式），`ensure_exporters_registered()` 后 `export_model("gguf", ...)` 可用；
`llm.export` 顶层再导出 `export_to_gguf` / `build_gguf_exporter`。

## Alternatives

### Alternative A — 直接依赖第三方 `gguf` Python 包

- 优点：少写代码，与 llama.cpp 官方工具链同步
- 缺点：新增运行时依赖；官方包重写文件需内存、带全套架构命名约定，与
  "registry + 自研模块" 的仓库风格不符；格式细节无法按需裁剪（v1 只需
  F32/F16/Q4_0/Q8_0）
- 拒绝原因：自研 500 行左右即可覆盖 v1 需求，且格式层可独立验证

### Alternative B — 复用 safetensors 存量化权重

- 优点：复用 ADR-006 的既有存储路径
- 缺点：Q4_0/Q8_0 是 GGUF/GGML 的块布局语义，safetensors 只存原始字节，
  无法表达块结构；目标生态（llama.cpp）只认 GGUF 容器
- 拒绝原因：语义不匹配，且无法互操作

### Alternative C — 直接把 GGUF 做成新的 checkpoint 格式

- 优点：训练/推理统一
- 缺点：GGUF 无 optimizer/scheduler 状态、无 partial-load 扩展，训练语义
  由 ADR-006 三文件布局承担
- 拒绝原因：GGUF 定位是**导出/推理**格式，与 checkpoint 格式正交

## Consequences

- **互操作性**：导出的文件遵循 GGUF v3 字节布局，理论可被 llama.cpp 生态消费
  （架构 key 需要后续按目标模型补齐，v1 只保证容器与张量语义正确）
- **量化收益**：Q4_0 导出把 2D 权重压到 ~0.56 byte/elem，Q8_0 ~1.06 byte/elem，
  模型级测试验证文件体积显著小于 F16 且重建误差有上界
- **明确边界**：v1 不实现 K-quants / Q4_1 / Q5_x / IQ 系列、不做 mmap 读取、
  不写 llama.cpp 架构专用 tensor 命名（`blk.*`）与 tokenizer 元数据、不做
  非对称量化
- **零破坏性变更**：ONNX / TorchScript 路径不动，EXPORT_REGISTRY 新增一个
  entry-point 目标；`llm.export` 公共面只增不改

## References

- [GGUF file format spec（ggml-org/ggml `docs/gguf.md`）](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- ggml `ggml-quants.c`（Q4_0/Q8_0 参考量化/反量化实现）
- [ADR-005](./005-export-registry.md) — Export Registry Parity
- [ADR-007](./007-gptq-integration.md) / [ADR-009](./009-awq-integration.md) /
  [ADR-010](./010-smoothquant-integration.md) — 量化路径
- ROADMAP §13.3（GGUF 原为留待后续切片）与 §14.2（模型导出）
