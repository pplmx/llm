# 故障排查指南

本文档提供了在使用本项目时可能遇到的常见问题及其解决方案。
如果您在这里找不到解决方案，请查阅相关文档或在 [GitHub Issues](https://github.com/pplmx/llm/issues) 提交问题。

## 目录

- [安装与环境](#安装与环境)
- [训练问题](#训练问题)
- [分布式训练 (DDP)](#分布式训练-ddp)
- [性能优化](#性能优化)
- [检查点问题](#检查点问题)
- [MoE (Mixture of Experts)](#moe-mixture-of-experts)
- LoRA / QLoRA 相关问题
- [KVCache 相关问题](#kvcache-相关问题)
- [命令行参数](#命令行参数)
- [Serving 推理服务](#serving-推理服务)
- [PEFT 相关问题](#peft-相关问题)
- [评估相关问题](#评估相关问题)
- [导出相关问题](#导出相关问题)

---

## 安装与环境

- **问题: `make init` 或 `make sync` 失败，或遇到依赖冲突.**
    - **解决方案:**
    1. 确保您的 Python 版本符合 `pyproject.toml` 中 `requires-python` 的要求(当前为 3.14+)。
    2. 尝试清理 `uv` 缓存: `uv clean`。
    3. 检查 `pyproject.toml` 和 `uv.lock` 文件，手动解决潜在的依赖冲突。
    4. 确保您的网络连接正常，可以访问 PyPI。

- **问题: `make` 命令无法执行，提示"command not found".*
    - **解决方案:** 确保您的系统安装了 `make` 工具。
        - **Linux/macOS:** 通常预装。
        - **Windows:** 可以通过 Chocolatey (`choco install make`) 或 Scoop (`scoop install make`) 安装，或者安装 Git for Windows (它通常包含 `make`)。

- **问题: 训练时遇到 `torch.cuda.is_available()` 返回 `False`，即使有 GPU.*
    - **解决方案:**
    1. 确保您安装了正确版本的 PyTorch，并且它与您的 CUDA 驱动版本兼容。
    2. 检查您的 CUDA 驱动是否已正确安装并更新到最新版本。
    3. 确认您的 GPU 设备已正确识别并启用。
    4. 如果使用 Docker，确保容器以 `--gpus all` 或类似方式运行。

---

## 训练问题

- **问题: 训练过程中出现内存不足 (OOM) 错误.*
    - **解决方案:**
    1. **减小 `batch_size`**: 这是最直接有效的方法。
    2. **减小模型大小**: 尝试减小 `hidden_size` 或 `num_layers`。
    3. **启用自动混合精度 (AMP)**: 在 `config.py` 中设置 `optimization.use_amp = True`，或在命令行中不使用 `--no-optimization-use-amp`。AMP 可以显著减少显存占用。
    4. **启用 `torch.compile`**: 在 `config.py` 中设置 `optimization.use_compile = True`，或在命令行中不使用 `--no-optimization-use-compile`。
    5. **梯度累积**: 如果您的任务支持，可以通过增大 `gradient_accumulation_steps` 来模拟更大的批次大小，同时保持较小的实际 `batch_size`。

- **问题: 分词器抛出 `KeyError`，提示字符不在词汇表中.*
    - **解决方案:** 当前的 `SimpleCharacterTokenizer` 是字符级别的，并且词汇表是根据初始化时提供的语料库构建的。确保您尝试编码的文本只包含在初始化分词器时语料库中存在的字符。如果需要处理更广泛的字符集，您可能需要更新分词器或其初始化语料。

---

## 分布式训练 (DDP)

- **问题: `DDP Misconfiguration: world_size is X, but ... insufficient GPUs`**
    - **解决方案:**
    - 检查您的 `DistributedConfig` 或环境变量 `GPUS_PER_NODE` 是否设置正确。
    - 运行 `nvidia-smi` 确认您的机器上有多少可用的 GPU。
    - `world_size` 应该等于 `num_nodes * gpus_per_node`。

- **问题: 训练进程卡死 (Hang)**
    - **解决方案:**
    1. **检查日志**: 查看每个 `rank` 的日志文件，找出是否有某个进程在其他进程卡住之前就抛出了错误。
    2. **网络问题**: 在多节点环境中，确保节点之间的网络连接是畅通的，特别是 `MASTER_ADDR` 和 `MASTER_PORT` 指定的端口没有被防火墙阻塞。
    3. **代码分支**: 确保在所有 DDP 进程中，参与分布式操作的代码路径是一致的。

---

## 性能优化

- **问题: GPU 利用率低 / 性能不理想**
    - **解决方案:**
    1. **增加数据加载 workers**: 提高 `optimization.num_workers` 值，使用更多进程并行加载数据。
    2. **启用内存钉选**: 确保 `optimization.pin_memory` 设置为 `true`，加速 CPU 到 GPU 数据传输。
    3. **启用 torch.compile**: 确保 `optimization.use_compile` 设置为 `true`。
    4. **减少 CPU-GPU 同步**: 减少不必要的同步操作，如频繁调用 `.item()`。

---

## 检查点问题

- **问题: `Error(s) in loading state_dict for ...: Missing key(s) in state_dict: ...`**
    - **解决方案:**
    - 确保您在恢复训练时使用的模型配置 (`ModelConfig`) 与保存该检查点时的配置完全相同。
    - 如果想加载结构不同的模型，可以手动编写代码加载匹配的权重部分。

- **问题: 恢复训练后，效果与之前不符**
    - **解决方案:** 确保 `CheckpointManager` 正确保存和加载了所有相关状态，包括优化器、学习率调度器和随机数生成器状态。

---

## MoE (Mixture of Experts)

- **问题: MoE 训练收敛困难或性能不佳**
    - **解决方案:**
    1. **调整 `top_k`**: 尝试不同的 `top_k` 值。
    2. **负载均衡损失**: 在损失函数中添加负载均衡损失项，鼓励所有专家被均匀利用。
    3. **门控网络初始化**: 确保门控网络的初始化有助于有效的专家路由。

- **问题: MoE 训练时出现 `NaN` 或 `Inf` 值**
    - **解决方案:**
    1. **检查门控网络输出**: 确保门控网络的 logits 不会过大或过小。
    2. **调整学习率**: 尝试减小学习率。
    3. **梯度裁剪**: 确保梯度裁剪 (`training.gradient_clip_val`) 已启用并设置合理。
    4. **检查专家 MLP**: 确保专家 MLP 内部的计算没有导致数值溢出。

---

## LoRA / QLoRA 相关问题

- **问题: 应用 LoRA 后模型输出与原始模型完全相同.**
    - **解决方案:** LoRA 的 B 矩阵初始化为零，因此初始输出应该相同。确保:
    1. 您已正确调用 `apply_lora(model, ...)`。
    2. 训练时只优化 LoRA 参数: `optimizer = AdamW(get_lora_parameters(model), lr=1e-4)`。
    3. 确认 LoRA 参数的 `requires_grad=True`。

- **问题: QLoRA 推理速度比预期慢.**
    - **解决方案:** QLoRA 在每次 forward 时需要反量化权重，这会增加开销:
    1. 对于推理，考虑使用标准 LoRA 并在训练后 `merge_lora(model)`。
    2. QLoRA 主要优势是训练时的内存节省，而非推理速度。

---

## KVCache 相关问题

- **问题: 使用 KVCache 时生成的文本与不使用时不同.**
    - **解决方案:** 确保:
    1. 在新序列开始前调用 `cache.reset()` 重置缓存。
    2. `max_seq_len` 足够大以容纳完整生成。
    3. 首次 forward 传入完整 prompt，后续 forward 只传入新生成的 token。

- **问题: KVCache 超出预分配长度导致错误.**
    - **解决方案:** 增大 `max_seq_len` 或在生成循环中检查 `cache.seq_len < cache.max_seq_len`。

---

## 命令行参数

- **问题: 运行 `train.py` 时出现 `unrecognized arguments` 错误**
    - **解决方案:**
    1. **检查参数名称**: 确保使用 `--<配置组名称>-<参数名称>` 格式(例如, `--model-hidden-size`, `--training-epochs`)。
    2. **布尔参数格式**: 布尔参数不应带值。默认为 `False` 时使用 `--<参数名>` 启用；默认为 `True` 时使用 `--no-<参数名>` 禁用。
    3. **查看帮助**: 运行 `llm-train --help` 查看所有可用参数及其格式。

---

## Serving 推理服务

**问题: llm-serve 启动失败，报 "Refusing to start: ... api_key is not set"**

- **解决方案:** 这是公开主机守卫机制。当绑定 0.0.0.0 时必须设置 API key。本地开发使用 `LLM_SERVING_HOST=127.0.0.1` 或设置 `LLM_SERVING_API_KEY`。

**问题: curl 调用 /v1/chat/completions 返回 403**

- **解决方案:** 服务需要 API key。确保请求头包含 `X-API-Key` 或 `Authorization: Bearer`。本地开发不需要 key（使用 127.0.0.1）。

**问题: 服务返回空输出或乱码**

- **解决方案:** 如果 `model_path=None`（dummy 模型），输出是随机 token 解码。需要训练 checkpoint 后配置 `model_path` 和 `tokenizer_path`。

---

## PEFT 相关问题

**问题: peft_kwargs 配置错误导致训练失败**

- **解决方案:** 不同 peft_method 需要不同的 kwargs。检查 peft_kwargs 是否与方法的预期参数匹配。常见错误：Adapter 需要 `bottleneck_dim`，Prefix Tuning 需要 `prefix_length`，LoRA 需要 `rank` 和 `alpha`。

**问题: PEFT adapter 加载到 serving 时提示 method_name mismatch**

- **解决方案:** 训练时的 `peft_method` 必须与 serving 时的 `LLM_SERVING_PEFT_METHOD` 一致。adapter sidecar 文件中的 `format_version` 和 `method_name` 元数据会在 load 时校验。

---

## 评估相关问题

**问题: import lm_eval 失败**

- **解决方案:** 需要安装可选依赖：`uv sync --extra eval` 或 `pip install 'llm[eval]'`。

**问题: MMLU 评估结果异常低**

- **解决方案:** 检查 few-shot 设置（默认 5-shot）和模型是否已训练。dummy 模型在 MMLU 上 ≈ 随机水平是正常的。

---

## 导出相关问题

**问题: ONNX 导出失败**

- **解决方案:** 确保安装了 onnx 和 onnxruntime：`uv sync --group test`。ONNX 导出要求模型处于 eval 模式且输入 shape 固定。
