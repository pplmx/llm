# LLM 项目完整生命周期路线图

**最后更新**: 2026-01-06

这是一个动态更新的路线图, 用于追踪整个项目的进展. 项目采用迭代式开发, 每个阶段完成后都会有可用的功能增量.

## 项目现状总览

✅ **已完成的核心能力**:

- 现代化 Decoder-only Transformer 架构 (GQA, SwiGLU, MoE)
- 完善的分布式训练框架 (DDP, AMP)
- 高质量工程实践 (262 测试用例全部通过)
- 基础推理能力 (KV Cache, Top-k/Top-p 采样)

🚀 **当前重点**: 推理服务化 & 性能优化

---

## 阶段一: 基础建设 ✅

> **状态**: 已完成 | **完成时间**: 2024 年

- [x] **核心依赖与环境**: 使用 `uv` 管理 Python 依赖
- [x] **构建与打包**: 使用 `hatchling` (`pyproject.toml`)
- [x] **代码质量**: `ruff` (格式化), `mypy` (类型检查), `.editorconfig`
- [x] **测试体系**: `pytest` (单元/集成测试), `pytest-cov` (覆盖率)
- [x] **自动化 (CI/CD)**: GitHub Actions (`.github/workflows`) 自动执行测试、构建和发布流程
- [x] **环境管理**: `Dockerfile` 和 `compose.yml` 已完成

---

## 阶段二: 数据工程 ✅

> **状态**: 基础完成 | **完成时间**: 2024 年

- [x] **数据加载**: `src/llm/data/loader.py` 用于读取和初步处理数据
- [x] **数据抽象**: `DataModule` (`data_module.py`) 封装数据集和数据加载器
- [ ] **流式数据处理** ⏭️ *下一阶段*
    - [ ] 实现流式数据加载器, 支持大规模预训练
    - [ ] 集成常见预训练数据集 (C4, The Pile, RedPajama)
    - [ ] 添加数据质量过滤和去重工具
- [ ] **数据版本控制** ⏭️ *下一阶段*
    - [ ] 引入 `DVC` 或类似工具, 对数据集和预处理脚本进行版本管理

---

## 阶段三: 模型架构 ✅

> **状态**: 已完成 | **完成时间**: 2024-2025 年

- [x] **Transformer 基础**: `Embedding`, `PositionalEncoding`, `LayerNorm`, `RMSNorm`, `MLP` 等核心模块
- [x] **注意力机制**: `DotProductAttn`, `MHA` (多头), `MLA` (多查询)
- [x] **完整模型**: `DecoderModel` (`decoder.py`) 组装成一个完整的解码器架构
    - [x] `MoE` (专家混合) 模块已基础实现
    - [x] 实现并整合 `GQA` (分组查询注意力)
    - [x] 实现 `SwiGLU` 激活函数
    - [x] 优化 `MHA` 为统一 QKV 投影以提升计算效率

---

## 阶段四: 分词 ✅

> **状态**: 已完成 | **完成时间**: 2024 年

- [x] **原型分词器**: `SimpleCharacterTokenizer` 已实现, 用于快速验证
- [x] **生产级分词器**: 替换为基于 `tokenizers` 的 BPETokenizer
    - [x] 支持训练自定义 BPE 模型
    - [x] 集成到训练和推理流程中

---

## 阶段五: 模型训练 ✅

> **状态**: 已完成 | **完成时间**: 2024-2025 年

- [x] **训练框架**: `training/core/engine.py` 提供了灵活的训练和验证循环
- [x] **配置驱动**: `config.py` 支持通过配置来定义和调整实验
- [x] **任务抽象**: `BaseTask` 允许轻松定义新的训练目标
- [x] **大规模训练支持**:
    - [x] **分布式训练**: 实现并验证 `DDP` (数据并行) 基础流程
    - [x] **混合精度**: 实现自动 BF16/FP16 检测及 `torch.cuda.amp` 支持
- [ ] **训练策略** ⏭️ *阶段十一*
    - [ ] **预训练 (Pre-training)**: 设计并执行在大型通用语料库上的预训练流程
    - [ ] **指令微调 (Instruction Fine-tuning)**: 设计并执行在特定指令数据集上的微调流程

---

## 阶段六: 模型评估 🔄

> **状态**: 部分完成 | **预计完成**: 2025 Q2

- [x] **评估指标**:
    - [x] **语言模型指标**: 实现 `Perplexity` (困惑度) 计算
    - [ ] **下游任务指标**: 根据微调任务实现相应的准确率、F1 等
- [ ] **评估框架**: 建立一个标准化的评估流程, 可以在多个数据集和基准上自动运行评估
- [ ] **人工评估**: 设计人工评估流程, 用于评估模型的生成质量、相关性和无害性

---

## 阶段七: 推理与部署 🚀

> **状态**: 进行中 (当前重点) | **预计完成**: 2025 Q2

- [x] **推理 API**:
    - [x] **基础实现**: `inference.py` 已创建
    - [x] **核心功能**: 实现 `Greedy Search`, `KV Cache`, `Top-k/Top-p` 采样
- [x] **模型服务化**:
    - [x] 使用 `FastAPI` 将推理功能封装成一个 REST API 服务
    - [x] 支持流式输出 (Server-Sent Events)
    - [x] Prometheus 监控集成
    - [x] API Key 认证
    - [x] 实现批处理推理支持 (`/batch_generate` 端点)
    - [x] 添加请求队列和并发控制 (`asyncio.Semaphore` + timeout)
- [ ] **性能优化** (基础) ⏭️ *阶段十*
    - [x] 集成 `torch.compile` 到推理流程 (可选配置)
    - [ ] 优化 KV Cache 内存管理
    - [ ] 实现推理缓存机制

---

## 阶段八: 测试与质量保证 ✅

> **状态**: 持续进行 | **最后更新**: 2026-01-07

- [x] **单元测试**: 核心模块覆盖, 目前 379 个测试用例全部通过
- [x] **代码质量**: 全面应用 `ruff` 规范并修复所有 lint 问题
- [x] **集成测试**:
    - [x] 完善 `MoE` 动态专家路由的深度集成测试
- [x] **端到端 (E2E) 测试**: 创建一个脚本, 自动完成"训练-评估-推理"的全流程, 作为最终的冒烟测试

---

## 阶段九: 文档与社区 🔄

> **状态**: 持续完善 | **最后更新**: 2025-12-21

- [x] **基础文档**: `README`, `CONTRIBUTING` 等已存在
- [x] **训练框架文档**: 8 个详细的训练框架文档 (components, flow, guides 等)
- [ ] **开发者文档** ⏭️ *阶段十五*
    - [ ] 完善代码内文档 (docstrings)
    - [ ] 使用 `Sphinx` 或 `MkDocs` 生成 API 文档网站
- [ ] **用户文档**: 编写详细的用户指南和教程
- [x] **示例**: 提供 `Jupyter Notebook` 示例, 展示框架的核心功能 (`notebooks/quick_start.ipynb`)

---

## 阶段十: 性能优化与加速 ⏭️

> **优先级**: P1 (高) | **预计时间**: 2025 Q2-Q3 | **预计工作量**: 2-3 个月

### 目标

优化推理和训练性能, 降低延迟和显存占用, 提升吞吐量.

### 关键指标

- 推理延迟降低 50%+
- 显存占用降低 30%+
- 训练吞吐提升 40%+

### 任务清单

#### 10.1 Flash Attention 集成

> [!NOTE]
> 暂缓: 当前使用自定义 SDPA 实现, Flash Attention 集成优先级降低.

- [ ] 集成 Flash Attention 2
- [ ] 在 MHA 和 GQA 中启用 Flash Attention
- [ ] 性能基准测试和优化

#### 10.2 高级推理优化

- [ ] 实现 Paged Attention (vLLM style)
- [ ] 优化 KV Cache 内存布局和管理
- [ ] 实现 Continuous Batching
- [ ] 添加请求级别的动态调度

#### 10.3 编译优化

- [ ] 深度集成 `torch.compile` (推理和训练)
- [ ] 自定义 CUDA kernels (关键路径)
- [ ] 优化算子融合策略

#### 10.4 内存优化

- [x] 实现 Gradient Checkpointing
- [ ] 优化激活值重计算策略
- [ ] 减少峰值显存占用

---

## 阶段十一: 模型对齐与 RLHF ⏭️

> **优先级**: P2 (高) | **预计时间**: 2025 Q3-Q4 | **预计工作量**: 3-4 个月

### 目标

实现完整的模型对齐流程, 支持 RLHF、DPO 等对齐技术.

### 关键指标

- 完成 SFT → RLHF → DPO 完整流程
- 模型对齐质量达到可用标准
- 支持自定义偏好数据集

### 任务清单

#### 11.1 监督微调 (SFT)

- [ ] 实现完整的 SFT 数据处理流程
- [ ] 支持多种指令格式 (Alpaca, ShareGPT, etc.)
- [ ] 实现高效的 padding 和 masking 策略

#### 11.2 RLHF (Reinforcement Learning from Human Feedback)

- [ ] 实现 Reward Model 训练
- [ ] 实现 PPO (Proximal Policy Optimization) 训练器
- [ ] 添加 KL 散度约束和 Value Head
- [ ] 实现经验回放和优势估计

#### 11.3 DPO (Direct Preference Optimization)

- [ ] 实现 DPO 损失函数
- [ ] 支持偏好数据集处理
- [ ] 对比 DPO vs RLHF 性能

#### 11.4 其他对齐技术

- [ ] 研究并实现 RLAIF (AI Feedback)
- [ ] 探索 Constitutional AI
- [ ] 实现 Rejection Sampling

---

## 阶段十二: 多模态扩展 ⏭️

> **优先级**: P2 (中) | **预计时间**: 2025 Q4-2026 Q1 | **预计工作量**: 3-4 个月

### 目标

扩展模型能力至多模态, 支持图像-文本、音频-文本等多模态任务.

### 关键指标

- 支持至少 2 种模态 (视觉 + 文本)
- 多模态预训练/微调流程完整
- 性能达到 baseline 水平

### 任务清单

#### 12.1 视觉-语言模型

- [ ] 集成视觉编码器 (CLIP, SigLIP)
- [ ] 实现图像-文本对齐模块
- [ ] 支持多模态预训练
- [ ] 实现 Visual Instruction Tuning

#### 12.2 音频扩展

- [ ] 集成音频编码器 (Whisper-style)
- [ ] 实现语音识别和生成
- [ ] 支持语音指令微调

#### 12.3 统一多模态接口

- [ ] 设计通用的多模态数据接口
- [ ] 实现模态特定的预处理器
- [ ] 构建多模态 tokenizer

---

## 阶段十三: 量化与压缩 ⏭️

> **优先级**: P3 (中) | **预计时间**: 2025 Q4-2026 Q2 | **预计工作量**: 3-5 个月

### 目标

实现模型量化和压缩, 降低模型大小和推理成本, 支持边缘设备部署.

### 关键指标

- 模型大小减少 50%+
- 推理速度提升 40%+
- 精度损失 < 2%

### 任务清单

#### 13.1 Post-Training Quantization (PTQ)

- [ ] 实现 INT8 PTQ
- [ ] 实现 INT4 PTQ
- [ ] 支持混合精度量化

#### 13.2 Quantization-Aware Training (QAT)

- [ ] 实现 QAT 训练流程
- [ ] 支持 fake quantization
- [ ] 优化量化参数搜索

#### 13.3 高级量化技术

- [ ] 集成 GPTQ (GPT Quantization)
- [ ] 集成 AWQ (Activation-aware Weight Quantization)
- [ ] 集成 SmoothQuant
- [ ] 研究 GGML/GGUF 格式支持

#### 13.4 模型压缩

- [ ] 实现知识蒸馏
- [ ] 实现模型剪枝
- [ ] 探索低秩分解

---

## 阶段十四: 生态系统集成 ⏭️

> **优先级**: P3 (高) | **预计时间**: 2025 Q3-2026 Q2 | **预计工作量**: 4-6 个月

### 目标

与主流 LLM 生态系统集成, 提高项目可用性和互操作性.

### 关键指标

- 兼容 HuggingFace API
- 支持主流模型格式导出
- 可被 LangChain/LlamaIndex 直接使用

### 任务清单

#### 14.1 HuggingFace 集成

- [ ] 实现 `PreTrainedModel` 接口兼容
- [ ] 支持 `transformers` 库直接加载
- [ ] 实现 `safetensors` 格式支持
- [ ] 发布模型到 HuggingFace Hub

#### 14.2 模型导出

- [ ] 实现 ONNX 导出
- [ ] 实现 TorchScript 导出
- [ ] 支持 TensorRT 优化
- [ ] 探索 Core ML 支持 (iOS 部署)

#### 14.3 应用框架集成

- [ ] 集成 LangChain
- [ ] 集成 LlamaIndex
- [ ] 提供标准化 API 接口
- [x] 实现 OpenAI API 兼容层

#### 14.4 模型共享

- [ ] 建立模型仓库
- [ ] 发布预训练模型权重
- [ ] 提供模型卡片和文档
- [ ] 实现模型版本管理

---

## 阶段十五: 前沿技术探索 ⏭️

> **优先级**: P4 (持续进行) | **预计时间**: 持续 | **预计工作量**: 不定

### 目标

研究和实现前沿技术, 保持项目的技术领先性和创新性.

### 任务清单

#### 15.1 Long Context 支持

- [x] 实现 RoPE Scaling (NTK-aware, YaRN)
- [ ] 集成 ALiBi (Attention with Linear Biases)
- [ ] 探索 StreamingLLM
- [ ] 研究 Infinite Attention

#### 15.2 Sparse Attention

- [x] 实现 Sliding Window Attention
- [ ] 实现 Block Sparse Attention
- [ ] 研究 Longformer-style Attention
- [ ] 探索 BigBird Attention

#### 15.3 高效微调技术

- [x] 实现 LoRA (Low-Rank Adaptation)
- [ ] 实现 QLoRA (Quantized LoRA)
- [ ] 实现 AdaLoRA (Adaptive LoRA)
- [ ] 探索 Prefix Tuning / P-Tuning

#### 15.4 新型 MoE 架构

- [ ] 研究 Expert Choice Routing
- [ ] 实现 Soft MoE
- [ ] 探索 Dynamic Expert Selection
- [ ] 优化 Expert Load Balancing

#### 15.5 高级分布式训练

- [ ] 实现 FSDP (Fully Sharded Data Parallel)
- [ ] 实现 Pipeline Parallelism
- [ ] 集成 DeepSpeed ZeRO (Stage 2/3)
- [ ] 探索 3D Parallelism (DP + PP + TP)

#### 15.6 文档与社区

- [ ] 使用 Sphinx/MkDocs 生成 API 文档网站
- [ ] 创建互动式 Jupyter 教程
- [ ] 发布技术博客和论文解读
- [ ] 建立社区论坛 (Discord/GitHub Discussions)

---

## 实施原则

### 迭代式开发

- 每个阶段完成后都有可用的功能增量
- 快速原型, 快速验证, 快速迭代

### 用户价值优先

- 优先开发对用户价值最高的功能
- 根据社区反馈动态调整优先级

### 质量保证

- 所有新功能必须有对应的单元测试
- 保持测试覆盖率 > 80%
- 每个 PR 必须通过 CI/CD 检查

### 技术债务管理

- 定期进行代码重构和清理
- 及时更新依赖和安全补丁
- 保持代码库的可维护性

---

## 版本历史

- **v0.0.4** (2026-01-07): Gradient Checkpointing, E2E Pipeline, OpenAI Chat API, Batch Inference, 测试数 337
- **v0.0.3** (2026-01-05): 同步路线图与实际状态, 修复 train.py 任务注册, 更新测试计数
- **v0.0.2** (2025-12-21): 全面更新路线图, 添加 10-15 阶段的详细规划
- **v0.0.1** (2024): 初始版本, 完成 1-9 阶段的基础建设

---

> 💡 **提示**: 路线图会根据项目进展和社区反馈持续更新. 如有建议或问题, 请在 [GitHub Discussions](https://github.com/pplmx/llm/discussions) 中讨论.
