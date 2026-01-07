# LLM 框架教程: 从零开始构建与训练

本教程旨在指导您如何使用本项目自定义的 LLM 框架进行模型构建、训练和实验. 我们将专注于框架的核心组件和工作流程, 帮助您理解如何利用其模块化和可扩展性.

## 目录

- [一、框架核心概念](#一、框架核心概念)
- [二、环境搭建](#二、环境搭建)
- [三、模型构建与组件](#三、模型构建与组件)
- [四、数据处理](#四、数据处理)
- [五、训练流程与配置](#五、训练流程与配置)
- [六、扩展与高级特性](#六、扩展与高级特性)
- [七、常见问题与故障排查](#七、常见问题与故障排查)

---

### 一、框架核心概念

本项目提供了一个模块化、可扩展的 PyTorch LLM 训练框架. 其核心设计理念是**解耦**和**可扩展性**, 旨在将通用的训练逻辑与具体的任务逻辑分离.

**主要特点:**

- **模块化架构**: Transformer 核心组件(如注意力、MLP、归一化)被设计为可插拔模块.
- **现代化特性**: 支持 Grouped Query Attention (GQA)、SwiGLU 激活函数、RMSNorm 等先进技术.
- **灵活配置**: 通过 YAML 文件和 Python `dataclasses` 实现高度可配置的模型和训练参数.
- **健壮的训练引擎**: 支持分布式训练 (DDP)、自动混合精度 (AMP)、`torch.compile` 优化、回调系统和检查点管理.
- **抽象的数据流**: 通过 `DataModule` 抽象数据加载和预处理.
- **可扩展的任务**: 通过 `TrainingTask` 抽象不同的训练任务.

---

### 二、环境搭建

本项目使用 `uv` 进行依赖管理, 并推荐使用 `Makefile` 进行环境设置和常用任务管理.

1. **安装 `uv`**: 如果您尚未安装 `uv`, 请按照官方说明进行安装: [uv 安装指南](https://github.com/astral-sh/uv#installation).
2. **克隆仓库**:

    ```bash
    git clone https://github.com/pplmx/llm.git
    cd llm
    ```

3. **初始化项目**: 运行 `make init` 命令. 这将创建虚拟环境、安装所有必要的依赖项, 并设置 pre-commit 钩子.

    ```bash
    make init
    ```

4. **同步依赖 (如果需要)**: 如果 `pyproject.toml` 或 `uv.lock` 发生变化, 您可以重新同步依赖:

    ```bash
    make sync
    ```

---

### 三、模型构建与组件

本项目框架的核心模型是 `DecoderModel`, 它由一系列模块化组件构成. 这些组件位于 `src/llm/core/` 和 `src/llm/models/` 目录下.

**核心组件概览:**

- **分词器**:
    - **`llm.tokenization.simple_tokenizer.SimpleCharacterTokenizer`** (`src/llm/tokenization/simple_tokenizer.py`): 一个基础的字符级分词器, 用于快速原型验证. 支持 `<PAD>` 特殊 token.
    - **`llm.tokenization.bpe_tokenizer.BPETokenizer`** (`src/llm/tokenization/bpe_tokenizer.py`): 生产级的 BPE (Byte Pair Encoding) 分词器, 基于 `tokenizers` 库实现, 支持训练自定义词表和高效的子词分词.

- **`llm.core.embedding.EmbeddingLayer`**:
    - **位置**: `src/llm/core/embedding.py`
    - **用途**: 结合 token 嵌入和位置编码, 为模型提供输入序列的向量表示.

- **`llm.core.attn.mha.MultiHeadAttention`**:
    - **位置**: `src/llm/core/attn/mha.py`
    - **用途**: 实现多头自注意力机制, 是 Transformer 的核心. 支持 **Grouped Query Attention (GQA)** 以平衡性能和显存效率.
    - **GQA 说明**: 通过 `num_kv_heads` 参数控制 K/V 头数, 当 `num_kv_heads < num_heads` 时启用 GQA, 多个 Q 头共享同一组 K/V 头, 显著减少 KV Cache 的显存占用.

- **`llm.core.mlp.MLP`**:
    - **位置**: `src/llm/core/mlp.py`
    - **用途**: Transformer 层中的多层感知器(前馈网络). 支持 **SwiGLU 激活函数**以提升性能.
    - **SwiGLU 说明**: 通过设置 `use_glu=True` 启用, 结合 Swish 激活和门控线性单元, 相比标准 GELU 激活能提供更好的性能.

- **`llm.core.transformer_block.TransformerBlock`**:
    - **位置**: `src/llm/core/transformer_block.py`
    - **用途**: 构成 Transformer 模型的基本单元, 结合了注意力机制和 MLP, 并处理层归一化和残差连接.

- **`llm.models.decoder.DecoderModel`**:
    - **位置**: `src/llm/models/decoder.py`
    - **用途**: 完整的解码器模型, 堆叠了多个 `TransformerBlock`, 并包含一个语言模型头用于预测下一个 token.

**模型配置:**

您可以通过 `Config` 类(特别是 `ModelConfig` 部分)来配置 `DecoderModel` 的参数. 关键配置项包括：

- `hidden_size`: 模型维度
- `num_layers`: Transformer 层数
- `num_heads`: 注意力头数
- `num_kv_heads`: K/V 头数(用于 GQA, 设为 `None` 使用标准 MHA)
- `use_glu`: 是否启用 SwiGLU(默认 `False`)
- `norm_type`: 归一化类型(`nn.LayerNorm` 或 `RMSNorm`)
- `use_moe`: 是否使用 Mixture of Experts

---

### 四、数据处理

本项目通过 `DataModule` 抽象数据处理和加载.

- **`llm.data.data_module.BaseDataModule`**:
    - **位置**: `src/llm/data/data_module.py`
    - **用途**: 定义了数据模块的抽象接口, 包括数据准备 (`prepare_data`)、设置 (`setup`) 和创建数据加载器 (`train_dataloader`, `val_dataloader`).

- **`llm.data.synthetic_data_module.SyntheticDataModule`**:
    - **位置**: `src/llm/data/synthetic_data_module.py`
    - **用途**: `BaseDataModule` 的一个实现, 用于生成合成数据进行训练和测试. 这对于框架的初步验证和功能开发非常有用.

**如何使用:**

在您的 `TrainingTask` 中, 您将实例化一个 `DataModule` 的子类, 并将其传递给 `TrainingEngine`.

---

### 五、训练流程与配置

项目的训练流程由 `TrainingEngine` 驱动, 并通过 `Config` 类进行全面配置.

**训练入口:**

主要的训练脚本是 `src/llm/training/train.py`. 您可以通过命令行参数来选择训练任务和覆盖默认配置.

**示例命令:**

```bash
llm-train --task regression --epochs 5 --batch-size 64
```

**配置管理:**

- **`llm.training.core.config.Config`**:
    - **位置**: `src/llm/training/core/config.py`
    - **用途**: 集中管理所有训练相关的配置, 包括模型、训练参数、分布式设置、优化选项、检查点和日志.
    - 支持从 YAML 文件加载配置, 并通过命令行参数和环境变量进行覆盖.

**训练引擎 (`TrainingEngine`):**

- **位置**: `src/llm/training/core/engine.py`
- **用途**: 协调整个训练循环, 包括:
    - 模型、优化器、调度器和损失函数的构建.
    - 分布式训练 (DDP) 的设置.
    - 自动混合精度 (AMP) 和 `torch.compile` 的集成.
    - 检查点加载和保存.
    - 通过回调系统触发自定义逻辑.

**训练任务 (`TrainingTask`):**

- **位置**: `src/llm/training/tasks/base_task.py`
- **用途**: 抽象了具体的训练任务. 您需要实现 `TrainingTask` 的子类来定义模型的构建、优化器、损失函数以及训练和验证步骤.

---

### 六、扩展与高级特性

本项目框架旨在高度可扩展, 允许您轻松添加新功能.

- **添加新的训练任务**:
    - 创建 `llm.training.tasks.base_task.TrainingTask` 的子类, 实现其抽象方法.
    - 在 `src/llm/training/train.py` 的 `AVAILABLE_TASKS` 字典中注册您的新任务.

- **自定义回调**:
    - 创建 `llm.training.core.callbacks.Callback` 的子类, 并在训练生命周期的不同阶段实现 `on_...` 方法.
    - 在 `train.py` 中将您的回调添加到 `TrainingEngine` 的回调列表中.

- **集成新的模型组件**:
    - 在 `src/llm/core/` 或 `src/llm/models/` 中添加新的注意力机制、归一化层或 MoE 实现.
    - 修改 `DecoderModel` 或其他模型定义以集成这些新组件.

- **性能优化**:
    - 利用 `Config` 中的 `optimization` 部分来启用或禁用 `torch.compile` 和 AMP.
    - 探索更高级的优化技术, 如梯度累积、模型并行等.

---

### 七、常见问题与故障排查

如果您在使用过程中遇到问题, 请参考项目的 [故障排查指南](docs/troubleshooting.md).
