# `llm.training` 模块

欢迎来到 `llm.training` 模块！这是一个为 PyTorch 设计的模块化、可扩展的训练框架.

## 设计理念

本框架的核心设计理念是**解耦**和**可扩展性**. 我们将通用的训练逻辑(如分布式训练、日志记录、检查点管理)与具体的任务逻辑(模型、优化器、损失函数)分离开来. 这使得开发者可以:

- **快速迭代**: 专注于实现 `TrainingTask`, 而无需关心底层的训练循环和分布式设置.
- **轻松扩展**: 通过自定义 `Callback` 来添加新功能, 或通过实现新的 `TrainingTask` 来支持新的模型和数据.

## 文档导航

为了帮助您更好地理解和使用本框架, 我们提供了以下详细文档, 覆盖了从入门到精通的各个方面:

### 理论与结构

- **[核心组件详解 (`components.md`)](./components.md)**
  - 深入了解构成训练框架的各个核心类, 例如 `TrainingEngine`、`Config`、`CheckpointManager` 等. 这里是理解“**什么**”的地方.

- **[端到端调用流程 (`training-flow.md`)](./training-flow.md)**
  - 从程序启动到结束, 宏观地了解各个组件是如何被依次调用和交互的. 这里是理解“**如何**”的地方.

- **[回调机制调用流程 (`flow-callbacks.md`)](./flow-callbacks.md)**
  - 详细剖析了回调(Callback)系统, 解释了在训练过程的各个阶段, 您的自定义逻辑可以在何时被触发. 这里是理解“**何时**”的地方.

### 实践与应用

- **[指南: 配置训练任务 (`guide-configuration.md`)](./guide-configuration.md)**
  - 学习如何通过命令行、YAML 文件和环境变量来配置您的训练任务.

- **[指南: 扩展训练框架 (`guide-extending.md`)](./guide-extending.md)**
  - 通过具体的“食谱”示例, 学习如何添加新的任务、回调、调度器等, 以扩展框架的功能.

- **[指南: 故障排查 (`troubleshooting.md`)](./troubleshooting.md)**
  - 快速诊断和解决训练过程中可能遇到的常见问题.

### 深入主题

- **[深度解析: 分布式数据并行 (DDP) (`deep-dive-ddp.md`)](./deep-dive-ddp.md)**
  - 为希望深入理解分布式训练背后原理的用户提供背景知识.

## 快速开始

1. **定义你的任务**: 创建一个 `TrainingTask` 的子类, 并实现其所有抽象方法.
2. **注册你的任务**: 在 `train.py` 的 `AVAILABLE_TASKS` 字典中添加你的新任务.
3. **运行训练**: 在命令行中执行以下命令:

    ```bash
    python -m llm.training.train --task <your_task_name>
    ```
