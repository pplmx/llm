# `src/llm/training` 模块文档

## 概述

`src/llm/training` 模块提供了一个模块化、可扩展的训练框架，用于训练 PyTorch 模型。该框架支持分布式数据并行（DDP）、混合精度训练、回调机制和灵活的配置管理。

## 核心组件

### 1. `train.py`

这是训练过程的入口点。其主要职责是:

- **参数解析**: 使用 `argparse` 解析命令行参数，特别是 `--task` 参数，用于选择要执行的训练任务。
- **配置加载**: 从命令行参数和环境变量中加载配置。

- **分布式训练管理**: 初始化 `DistributedManager`，用于处理 DDP 的设置和清理。
- **任务分派**: 根据 `--task` 参数，从 `AVAILABLE_TASKS` 字典中选择并实例化相应的任务类。
- **数据模块实例化**: 创建数据模块（例如，`SyntheticDataModule`）来处理数据的准备和加载。
- **回调实例化**: 创建一系列回调，如 `MetricsLogger`、`TensorBoardLogger` 和 `LRSchedulerCallback`。
- **`train_worker` 函数**: 这是每个分布式数据并行 (DDP) 进程的实际入口点。它负责在每个进程中实例化数据模块、任务、回调和训练引擎，并启动训练循环。
- **启动训练**:
  - 如果 `world_size` > 1，则使用 `torch.multiprocessing.spawn` 启动多个 DDP 进程，每个进程执行 `train_worker` 函数。
  - 如果 `world_size` == 1，则在单个进程中直接调用 `train_worker` 函数。

### 2. `core/engine.py`

`TrainingEngine` 类是训练循环的核心。其主要功能包括:

- **初始化**: 设置设备（CPU 或 GPU）、日志记录器、性能监视器、检查点管理器和回调。
- **组件设置**:
  - 从 `TrainingTask` 实例中构建模型、优化器、学习率调度器和损失函数。**模型构建时，`TransformerBlock` 会根据配置（`use_moe`、`num_experts`、`top_k`）来决定使用标准 MLP 还是 MoE 层。**
  - （可选）编译模型 (`torch.compile`)。
  - （可选）用 `DDP` 包装模型。
  - 从数据模块中获取数据加载器。
  - 从检查点（如果存在）恢复训练状态。
- **训练循环**:
  - 迭代指定的 `epochs`。
  - 在每个 `epoch` 中，执行 `_run_epoch` 进行训练，并可选择执行 `_run_validation_epoch` 进行验证。
  - 在 `_run_epoch` 中，迭代数据加载器中的批次，执行训练步骤（前向传播、反向传播、优化器步骤），并记录批次级别的统计信息。
  - **自动混合精度 (AMP)**: 使用 `torch.amp.GradScaler` 进行自动混合精度（AMP）训练，这有助于减少显存占用并加速训练。
- **回调管理**: 在训练过程的不同阶段（例如，`on_epoch_start`、`on_batch_end`）调用注册的回调函数。
- **日志记录**: 记录训练和验证过程中的损失、学习率、内存使用情况等详细信息。
- **检查点**: 在每个 `epoch` 结束时，根据配置保存模型、优化器、调度器和 `scaler` 的状态。

### 3. `core/config.py`

该文件使用 `dataclasses` 定义了用于配置训练过程的各个方面:

- `ModelConfig`: 模型相关的超参数，包括隐藏层大小、层数，以及 **MoE (Mixture of Experts) 相关的参数（`use_moe`、`num_experts`、`top_k`）**。
- `TrainingConfig`: 训练过程的超参数（例如，学习率、批量大小、`epochs`）。
- `DistributedConfig`: 分布式训练的配置（例如，`master_addr`、`master_port`）。
- `OptimizationConfig`: 性能优化的配置（例如，是否使用 `torch.compile` 和 AMP）。
- `CheckpointConfig`: 检查点相关的配置（例如，目录、恢复路径）。
- `LoggingConfig`: 日志相关的配置（例如，日志级别、目录）。
- `Config`: 主配置类，将所有其他配置组合在一起，并提供从 YAML 文件或命令行参数加载配置的功能。

### 4. `core/callbacks.py`

该文件定义了回调机制，允许在训练过程的关键点插入自定义逻辑:

- `Callback`: 所有回调的基类，定义了 `on_train_start`、`on_epoch_end` 等接口。
- `MetricsLogger`: 在每个 `epoch` 结束时记录指标。
- `EarlyStopping`: 如果监控的指标在一段时间内没有改善，则提前停止训练。
- `TensorBoardLogger`: 将指标记录到 TensorBoard 以进行可视化。
- `LRSchedulerCallback`: 记录学习率的变化。

### 5. `core/utils.py`

该文件提供了一系列辅助类，以支持训练框架:

- `PerformanceMonitor`: 监控性能指标，如批处理时间、损失和 GPU 内存使用情况。
- `Logger`: 一个增强的日志管理器，支持分布式环境下的日志记录，并可将日志保存到文件。
- `DistributedManager`: 封装了 `torch.distributed` 的功能，简化了分布式训练的设置、清理和通信（例如，`reduce_mean`）。
- `CheckpointManager`: 处理检查点的保存和加载，支持保存最新的、周期性的和最佳的检查点，并能自动清理旧的检查点。

### 6. `tasks/base_task.py`

`TrainingTask` 是一个抽象基类，定义了所有具体训练任务必须实现的接口。这确保了不同任务之间的一致性，并使得 `TrainingEngine` 可以以通用的方式处理它们。

必须实现的抽象方法包括:

- `build_model`: 构建并返回模型实例。
- `build_optimizer`: 构建并返回优化器实例。
- `build_scheduler`: 构建并返回学习率调度器实例。
- `build_criterion`: 构建并返回损失函数实例。
- `train_step`: 定义单个训练批次的前向传播、损失计算和反向传播逻辑。
- `validation_step`: 定义单个验证批次的前向传播和损失计算逻辑。

### 7. `tasks/regression_task.py`

`RegressionTask` 是 `TrainingTask` 的一个具体实现，用于解决一个简单的回归问题。它展示了如何实现 `TrainingTask` 的所有抽象方法，以定义一个完整的训练任务。

## 实现逻辑

该训练框架的实现逻辑遵循“控制反转”（Inversion of Control）的原则。`TrainingEngine` 作为框架的核心，负责管理整个训练流程，而具体的任务逻辑（如模型结构、优化器选择、损失计算）则由用户通过实现 `TrainingTask` 的子类来提供。

1.  **启动**: 用户通过命令行运行 `train.py`，并使用 `--task` 参数指定要运行的任务。
2.  **配置**: `Config` 类从命令行参数、环境变量或 YAML 文件中加载配置。
3.  **初始化**: `train.py` 根据配置初始化 `DistributedManager`、数据模块、任务和回调。
4.  **引擎创建**: `TrainingEngine` 被创建，并接收所有必要的组件。
5.  **训练**: `TrainingEngine` 调用 `run` 方法启动训练循环。在循环中，它会依次调用 `train_step` 和 `validation_step`（由具体的 `TrainingTask` 实现），并触发相应的回调。
6.  **解耦**: 通过将通用逻辑（如分布式设置、检查点、日志记录）与特定于任务的逻辑分离开来，该框架实现了高度的模块化和可扩展性。用户可以轻松地添加新的任务、回调或配置，而无需修改框架的核心代码。
