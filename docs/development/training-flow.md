# 训练框架调用流程

本文档旨在详细描述 `llm.training` 模块的动态执行流程, 展示从程序启动到训练完成的完整调用顺序.

## 端到端的高阶流程

整个训练过程可以分为以下几个主要阶段:

1. **启动入口 (`train.py`)**
    - 用户在命令行执行 `python -m llm.training.train --task <task_name> [other_args]`.
    - Python 解释器执行 `train.py` 中的 `main()` 函数.

2. **主进程初始化 (`main` 函数)**
    - **参数解析**: 解析 `--task` 等命令行参数.
    - **配置加载**: `Config.from_args_and_env()` 创建并填充配置对象.
    - **分布式管理器**: 创建 `DistributedManager` 实例, 并调用 `get_world_size()` 来确定需要启动的进程数.
    - **日志设置**: 为主进程(Rank 0)设置 `Logger`.

3. **工作进程派生 (DDP)**
    - 如果 `world_size` > 1, 主进程调用 `torch.multiprocessing.spawn()`.
    - `spawn` 会创建 `world_size` 个新的工作进程, 每个进程都从 `train_worker()` 函数开始执行, 并被分配一个唯一的 `rank` (从 0 到 `world_size - 1`).
    - 如果 `world_size` == 1, 则直接在主进程中调用 `train_worker(rank=0, world_size=1, ...)`.

4. **工作进程设置 (`train_worker` 函数)**
    - **分布式环境设置**: 每个进程调用 `distributed_manager.setup()`, 初始化进程组 (`torch.distributed.init_process_group`).
    - **数据模块准备**: 实例化 `DataModule`, 并调用 `.prepare_data()` 和 `.setup()`.
    - **任务和回调实例化**: 实例化具体的 `TrainingTask` 子类和所有定义的回调(Callbacks).
    - **训练引擎初始化**: 实例化 `TrainingEngine`.

5. **训练引擎初始化 (`TrainingEngine.__init__`)**
    - 引擎接收 `config`, `task`, `data_module`, `callbacks` 等对象.
    - 调用 `_setup_components()` 方法, 该方法会:
        - 调用 `task.build_model()` 创建模型, 并移至当前进程对应的设备. **模型构建时, `TransformerBlock` 会根据配置(`use_moe`、`num_experts`、`top_k`)来决定使用标准 MLP 还是 MoE 层.**
        - (可选)编译模型 (`torch.compile`).
        - (可选)用 `DDP` 包装模型.
        - 调用 `task.build_optimizer()`, `task.build_scheduler()`, `task.build_criterion()` 创建优化器、调度器和损失函数.
        - 调用 `data_module.train_dataloader()` 和 `val_dataloader()` 获取数据加载器.
        - 调用 `checkpoint_manager.load_checkpoint()` 尝试恢复训练状态.

6. **执行训练循环 (`TrainingEngine.run`)**
    - `TrainingEngine` 调用自身的 `run()` 方法, 启动主训练循环.
    - 循环遍历 `epochs`, 在每个 `epoch` 内, 依次调用 `_run_epoch()` 和 `_run_validation_epoch()`.
    - 在 `epoch` 结束后, 更新学习率调度器 (`scheduler.step()`) 并保存检查点 (`checkpoint_manager.save_checkpoint()`).

7. **清理**
    - 训练循环结束后(或出现异常时), `finally` 块确保 `distributed_manager.cleanup()` 被调用.\
    - `cleanup()` 会销毁进程组 (`torch.distributed.destroy_process_group`), 释放资源.

---

## 深入了解

为了更清晰地理解特定部分的交互逻辑, 请参阅以下详细文档:

- **[回调机制调用流程](./flow-callbacks.md)**: 详细解释了在训练循环中, 回调函数在何时以及如何被触发.
