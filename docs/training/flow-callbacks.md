# 回调机制调用流程 (`FLOW_CALLBACKS.md`)

回调(Callbacks)是训练框架中实现“钩子”(Hooks)功能的关键机制. 它允许在训练过程的特定时间点执行自定义代码, 而无需修改 `TrainingEngine` 的核心逻辑. `TrainingEngine` 负责在预定义的事件点触发所有注册的回调方法.

以下是回调方法在 `TrainingEngine` 的生命周期中被调用的确切顺序.

## 1. 训练开始

当 `TrainingEngine.run()` 方法被调用时, 在任何训练循环开始之前, 会立即触发:

- **`Callback.on_train_start()`**
  - **触发点**: `TrainingEngine.run()` 的开头.
  - **用途**: 用于执行一次性的设置任务, 例如初始化日志(如 `TensorBoardLogger` 创建 `SummaryWriter`), 或重置状态(如 `EarlyStopping` 重置 `wait` 计数器).

## 2. Epoch 开始

在每个 `epoch` 的训练循环开始时:

- **`Callback.on_epoch_start(epoch)`**
  - **触发点**: `TrainingEngine.run()` 内部的 `for epoch in ...` 循环的开始处.
  - **用途**: 准备 `epoch` 级别的任务, 例如记录 `epoch` 开始的日志.

## 3. Batch 开始 (训练)

在每个训练 `batch` 处理之前:

- **`Callback.on_batch_start(epoch, batch_idx)`**
  - **触发点**: `TrainingEngine._run_epoch()` 内部的 `for batch_idx, batch in ...` 循环的开始处.
  - **用途**: 执行 `batch` 开始前的准备工作.

## 4. 训练步骤结束

在一个训练 `batch` 完成前向传播、损失计算、反向传播和优化器步骤之后:

- **`Callback.on_train_step_end(epoch, batch_idx, loss, metrics)`**
  - **触发点**: `TrainingEngine._run_epoch()` 中, 在 `scaler.step(optimizer)` 和 `scaler.update()` 之后.
  - **用途**: 这是记录 `batch` 级别指标最常用的钩子. 例如, `TensorBoardLogger` 在这里记录 `batch` 损失, `LRSchedulerCallback` 在这里记录当前的学习率.

## 5. Batch 结束 (训练)

在每个训练 `batch` 的所有处理完成之后:

- **`Callback.on_batch_end(epoch, batch_idx)`**
  - **触发点**: `TrainingEngine._run_epoch()` 内部的 `for batch_idx, batch in ...` 循环的末尾.
  - **用途**: 执行 `batch` 结束后的清理或检查任务.

## 6. 验证开始

如果启用了验证, 在验证循环开始之前:

- **`Callback.on_validation_start(epoch)`**
  - **触发点**: `TrainingEngine._run_validation_epoch()` 的开头.
  - **用途**: 准备验证环境.

## 7. 验证结束

在所有验证数据处理完毕, 计算出平均验证损失之后:

- **`Callback.on_validation_end(epoch, logs)`**
  - **触发点**: `TrainingEngine._run_validation_epoch()` 的末尾.
  - **`logs`**: 包含验证结果的字典, 例如 `{'val_loss': 0.123}`.
  - **用途**: `EarlyStopping` 在这里检查 `val_loss` 是否有改善.

## 8. 保存检查点

在 `rank 0` 进程成功保存一个检查点之后:

- **`Callback.on_save_checkpoint(epoch)`**
  - **触发点**: `TrainingEngine.run()` 中, 在 `checkpoint_manager.save_checkpoint()` 调用之后.
  - **用途**: 执行与检查点保存相关的额外操作.

## 9. Epoch 结束

在一个 `epoch` 的训练和验证(如果启用)都完成之后:

- **`Callback.on_epoch_end(epoch, logs)`**
  - **触发点**: `TrainingEngine.run()` 内部的 `for epoch in ...` 循环的末尾.
  - **`logs`**: 包含该 `epoch` 的聚合指标, 例如 `{'avg_loss': ..., 'val_loss': ...}`.
  - **用途**: 这是记录和汇总 `epoch` 级别指标的地方. `MetricsLogger` 和 `TensorBoardLogger` 都在这里记录 `epoch` 的最终指标.

## 10. 异常发生

如果在训练过程中(`try` 块内)捕获到任何异常:

- **`Callback.on_exception(exception)`**
  - **触发点**: `TrainingEngine.run()` 的 `except` 块中.
  - **用途**: 执行自定义的异常处理逻辑.

## 11. 训练结束

在整个训练过程(所有 `epochs`)正常完成或被中断后, 在 `finally` 块中:

- **`Callback.on_train_end()`**
  - **触发点**: `TrainingEngine.run()` 的 `finally` 块中.
  - **用途**: 执行最终的清理工作, 例如 `TensorBoardLogger` 在这里调用 `writer.close()`.

---

通过这个详细的调用流程, 开发者可以精确地知道应该在哪个回调方法中放置自己的逻辑, 以实现预期的功能.
