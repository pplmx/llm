# 指南：扩展训练框架 (`GUIDE_EXTENDING.md`)

本框架的核心优势在于其可扩展性。本指南将通过几个“食谱式”的示例，向您展示如何轻松地添加新功能。

---

### 食谱1：如何添加一个新的学习率调度器？

假设我们想添加一个 `ExponentialLR` 调度器。

**第1步：在 `config.py` 中添加选项**

在 `TrainingConfig` 数据类中，我们可以为 `scheduler_type` 添加一个新的有效选项的注释，以方便其他开发者知道它的存在。

```python
# in core/config.py
@dataclass
class TrainingConfig:
    # ...
    scheduler_type: str = "cosine"  # 新增: cosine, step, plateau, exponential
    # ...
```

**第2步：在 `TrainingTask` 中实现逻辑**

在您的 `TrainingTask` 子类（例如 `RegressionTask`）的 `build_scheduler` 方法中，添加对新调度器的支持。

```python
# in tasks/regression_task.py
from torch.optim.lr_scheduler import ExponentialLR # 导入新的调度器

class RegressionTask(TrainingTask):
    # ...
    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        scheduler_map = {
            "cosine": optim.lr_scheduler.CosineAnnealingLR(...),
            "step": optim.lr_scheduler.StepLR(...),
            "plateau": optim.lr_scheduler.ReduceLROnPlateau(...), # 假设已存在
            "exponential": ExponentialLR(optimizer, gamma=0.9), # <-- 新增逻辑
        }
        scheduler = scheduler_map.get(self.config.training.scheduler_type)
        # ... 后续的 warmup 逻辑保持不变 ...
        return scheduler
```

现在，您就可以通过配置 `--scheduler-type exponential` 来使用新的调度器了。

---

### 食谱2：如何添加一个新的回调？

假设我们想创建一个在训练开始和结束时打印一条自定义消息的回调。

**第1步：创建 `Callback` 子类**

在 `core/callbacks.py` 中（或一个新文件中），创建一个新类。

```python
# in core/callbacks.py
class WelcomeMessage(Callback):
    def on_train_start(self, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0:
            self.engine.logger.info("======================================")
            self.engine.logger.info("🎉 Welcome to the training session! 🎉")
            self.engine.logger.info("======================================")

    def on_train_end(self, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0:
            self.engine.logger.info("======================================")
            self.engine.logger.info("👋 Training finished. Goodbye! 👋")
            self.engine.logger.info("======================================")
```

**第2步：在 `train.py` 中实例化并添加它**

在 `train_worker` 函数中，将您的新回调添加到 `callbacks` 列表中。

```python
# in train.py
from llm.training.core.callbacks import ..., WelcomeMessage # 导入新回调

def train_worker(...):
    # ...
    callbacks = [
        MetricsLogger(),
        TensorBoardLogger(...),
        LRSchedulerCallback(),
        WelcomeMessage(), # <-- 添加新的回调实例
    ]
    # ...
    engine = TrainingEngine(..., callbacks=callbacks)
    engine.run()
```

现在，每次训练运行时，都会在开始和结束时看到您的欢迎和告别消息。

---

### 食谱3：如何添加一个全新的训练任务？

这是最常见的扩展方式。假设您要添加一个图像分类任务。

1.  **创建数据模块**: 创建一个新的 `DataModule` 子类（例如 `ImageNetDataModule`），负责下载、预处理和加载您的数据。
2.  **创建模型**: 创建一个新的 `nn.Module`（例如 `ResNet`）。
3.  **创建任务类**: 创建一个新的 `TrainingTask` 子类，例如 `ClassificationTask`。

    ```python
    # in tasks/classification_task.py
    from .base_task import TrainingTask
    from my_models import ResNet # 假设您定义了 ResNet

    class ClassificationTask(TrainingTask):
        def build_model(self) -> nn.Module:
            return ResNet(num_classes=self.config.model.num_classes)

        def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
            return optim.SGD(model.parameters(), lr=self.config.training.lr, momentum=0.9)

        def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
            # ...

        def build_criterion(self) -> nn.Module:
            return nn.CrossEntropyLoss()

        def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
            images, labels = batch
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            return loss, {"loss": loss.item(), "accuracy": accuracy}

        def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
            # 类似 train_step 的逻辑
            ...
    ```

4.  **在 `train.py` 中注册新任务**

    ```python
    # in train.py
    from llm.training.tasks.regression_task import RegressionTask
    from llm.training.tasks.classification_task import ClassificationTask # <-- 导入新任务

    AVAILABLE_TASKS = {
        "regression": RegressionTask,
        "classification": ClassificationTask, # <-- 注册新任务
    }
    ```

现在，您可以通过运行 `python -m llm.training.train --task classification` 来启动您的新任务。
