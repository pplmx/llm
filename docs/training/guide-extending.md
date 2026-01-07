# 指南: 扩展训练框架 (`GUIDE_EXTENDING.md`)

本框架的核心优势在于其可扩展性. 本指南将通过几个“食谱式”的示例, 向您展示如何轻松地添加新功能.

---

### 食谱1: 如何添加一个新的学习率调度器？

假设我们想添加一个 `ExponentialLR` 调度器.

**第1步: 在 `config.py` 中添加选项**

在 `TrainingConfig` 数据类中, 我们可以为 `scheduler_type` 添加一个新的有效选项的注释, 以方便其他开发者知道它的存在.

```python
# in core/config.py
@dataclass
class TrainingConfig:
    # ...
    scheduler_type: str = "cosine"  # 新增: cosine, step, plateau, exponential
    # ...
```

**第2步: 在 `TrainingTask` 中实现逻辑**

在您的 `TrainingTask` 子类(例如 `RegressionTask`)的 `build_scheduler` 方法中, 根据配置添加对新调度器的支持.

```python
# in tasks/regression_task.py
from torch.optim.lr_scheduler import ExponentialLR # 导入新的调度器
from torch import optim # 导入 optim

class RegressionTask(TrainingTask):
    # ...
    def build_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        # 根据配置动态创建调度器
        if self.config.training.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.epochs)
        elif self.config.training.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.training.step_size, gamma=self.config.training.gamma)
        elif self.config.training.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        elif self.config.training.scheduler_type == "exponential": # <-- 新增逻辑
            scheduler = ExponentialLR(optimizer, gamma=self.config.training.gamma)
        else:
            scheduler = None # 或者抛出错误

        # ... 后续的 warmup 逻辑保持不变 ...
        return scheduler
```

现在, 您就可以通过配置 `--scheduler-type exponential` 来使用新的调度器了.

---

### 食谱2: 如何添加一个新的回调？

假设我们想创建一个在训练开始和结束时打印一条自定义消息的回调.

**第1步: 创建 `Callback` 子类**

在 `core/callbacks.py` 中(或一个新文件中), 创建一个新类.

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

**第2步: 通过配置动态添加回调**

为了保持 `train.py` 的简洁性和灵活性, 我们推荐通过配置来动态添加回调. 这需要您在 `Config` 中定义一个回调列表, 并在 `train.py` 中根据配置实例化它们.

**在 `Config` 中定义回调配置 (示例)**:

```python
# in core/config.py
from typing import List

@dataclass
class TrainingConfig:
    # ...
    callbacks: List[str] = field(default_factory=lambda: ["MetricsLogger", "TensorBoardLogger", "LRSchedulerCallback"])
    # ...
```

**在 `train.py` 中实例化回调**:

```python
# in train.py
from llm.training.core import callbacks as training_callbacks # 导入回调模块

def train_worker(rank: int, world_size: int, config: Config, task_class):
    # ...
    instantiated_callbacks = []
    for cb_name in config.training.callbacks:
        if hasattr(training_callbacks, cb_name):
            cb_class = getattr(training_callbacks, cb_name)
            # 根据回调类型传递必要的参数
            if cb_name == "TensorBoardLogger":
                instantiated_callbacks.append(cb_class(log_dir=config.logging.log_dir))
            else:
                instantiated_callbacks.append(cb_class())
        else:
            config.logger.warning(f"Callback {cb_name} not found. Skipping.")

    engine = TrainingEngine(
        config,
        task,
        rank,
        world_size,
        data_module=data_module,
        callbacks=instantiated_callbacks, # 使用动态实例化的回调
    )
    engine.run()
```

现在, 您可以通过修改 `config.yaml` 或命令行参数来轻松添加或移除回调, 例如 `--training-callbacks MetricsLogger TensorBoardLogger WelcomeMessage`.

---

### 食谱3: 如何添加一个全新的训练任务？

这是最常见的扩展方式. 假设您要添加一个图像分类任务.

1. **创建数据模块**: 创建一个新的 `DataModule` 子类(例如 `ImageNetDataModule`), 负责下载、预处理和加载您的数据.
2. **创建模型**: 创建一个新的 `nn.Module`(例如 `ResNet`). **请注意, 模型应根据 `Config` 中的参数进行构建, 而不是硬编码.**
3. **创建任务类**: 创建一个新的 `TrainingTask` 子类, 例如 `ClassificationTask`.

    ```python
    # in tasks/classification_task.py
    from .base_task import TrainingTask
    from my_models import ResNet # 假设您定义了 ResNet
    from llm.models.decoder import DecoderModel # 导入 DecoderModel

    class ClassificationTask(TrainingTask):
        def build_model(self) -> nn.Module:
            # 示例: 根据配置构建模型
            if self.config.model.use_moe:
                # 如果配置中启用了 MoE, 则构建一个带有 MoE 的 DecoderModel
                return DecoderModel(
                    vocab_size=self.config.model.vocab_size, # 假设 vocab_size 在 ModelConfig 中
                    hidden_size=self.config.model.hidden_size,
                    num_layers=self.config.model.num_layers,
                    num_heads=self.config.model.num_heads,
                    use_moe=self.config.model.use_moe,
                    num_experts=self.config.model.num_experts,
                    top_k=self.config.model.top_k,
                    # ... 其他参数
                )
            else:
                # 否则, 构建一个标准的 DecoderModel
                return DecoderModel(
                    vocab_size=self.config.model.vocab_size,
                    hidden_size=self.config.model.hidden_size,
                    num_layers=self.config.model.num_layers,
                    num_heads=self.config.model.num_heads,
                    # ... 其他参数
                )

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

4. **在 `train.py` 中注册新任务**

    ```python
    # in train.py
    from llm.training.tasks.regression_task import RegressionTask
    from llm.training.tasks.classification_task import ClassificationTask # <-- 导入新任务

    AVAILABLE_TASKS = {
        "regression": RegressionTask,
        "classification": ClassificationTask, # <-- 注册新任务
    }
    ```

现在, 您可以通过运行 `python -m llm.training.train --task classification` 来启动您的新任务.

---

### 食谱4: 如何启用 MoE (Mixture of Experts) 功能？

本项目框架支持 MoE 架构, 您可以通过配置轻松启用它.

**第1步: 在 `Config` 中配置 MoE 参数**

在 `core/config.py` 的 `ModelConfig` 中, 设置 `use_moe` 为 `True`, 并指定 `num_experts` 和 `top_k`.

```python
# in config.yaml (或通过命令行参数)
model:
  hidden_size: 512
  num_layers: 2
  # ... 其他模型参数
  use_moe: true       # 启用 MoE
  num_experts: 8      # 专家总数
  top_k: 2            # 每个 token 激活的专家数量
```

**第2步: 运行训练**

当您运行训练时, `TrainingEngine` 会根据 `Config` 中的设置, 在 `TransformerBlock` 中自动实例化 MoE 层而不是标准 MLP.

```bash
llm-train --task regression --model-use-moe --model-num-experts 8 --model-top-k 2
```

通过这种方式, 您可以轻松地在模型中启用和配置 MoE 功能, 而无需修改核心模型代码.
