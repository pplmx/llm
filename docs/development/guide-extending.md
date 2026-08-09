# 指南: 扩展训练框架 (`GUIDE_EXTENDING.md`)

本框架的核心优势在于其可扩展性. 本指南将通过几个“食谱式”的示例, 向您展示如何轻松地添加新功能.

---

## 食谱1: 如何添加一个新的学习率调度器？

假设我们想添加一个 `ExponentialLR` 调度器.

**第1步: 在 `config.py` 中添加选项**

在 `TrainingConfig`（Pydantic `BaseModel`）中，把 `scheduler_type` 的
`pattern` 校验扩展到新值：

```python
# in core/config.py
class TrainingConfig(BaseModel):
    # ...
    scheduler_type: str = Field(
        "cosine",
        pattern="^(cosine|step|plateau|exponential)$",  # 新增 exponential
    )
    # ...
```

**第2步: 在 `TrainingTask` 中实现逻辑**

在您的 `TrainingTask` 子类(例如 `RegressionTask`)的 `build_scheduler` 方法中, 根据配置添加对新调度器的支持.

```python
# in tasks/regression_task.py
from torch.optim.lr_scheduler import ExponentialLR  # 导入新的调度器
from torch import optim  # 导入 optim


class RegressionTask(TrainingTask):
    # ...
    def build_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        # 根据配置动态创建调度器
        if self.config.training.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.epochs)
        elif self.config.training.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.training.step_size, gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
        elif self.config.training.scheduler_type == "exponential":  # <-- 新增逻辑
            scheduler = ExponentialLR(optimizer, gamma=self.config.training.gamma)
        else:
            scheduler = None  # 或者抛出错误

        # ... 后续的 warmup 逻辑保持不变 ...
        return scheduler
```

现在, 在 YAML 中配置 `training.scheduler_type: exponential` 即可使用新的调度器
（`llm-train` 没有 `--scheduler-type` 之类的嵌套 CLI 参数，模型/优化器细节统一走 YAML）。

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

**第2步: 注册到训练引擎**

回调是 `TrainingEngine` 构造时传入的普通对象列表。当前 `train.py` 的
`train_worker` 里实例化了三个内置回调，把新回调加进这个列表即可：

```python
# in train.py
from llm.training.core.callbacks import (
    Callback,
    LRSchedulerCallback,
    MetricsLogger,
    TensorBoardLogger,
)

class WelcomeMessage(Callback):  # 上一步定义的子类
    ...

def train_worker(rank, world_size, config, task_name):
    # ...
    callbacks: list[Callback] = [
        MetricsLogger(),
        TensorBoardLogger(log_dir=config.logging.log_dir),
        LRSchedulerCallback(),
        WelcomeMessage(),          # <-- 新回调
    ]
    engine = TrainingEngine(
        config,
        task,
        rank,
        world_size,
        data_module=data_module,
        callbacks=callbacks,
    )
    engine.run()
```

> 注意：回调**不是**通过 YAML / CLI 配置的（没有 `--training-callbacks` 之类的参数）。
> 需要按运行环境动态组合回调时，写一个自定义入口脚本，构造 `TrainingEngine` 并传入
> 不同的回调列表即可。带可恢复状态的回调还可以覆写
> `get_checkpoint_state()` / `load_checkpoint_state()`，状态会自动并入 checkpoint。

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
            if self.config.model.mlp_impl == "moe":
                # If MoE is configured, build a DecoderModel with MoE layers
                return DecoderModel(
                    vocab_size=self.config.model.vocab_size, # assume vocab_size in ModelConfig
                    hidden_size=self.config.model.hidden_size,
                    num_layers=self.config.model.num_layers,
                    num_heads=self.config.model.num_heads,
                    mlp_impl="moe",
                    num_experts=self.config.model.num_experts,
                    top_k=self.config.model.top_k,
                    # ... other params
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

4. **注册新任务**

在 `training/tasks/builtin.py` 中调用 `TASK_REGISTRY.register(...)`，让内置 CLI 在
启动时就能看到新任务：

```python
# in training/tasks/builtin.py
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks.classification_task import ClassificationTask  # <-- 导入新任务

TASK_REGISTRY.register(
    "classification",
    ClassificationTask,
    ImageNetDataModule,
    description="Image classification task",
)
```

第三方包也可以不改仓库代码，通过 `pyproject.toml` 的 `llm.tasks` entry-point
组注册（`llm-train` 启动时会 `load_entry_point_hooks("llm.tasks")`）。

现在, 您可以通过运行 `uv run llm-train --task classification` 来启动您的新任务.

---

### 食谱4: 如何启用 MoE (Mixture of Experts) 功能？

本项目框架支持 MoE 架构, 您可以通过配置轻松启用它.

**第1步: 在 `Config` 中配置 MoE 参数**

在 `core/config.py` 的 `ModelConfig` 中, 设置 `mlp_impl` 为 `"moe"`, 并指定 `num_experts` 和 `top_k`.

```python
# in config.yaml (或通过命令行参数)
model:
  hidden_size: 512
  num_layers: 2
  # ... 其他模型参数
  mlp_impl: moe      # Enable MoE
  num_experts: 8      # 专家总数
  top_k: 2            # 每个 token 激活的专家数量
```

**第2步: 运行训练**

当您运行训练时, `TrainingEngine` 会根据 `Config` 中的设置, 在 `TransformerBlock` 中自动实例化 MoE 层而不是标准 MLP.

```bash
uv run llm-train --task stream_lm --config-path configs/moe-demo.yaml
```

通过这种方式，您可以轻松地在模型中启用和配置 MoE 功能，而无需修改核心模型代码。
`llm-train` 没有 `--model-*` 嵌套参数，模型结构一律通过 YAML 的 `model:` 段配置。
