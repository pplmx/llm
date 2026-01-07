# 从零开始的 LLM 训练指南 - CPU 版本 (概念演示)

> [!NOTE]
> 这是一份**概念性**的入门指南，旨在帮助初学者理解训练流程的基本原理。
>
> 如果您想学习如何使用本项目提供的**正式框架**（包含 DDP、AMP、可配置化等高级特性），请阅读官方教程：
> 👉 [LLM 框架教程: 从零开始构建与训练](../../docs/tutorial-cpu-llm.md)

## 目录

- [阶段一: 环境准备与基础知识(1-2天)](#阶段一环境准备与基础知识1-2天)
- [阶段二: 数据处理与预训练模型(2-3天)](#阶段二数据处理与预训练模型2-3天)
- [阶段三: 模型训练与优化(3-4天)](#阶段三模型训练与优化3-4天)
- [阶段四: 评估与扩展(2-3天)](#阶段四评估与扩展2-3天)

## 阶段一: 环境准备与基础知识(1-2天)

### 1.1 环境配置

首先创建项目目录并使用 `uv` 初始化虚拟环境:

```bash
mkdir llm-training
cd llm-training
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

创建 `pyproject.toml`:

```toml
[project]
name = "llm-training"
version = "0.1.0"
description = "LLM training project"
requires-python = ">=3.9"

dependencies = [
    "torch>=2.2.0",
    "transformers>=4.37.0",
    "datasets>=2.17.0",
    "accelerate>=0.27.0",
    "evaluate>=0.4.0",
    "scikit-learn>=1.4.0",
    "tensorboard>=2.15.0",
    "tqdm>=4.66.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
```

使用 uv 安装依赖:

```bash
uv pip install -e .
```

### 1.2 基础配置文件

创建配置文件 `config.py`:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128

    # 训练配置
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500

    # 数据配置
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "test"

    # 硬件配置
    device: str = "cpu"
    num_workers: int = 4

    # 输出配置
    output_dir: str = "outputs"
    logging_steps: int = 100
    save_steps: int = 1000


config = TrainingConfig()
```

### 1.3 工具函数

创建 `utils.py`:

```python
import os
import logging
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)

    return logger


def create_dataloaders(
    dataset,
    tokenizer,
    config,
    shuffle: bool = True
) -> DataLoader:
    """创建数据加载器"""

    def collate_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt"
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )


def get_optimizer_and_scheduler(
    model,
    config,
    num_training_steps: int
):
    """配置优化器和学习率调度器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler
```

## 阶段二: 数据处理与预训练模型(2-3天)

### 2.1 数据加载与预处理

创建 `data.py`:

```python
from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_preprocess_data(config):
    """加载并预处理数据集"""
    # 加载数据集
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 定义预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )

    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset, tokenizer


def prepare_dataloaders(dataset, tokenizer, config):
    """准备数据加载器"""
    train_dataloader = create_dataloaders(
        dataset["train"],
        tokenizer,
        config,
        shuffle=True
    )

    eval_dataloader = create_dataloaders(
        dataset["test"],
        tokenizer,
        config,
        shuffle=False
    )

    return train_dataloader, eval_dataloader
```

### 2.2 模型定义

创建 `model.py`:

```python
from transformers import AutoModelForMaskedLM


def create_model(config):
    """创建和配置模型"""
    model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    model = model.to(config.device)
    return model
```

## 阶段三: 模型训练与优化(3-4天)

### 3.1 训练循环

创建 `trainer.py`:

```python
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config

        # 设置优化器和调度器
        num_training_steps = len(train_dataloader) * config.num_epochs
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            model, config, num_training_steps
        )

        # 设置tensorboard
        self.writer = SummaryWriter(config.output_dir)
        self.logger = setup_logging("trainer")

    def train(self):
        """训练循环"""
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}")

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}"
            )

            for step, batch in enumerate(progress_bar):
                # 将数据移动到设备
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                # 优化器步进
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item()})

                # 记录训练指标
                if step % self.config.logging_steps == 0:
                    self.writer.add_scalar(
                        "train/loss",
                        loss.item(),
                        step
                    )

                # 保存检查点
                if step % self.config.save_steps == 0:
                    self.save_checkpoint(epoch, step)

            # 每个epoch结束后评估
            eval_loss = self.evaluate()
            self.logger.info(f"Epoch {epoch + 1} evaluation loss: {eval_loss}")

    def evaluate(self):
        """评估循环"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.eval_dataloader)

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, step):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{epoch}-{step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(checkpoint_dir)

        # 保存训练状态
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, "training_state.pt"))
```

### 3.2 主训练脚本

创建 `train.py`:

```python
from config import config
from data import load_and_preprocess_data, prepare_dataloaders
from model import create_model
from trainer import Trainer


def main():
    # 加载数据
    dataset, tokenizer = load_and_preprocess_data(config)
    train_dataloader, eval_dataloader = prepare_dataloaders(
        dataset, tokenizer, config
    )

    # 创建模型
    model = create_model(config)

    # 初始化训练器
    trainer = Trainer(
        model,
        train_dataloader,
        eval_dataloader,
        config
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
```

## 阶段四: 评估与扩展(2-3天)

### 4.1 模型评估

创建 `evaluate.py`:

```python
import torch
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate_model(model, eval_dataloader, config):
    """评估模型性能"""
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)

            # 获取预测结果
            pred = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(pred.cpu().numpy())

            # 获取真实标签
            refs = batch["labels"]
            references.extend(refs.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(references, predictions)

    # 计算困惑度
    perplexity = torch.exp(torch.tensor(outputs.loss)).item()

    return {
        "accuracy": accuracy,
        "perplexity": perplexity
    }
```

### 4.2 性能优化建议

对于CPU训练, 以下是一些优化建议:

1. 数据加载优化:
    - 使用 `num_workers` 进行并行数据加载
    - 适当调整 `batch_size`
    - 使用 `pin_memory=True` 加速数据传输

2. 模型优化:
    - 使用混合精度训练(虽然在CPU上收益较小)
    - 使用梯度累积减少内存占用
    - 选择较小的模型架构(如DistilBERT)

3. 训练策略优化:
    - 使用较小的序列长度
    - 实施早停机制
    - 使用学习率预热和衰减

### 4.3 扩展建议

1. 硬件升级路线:
    - 首先升级到更大容量的内存(128GB)
    - 添加入门级GPU(如NVIDIA RTX 3060)
    - 考虑使用云服务进行大规模训练

2. 分布式训练准备:
    - 学习PyTorch DDP(DistributedDataParallel)
    - 了解Hugging Face Accelerate库
    - 研究模型并行和数据并行策略

## 使用说明

1. 克隆代码仓库:

```bash
git clone <repository-url>
cd llm-training
```

1. 创建虚拟环境:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

1. 安装依赖:

```bash
uv pip install -e .
```

1. 运行训练:

```bash
python best_minimal_train.py
```

## 训练过程优化

### 1. 内存优化

创建 `memory_utils.py`:

```python
import gc
import torch
from typing import Optional


def optimize_memory_usage(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    hidden_size: int,
    available_memory: Optional[float] = None
) -> dict:
    """
    计算并优化内存使用
    """
    # 估算单个样本的内存占用
    sample_size = sequence_length * hidden_size * 4  # float32 = 4 bytes
    batch_memory = sample_size * batch_size

    # 估算模型参数的内存占用
    model_memory = (vocab_size * hidden_size + hidden_size * hidden_size) * 4

    # 估算优化器状态的内存占用
    optimizer_memory = model_memory * 2  # AdamW 需要存储动量和方差

    total_memory = batch_memory + model_memory + optimizer_memory

    # 返回优化建议
    return {
        "recommended_batch_size": min(batch_size, max(1,
                                                      batch_size * available_memory // total_memory)) if available_memory else batch_size,
        "estimated_memory_usage": total_memory / (1024 ** 3),  # Convert to GB
        "can_use_gradient_accumulation": total_memory > (available_memory if available_memory else float('inf'))
    }


def clear_memory():
    """
    清理未使用的内存
    """
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 2. 梯度累积实现

更新 `trainer.py` 添加梯度累积支持:

```python
class GradientAccumulationTrainer(Trainer):
    def __init__(self, *args, accumulation_steps: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def train(self):
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}")

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # 将数据移动到设备
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss / self.accumulation_steps  # 缩放损失

                # 反向传播
                loss.backward()

                # 每 accumulation_steps 步进行一次优化器更新
                if (step + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item() * self.accumulation_steps})

                # 记录训练指标
                if step % self.config.logging_steps == 0:
                    self.writer.add_scalar(
                        "train/loss",
                        loss.item() * self.accumulation_steps,
                        step
                    )
```

### 3. 提前停止机制

创建 `early_stopping.py`:

```python
class EarlyStopping:
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if self.mode == "min":
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_loss > self.best_loss + self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False
```

## 高级功能扩展

### 1. 自定义数据集支持

创建 `custom_dataset.py`:

```python
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import json


class CustomTextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 移除批次维度
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }

        # 添加标签(如果有)
        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item

    @classmethod
    def from_json(cls, json_path: str, **kwargs):
        """从JSON文件加载数据集"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item["text"] for item in data]
        labels = [item.get("label") for item in data] if "label" in data[0] else None

        return cls(texts, labels, **kwargs)
```

### 2. 实验跟踪

创建 `experiment_tracking.py`:

```python
import json
import os
from datetime import datetime
from typing import Dict, Any


class ExperimentTracker:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metrics = []
        self.config = None

    def log_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        self.config = config
        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录训练指标"""
        metrics["step"] = step
        self.metrics.append(metrics)

        with open(os.path.join(self.experiment_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_artifact(self, name: str, content: Any):
        """保存实验产物"""
        artifact_path = os.path.join(self.experiment_dir, "artifacts", name)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(artifact_path, "w") as f:
                json.dump(content, f, indent=2)
        else:
            with open(artifact_path, "w") as f:
                f.write(str(content))
```

## 进阶学习路线建议

1. 理论深化:
    - 学习 Transformer 架构详细原理
    - 研究不同的注意力机制
    - 了解各种预训练任务的设计

2. 实践提升:
    - 尝试实现简单的 Transformer 模块
    - 探索不同的优化器和学习率调度策略
    - 实验各种正则化技术

3. 工程技能:
    - 学习分布式训练框架(如 PyTorch Lightning)
    - 掌握模型量化和压缩技术
    - 研究模型部署和服务化方案

## 常见问题解答

1. 内存不足问题:
    - 减小批次大小
    - 使用梯度累积
    - 实施模型检查点机制
    - 使用较小的序列长度

2. 训练速度慢:
    - 使用较小的模型架构
    - 优化数据加载流程
    - 实施多进程数据预处理
    - 考虑使用预计算的特征

3. 模型效果不佳:
    - 检查数据质量和预处理流程
    - 调整学习率和优化器参数
    - 增加训练轮次
    - 使用交叉验证选择最佳参数

## 项目结构和代码组织

```
llm-training/
├── .venv/                    # 虚拟环境目录
├── data/                     # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 预处理后的数据
├── src/                     # 源代码
│   ├── __init__.py
│   ├── config.py           # 配置文件
│   ├── data.py             # 数据处理
│   ├── model.py            # 模型定义
│   ├── trainer.py          # 训练器
│   ├── evaluate.py         # 评估脚本
│   ├── utils/              # 工具函数
│   │   ├── __init__.py
│   │   ├── memory_utils.py
│   │   └── logging_utils.py
│   └── experiments/        # 实验跟踪
│       ├── __init__.py
│       └── tracking.py
├── notebooks/               # Jupyter notebooks
│   ├── 1_data_exploration.ipynb
│   └── 2_model_analysis.ipynb
├── tests/                   # 单元测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── outputs/                 # 模型输出
│   ├── checkpoints/
│   └── logs/
├── examples/                # 示例代码
│   └── custom_dataset_example.py
├── scripts/                 # 实用脚本
│   ├── prepare_data.sh
│   └── train_model.sh
├── pyproject.toml          # 项目配置
├── README.md               # 项目说明
└── requirements.txt        # 依赖清单
```

## 错误处理和调试指南

### 1. 常见错误及解决方案

1. 内存相关错误:

    ```python
    # 错误: RuntimeError: out of memory
    # 解决方案: 实现渐进式数据加载
    from typing import Iterator, List


    def batch_generator(data: List, batch_size: int) -> Iterator:
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]


    # 使用示例
    for batch in batch_generator(large_dataset, batch_size=32):
        process_batch(batch)
    ```

2. 数据加载错误:

    ```python
    # 错误: FileNotFoundError: [Errno 2] No such file or directory
    # 解决方案: 添加路径检查
    def safe_load_data(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        try:
            # 尝试不同的编码
            for encoding in ['utf-8', 'gbk', 'latin1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError("无法识别文件编码")
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
    ```

3. 模型训练错误:

    ```python
    # 错误: Loss is NaN
    # 解决方案: 添加梯度裁剪和损失检查
    def check_loss(loss: torch.Tensor) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"检测到无效损失值: {loss.item()}")
            return False
        return True

    # 训练循环中使用
    loss = outputs.loss
    if not check_loss(loss):
        logger.info("跳过当前批次")
        continue
    ```

### 2. 调试技巧

1. 使用日志记录关键信息:

    ```python
    import logging

    def setup_debug_logging():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debug.log'),
                logging.StreamHandler()
            ]
        )

    # 在关键位置添加日志
    logging.debug(f"数据形状: {batch.shape}")
    logging.debug(f"当前学习率: {optimizer.param_groups[0]['lr']}")
    ```

2. 添加断言检查:

    ```python
    def validate_inputs(inputs: dict):
        """验证模型输入"""
        assert 'input_ids' in inputs, "缺少 input_ids"
        assert 'attention_mask' in inputs, "缺少 attention_mask"
        assert inputs['input_ids'].dim() == 2, f"input_ids 维度错误: {inputs['input_ids'].dim()}"
        return True
    ```

3. 性能分析工具:

    ```python
    import cProfile
    import pstats


    def profile_function(func):
        """函数性能分析装饰器"""

        def wrapper(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                return profile.runcall(func, *args, **kwargs)
            finally:
                ps = pstats.Stats(profile)
                ps.sort_stats('cumulative')
                ps.print_stats(20)  # 打印前20行统计信息

        return wrapper


    @profile_function
    def train_epoch(model, dataloader):
        # 训练代码
        pass
    ```

## 实践任务示例

### 任务1: 文本分类

创建一个简单的文本分类模型:

```python
from transformers import DistilBertForSequenceClassification


def create_classification_model(num_labels: int = 2):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    return model


# 使用示例
model = create_classification_model()
outputs = model(**batch)
loss = outputs.loss
logits = outputs.logits
```

### 任务2: 文本生成

实现一个简单的文本生成任务:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(
    prompt: str,
    max_length: int = 50,
    num_return_sequences: int = 1
):
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.7
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# 使用示例
generated_texts = generate_text("Once upon a time")
```

### 任务3: 模型微调

```python
def fine_tune_model(
    model,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int = 3
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    return trainer


# 使用示例
trainer = fine_tune_model(model, train_dataset, eval_dataset, "./output")
```

## 资源推荐

1. 学习资源:
    - Hugging Face 课程
    - "Attention is All You Need" 论文
    - The Annotated Transformer
    - PyTorch 官方教程

2. 开源项目:
    - transformers 库
    - datasets 库
    - accelerate 库
    - PyTorch Lightning

3. 社区资源:
    - Hugging Face 论坛
    - PyTorch 论坛
    - Papers with Code
    - arXiv ML 版块

这个教程提供了一个完整的框架, 帮助你在本地 CPU 环境下开始 LLM 训练. 建议按照阶段循序渐进地学习, 确保每个概念都掌握后再进入下一阶段. 同时, 可以根据实际情况调整学习计划和进度.
