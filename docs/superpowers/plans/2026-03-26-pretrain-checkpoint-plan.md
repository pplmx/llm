# 预训练 Checkpoint 支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 train_simple_decoder.py 中添加 Checkpoint 支持，使预训练流程更完整。

**Architecture:** 将 checkpoint 逻辑封装为 CheckpointManager 类，提供 save/load 接口，在训练循环中定期保存。

**Tech Stack:** PyTorch, tqdm, typer

---

## 文件结构

```
scripts/train_simple_decoder.py
├── 新增: CheckpointManager 类
├── 新增: CLI 参数 (--save-dir, --resume, --save-interval)
└── 修改: 训练循环 (集成 checkpoint)
```

---

## Task 1: 添加 CLI 参数

**Files:**
- Modify: `scripts/train_simple_decoder.py:18-45`

- [ ] **Step 1: 读取现有代码结构**

```bash
head -50 scripts/train_simple_decoder.py
```

- [ ] **Step 2: 添加新参数**

找到 `main` 函数定义，在现有参数后添加:

```python
    save_dir: Path = typer.Option(
        Path("./checkpoints"),
        help="Directory to save checkpoints.",
    ),
    resume: Path | None = typer.Option(
        None,
        help="Path to checkpoint to resume from.",
    ),
    save_interval: int = typer.Option(
        100,
        help="Save checkpoint every N steps.",
    ),
```

- [ ] **Step 3: 测试参数解析**

```bash
python scripts/train_simple_decoder.py --help
```
Expected: 显示新参数

- [ ] **Step 4: 提交**

```bash
git add scripts/train_simple_decoder.py
git commit -m "feat: add checkpoint CLI parameters"
```

---

## Task 2: 实现 CheckpointManager 类

**Files:**
- Modify: `scripts/train_simple_decoder.py` (在文件顶部添加)

- [ ] **Step 1: 添加 CheckpointManager 类**

在 import 之后、@app.command 之前添加:

```python
from pathlib import Path
from typing import Any
import torch
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """训练配置，用于保存到 checkpoint"""
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-4
    file_path: str = ""
    # 新增参数
    save_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    save_interval: int = 100


class CheckpointManager:
    """Checkpoint 管理器"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model, optimizer, epoch: int, global_step: int, loss: float, config: TrainConfig):
        """保存 checkpoint"""
        path = self.save_dir / f"checkpoint_step_{global_step}.pt"
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "config": {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "max_seq_len": config.max_seq_len,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "lr": config.lr,
            },
        }, path)
        
        # 同时保存最新 checkpoint 链接
        latest_link = self.save_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        # 简单复制 (Windows 不支持 symlink)
        import shutil
        shutil.copy(path, latest_link)
        
        return path
    
    def load(self, path: Path, model, optimizer=None):
        """加载 checkpoint"""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint
```

- [ ] **Step 2: 提交**

```bash
git add scripts/train_simple_decoder.py
git commit -m "feat: add CheckpointManager class"
```

---

## Task 3: 集成 Checkpoint 到训练循环

**Files:**
- Modify: `scripts/train_simple_decoder.py` (主训练循环部分)

- [ ] **Step 1: 找到训练循环位置**

```bash
grep -n "for epoch in range" scripts/train_simple_decoder.py
```

- [ ] **Step 2: 修改训练循环，添加 checkpoint 保存**

找到训练循环开始前，添加:

```python
# 初始化 checkpoint manager
checkpoint_mgr = CheckpointManager(config.save_dir)

# 恢复训练 (如果指定了 resume)
start_epoch = 0
global_step = 0
if resume:
    print(f"Resuming from checkpoint: {resume}")
    checkpoint = checkpoint_mgr.load(resume, model, optimizer)
    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    print(f"Resumed from epoch {start_epoch}, step {global_step}")
```

- [ ] **Step 3: 在训练循环内添加 checkpoint 保存**

在每个 batch 训练后添加:

```python
# 在 optimizer.step() 后添加
if (global_step + 1) % config.save_interval == 0:
    checkpoint_path = checkpoint_mgr.save(
        model, optimizer, epoch, global_step, loss.item(), config
    )
    print(f"\n[Checkpoint saved] {checkpoint_path}")
```

- [ ] **Step 4: 修改 epoch 循环起始**

找到:
```python
for epoch in range(epochs):
```

改为:
```python
for epoch in range(start_epoch, epochs):
```

- [ ] **Step 5: 提交**

```bash
git add scripts/train_simple_decoder.py
git commit -m "feat: integrate checkpoint into training loop"
```

---

## Task 4: 添加优雅退出处理

**Files:**
- Modify: `scripts/train_simple_decoder.py`

- [ ] **Step 1: 添加 signal 处理**

在 import 部分添加:

```python
import signal
import sys
```

在主函数开始时添加:

```python
# 优雅退出处理
should_save_checkpoint = False

def signal_handler(signum, frame):
    global should_save_checkpoint
    print("\n\nReceived interrupt signal. Saving checkpoint before exit...")
    should_save_checkpoint = True
    # 强制保存
    if 'checkpoint_mgr' in dir() and 'model' in dir():
        checkpoint_mgr.save(model, optimizer, epoch, global_step, loss.item(), config)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

- [ ] **Step 2: 提交**

```bash
git add scripts/train_simple_decoder.py
git commit -m "feat: add graceful shutdown with checkpoint"
```

---

## Task 5: 添加进度条 (可选)

**Files:**
- Modify: `scripts/train_simple_decoder.py`

- [ ] **Step 1: 添加 tqdm 进度条**

在训练循环处添加:

```python
# 在 import 部分添加
from tqdm import tqdm

# 在训练循环处
for epoch in range(start_epoch, epochs):
    model.train()
    
    # 使用 tqdm 包装 dataloader
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in pbar:
        # ... 训练代码 ...
        
        # 更新进度条
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "step": global_step
        })
```

- [ ] **Step 2: 提交**

```bash
git add scripts/train_simple_decoder.py
git commit -m "feat: add progress bar to training"
```

---

## Task 6: 测试验证

**Files:**
- Test: 运行脚本验证功能

- [ ] **Step 1: 运行帮助命令**

```bash
python scripts/train_simple_decoder.py --help
```
Expected: 显示所有参数包括新参数

- [ ] **Step 2: 创建测试数据**

```bash
echo "hello world this is a test" > /tmp/test_data.txt
```

- [ ] **Step 3: 运行训练 (1 step, 快速测试)**

```bash
python scripts/train_simple_decoder.py \
    --file-path /tmp/test_data.txt \
    --epochs 1 \
    --save-interval 1 \
    --save-dir /tmp/test_ckpt
```
Expected: 训练完成，生成 checkpoint

- [ ] **Step 4: 验证 checkpoint 存在**

```bash
ls -la /tmp/test_ckpt/
```
Expected: checkpoint_step_*.pt 和 latest.pt

- [ ] **Step 5: 从 checkpoint 恢复训练**

```bash
python scripts/train_simple_decoder.py \
    --file-path /tmp/test_data.txt \
    --epochs 1 \
    --resume /tmp/test_ckpt/latest.pt \
    --save-dir /tmp/test_ckpt2
```
Expected: 从 checkpoint 恢复训练

- [ ] **Step 6: 清理测试文件**

```bash
rm -rf /tmp/test_ckpt /tmp/test_ckpt2 /tmp/test_data.txt
```

- [ ] **Step 7: 最终提交**

```bash
git add -A
git commit -m "feat: complete checkpoint support for pretraining"
```

---

## 验证清单

| 任务 | 验证项 |
|------|--------|
| Task 1 | CLI 参数正确解析 |
| Task 2 | CheckpointManager 类正常工作 |
| Task 3 | 训练中 checkpoint 自动保存 |
| Task 4 | Ctrl+C 优雅退出 |
| Task 5 | 进度条显示 (可选) |
| Task 6 | 完整功能测试通过 |

---

## 使用示例

```bash
# 基本使用
python scripts/train_simple_decoder.py --file-path data.txt

# 自动保存 checkpoint
python scripts/train_simple_decoder.py -f data.txt --save-dir ./ckpt

# 从 checkpoint 恢复
python scripts/train_simple_decoder.py -f data.txt --resume ./ckpt/latest.pt

# 完整参数
python scripts/train_simple_decoder.py \
    -f data.txt \
    --save-dir ./ckpt \
    --save-interval 50 \
    --hidden-size 256 \
    --num-layers 4 \
    --batch-size 32 \
    --epochs 3
```

---

Plan complete and saved to `docs/superpowers/plans/2026-03-26-pretrain-checkpoint-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?