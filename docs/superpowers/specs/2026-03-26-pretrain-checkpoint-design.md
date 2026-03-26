# 预训练流程完善 - 设计文档

**Date**: 2026-03-26
**Author**: AI Assistant
**Status**: Draft

## 目标

在现有 `train_simple_decoder.py` 基础上添加 Checkpoint 支持，使预训练流程更加完整。

## 设计原则

1. **简洁优先** - 不引入配置文件，保持脚本独立性
2. **优雅代码** - 清晰的代码结构，合理封装
3. **开箱即用** - 合理的默认值
4. **学习导向** - 代码易于理解，作为学习参考

## 现有能力

- 文本文件加载
- 字符级 tokenizer
- 基础训练循环
- CPU/CUDA 自动检测

## 新增功能

### 1. Checkpoint 支持

```python
class CheckpointManager:
    """Checkpoint 管理器"""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, epoch, global_step, loss, config):
        """保存 checkpoint"""
        path = self.save_dir / f"checkpoint_step_{global_step}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "config": config,
        }, path)
        return path

    def load(self, path, model, optimizer=None):
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
```

### 2. 新增 CLI 参数

| 参数              | 类型 | 默认值            | 说明              |
| ----------------- | ---- | ----------------- | ----------------- |
| `--save-dir`      | Path | `"./checkpoints"` | 保存目录          |
| `--resume`        | Path | None              | 恢复的 checkpoint |
| `--save-interval` | int  | 100               | 每 N 步保存一次   |

### 3. 优雅的训练循环

```python
def train_loop(model, dataloader, optimizer, config):
    """主训练循环，包含进度显示和优雅退出"""

    checkpoint = CheckpointManager(config.save_dir)
    global_step = 0

    for epoch in range(config.epochs):
        model.train()

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
            for batch in pbar:
                # Forward
                loss = model(**batch)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1

                # 进度显示
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # 保存 checkpoint
                if global_step % config.save_interval == 0:
                    checkpoint.save(model, optimizer, epoch, global_step, loss.item(), config)

    return model
```

## 文件修改

```text
scripts/train_simple_decoder.py
├── 新增 CheckpointManager 类
├── 新增 CLI 参数解析
├── 新增恢复训练逻辑
└── 改进训练循环 (进度条、优雅退出)
```

## 使用示例

```bash
# 基本使用 (开箱即用)
python scripts/train_simple_decoder.py --file-path data.txt

# 带 checkpoint 自动保存
python scripts/train_simple_decoder.py -f data.txt --save-dir ./ckpt

# 从 checkpoint 恢复
python scripts/train_simple_decoder.py -f data.txt --resume ./ckpt/checkpoint_step_100.pt

# 完整参数
python scripts/train_simple_decoder.py \
    --file-path data.txt \
    --save-dir ./checkpoints \
    --save-interval 50 \
    --hidden-size 256 \
    --num-layers 4 \
    --batch-size 32 \
    --epochs 3
```

## 优雅细节

| 细节             | 实现                                 |
| ---------------- | ------------------------------------ |
| **自动设备检测** | `cuda` 可用时自动使用 GPU            |
| **进度条**       | tqdm 显示 epoch/step/loss            |
| **简洁日志**     | `[Epoch 1/3] Step 100/500 loss=2.34` |
| **优雅退出**     | Ctrl+C 保存 checkpoint 后退出        |
| **首次运行提示** | 打印关键配置信息                     |
