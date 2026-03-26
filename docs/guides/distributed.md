# 分布式训练指南

使用多 GPU 加速训练。

## 概述

支持两种分布式训练方式：

- **DDP (Distributed Data Parallel)** - 数据并行
- **FSDP (Fully Sharded Data Parallel)** - 即将支持

---

## DDP 快速开始

### 单机多卡

```bash
# 使用 2 个 GPU
torchrun --nproc_per_node=2 scripts/train.py

# 使用 4 个 GPU
torchrun --nproc_per_node=4 scripts/train.py
```

### 多机多卡

```bash
# 节点 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train.py

# 节点 1
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    --node_rank=1 \
    scripts/train.py
```

---

## 训练脚本配置

### 修改代码支持 DDP

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    # 1. 初始化进程组
    dist.init_process_group(backend="nccl")

    # 2. 创建模型并移动到 GPU
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 3. 训练循环
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # 4. 清理
    dist.destroy_process_group()
```

---

## 配置说明

### 环境变量

| 变量                 | 说明            | 默认值 |
| -------------------- | --------------- | ------ |
| `NCCL_DEBUG`         | NCCL 调试信息   | WARN   |
| `NCCL_IB_DISABLE`    | 禁用 InfiniBand | 0      |
| `NCCL_NET_GDR_LEVEL` | RDMA 级别       | 2      |

### 启动参数

| 参数               | 说明                  |
| ------------------ | --------------------- |
| `--nnodes`         | 节点数量              |
| `--nproc_per_node` | 每节点进程数 (GPU 数) |
| `--node_rank`      | 节点编号              |
| `--master_addr`    | 主节点地址            |
| `--master_port`    | 主节点端口            |

---

## 性能优化

### 1. 通信优化

```bash
# 使用 GPUDirect RDMA (需要硬件支持)
export NCCL_NET_GDR_LEVEL=2

# 禁用 IB
export NCCL_IB_DISABLE=1
```

### 2. 梯度同步

```python
# 使用 gradient_as_bucket_view 减少内存
model = DDP(model, gradient_as_bucket_view=True)
```

### 3. 批量同步

```python
# 减少同步频率
model = DDP(model, broadcast_buffers=False)
```

---

## 监控

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

### NCCL 调试

```bash
export NCCL_DEBUG=INFO
torchrun script.py
```

---

## 常见问题

**Q: NCCL 连接失败？**

- 检查 GPU 是否可见 `nvidia-smi`
- 使用 `NCCL_DEBUG=INFO` 调试

**Q: 显存不足？**

- 减小 batch size
- 使用 gradient accumulation
- 启用 mixed precision

**Q: 训练变慢？**

- 检查网络延迟
- 使用 Profiler 分析
- 确认使用 NCCL backend

---

## 相关文档

- [Deep Dive DDP](../development/deep-dive-ddp.md)
- [Training Flow](../development/training-flow.md)
