# Train Instruction

## 单机 4 卡

```bash
export NUM_GPUS_PER_NODE=2
export EPOCHS=10
python train_01.py
```

## 双机各 4 卡

```bash
# 主节点
export MASTER_ADDR=192.168.1.100
export NUM_NODES=2
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
python train_01.py

# 从节点
export MASTER_ADDR=192.168.1.100
export NUM_NODES=2
export NUM_GPUS_PER_NODE=4
export NODE_RANK=1
python train_01.py
```
