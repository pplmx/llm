# 指南：配置训练任务 (`GUIDE_CONFIGURATION.md`)

本指南详细说明了如何配置、覆盖和管理训练过程中的所有设置。框架的配置系统非常灵活，支持通过默认值、YAML 文件、命令行参数和环境变量进行设置。

## 配置的层级结构

配置系统是分层的，优先级从低到高如下：

1.  **数据类中的默认值**: 这是最底层的设置，在 `core/config.py` 中定义。
2.  **YAML 配置文件**: 您可以提供一个 YAML 文件来覆盖默认值。
3.  **命令行参数**: 在运行时指定的参数具有最高优先级，可以覆盖 YAML 文件和默认值。
4.  **环境变量**: 主要用于分布式训练，用于设置 `MASTER_ADDR` 等参数。

## 方法一：使用命令行参数 (最常用)

对于快速实验和微调，直接在命令行中覆盖参数是最方便的。框架暴露了最常用的一些参数。

```bash
# 运行一个训练任务，并覆盖学习率和批量大小
python -m llm.training.train --task regression --lr 0.005 --batch-size 256

# 禁用 torch.compile 以进行调试
python -m llm.training.train --task regression --no-compile

# 从指定的检查点恢复训练
python -m llm.training.train --task regression --resume-from-checkpoint checkpoints/latest.pt
```

-   要查看所有可用的命令行参数，请运行 `python -m llm.training.train --help`。
-   注意：并非 `Config` 中的所有参数都暴露在命令行中，只有最关键的那些被暴露出来以保持简洁性。

## 方法二：使用 YAML 配置文件 (推荐用于可复现的实验)

对于正式的、可复现的实验，强烈建议使用 YAML 文件来管理配置。这使得跟踪和版本控制实验设置变得容易。

1.  **创建一个 `config.yaml` 文件**:

    ```yaml
    # config.yaml
    model:
      hidden_size: 256
      num_layers: 4
      dropout: 0.15

    training:
      epochs: 50
      batch_size: 256
      lr: 0.001
      scheduler_type: cosine
      warmup_epochs: 5

    optimization:
      use_compile: true
      use_amp: true
      num_workers: 8

    checkpoint:
      checkpoint_dir: "checkpoints/my_experiment"
      save_best: true

    logging:
      log_dir: "logs/my_experiment"
      log_interval: 50
    ```

2.  **在代码中加载它**: (这需要对 `train.py` 进行少量修改)

    目前 `train.py` 默认从命令行和环境变量加载配置。要支持从 YAML 加载，您需要修改 `main` 函数：

    ```python
    # 在 train.py 的 main 函数中
    import argparse

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    # ...

    args, remaining_argv = parser.parse_known_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_args_and_env()
    
    # ... 后续逻辑保持不变 ...
    ```
    *注意：此修改尚未应用，此处仅为示例。*

## 方法三：使用环境变量 (分布式训练)

在进行多节点分布式训练时，通常使用环境变量来告知每个进程其角色和如何找到主进程。

```bash
# 节点 0 (主节点)
export MASTER_ADDR='10.1.1.1'
export MASTER_PORT='12355'
export NUM_NODES=2
export NODE_RANK=0
export GPUS_PER_NODE=8
python -m llm.training.train --task ...

# 节点 1
export MASTER_ADDR='10.1.1.1' # 与主节点相同
export MASTER_PORT='12355'  # 与主节点相同
export NUM_NODES=2
export NODE_RANK=1
export GPUS_PER_NODE=8
python -m llm.training.train --task ...
```

`DistributedManager` 会自动从环境变量中读取这些值来初始化 `torch.distributed`。
