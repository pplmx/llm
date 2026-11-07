# 指南: 配置训练任务 (`GUIDE_CONFIGURATION.md`)

本指南详细说明了如何配置、覆盖和管理训练过程中的所有设置. 框架的配置系统非常灵活, 支持通过默认值、YAML 文件、命令行参数和环境变量进行设置.

## 配置的层级结构

配置系统是分层的, 优先级从低到高如下:

1. **数据类中的默认值**: 这是最底层的设置, 在 `core/config.py` 中定义.
2. **YAML 配置文件**: 您可以提供一个 YAML 文件来覆盖默认值.
3. **命令行参数**: 在运行时指定的参数具有最高优先级, 可以覆盖 YAML 文件和默认值.
4. **环境变量**: 主要用于分布式训练, 用于设置 `MASTER_ADDR` 等参数.

## 方法一: 使用命令行参数 (最常用)

对于快速实验和微调, 直接在命令行中覆盖参数是最方便的. `llm train` 暴露了少量高频参数用于 CLI 覆盖; 其余模型/优化器细节通过 YAML 配置.

```bash
# 运行一个训练任务, 并覆盖学习率和批量大小
llm train --task regression --lr 0.001 --batch-size 32

# 禁用 torch.compile 以进行调试
llm train --task regression --no-compile

# 从指定的检查点恢复训练 (通过 YAML 配置)
llm train --task regression --config-path config_with_checkpoint.yaml

# 使用 YAML 配置进行复杂设置
llm train --task sft --config-path config.yaml

# 启用/禁用 AMP 混合精度
llm train --task regression --amp      # 启用 (默认)
llm train --task regression --no-amp   # 禁用
```

- 要查看所有可用的命令行参数, 请运行 `llm train --help`.
- 当前 CLI 暴露的参数包括: `--task` (必填), `--config-path`, `--epochs`, `--batch-size`, `--lr`, `--num-samples`, `--steps-per-epoch`, `--compile`/`--no-compile`, `--amp`/`--no-amp`.
- 完整的模型配置(如 `hidden_size`, `mlp_impl`, `num_experts` 等)建议通过 YAML 配置文件设置.

## 方法二: 使用 YAML 配置文件 (推荐用于可复现的实验)

对于正式的、可复现的实验, 强烈建议使用 YAML 文件来管理配置. 这使得跟踪和版本控制实验设置变得容易.

1. **创建一个 `config.yaml` 文件**:

    ```yaml
    # config.yaml
    model:
      hidden_size: 256
      num_layers: 4
      dropout: 0.15
      mlp_impl: moe # Enable MoE
      num_experts: 8
      top_k: 2

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

   `training.scheduler_type` 有三档，注意 `plateau` 的前提：

   | 取值             | 行为                    | 前提                            |
   | ---------------- | ----------------------- | ------------------------------- |
   | `cosine`（默认） | 余弦退火到 `lr` 的 1/10 | 无                              |
   | `step`           | 每 `epochs//2` 轮乘 0.5 | 无                              |
   | `plateau`        | 指标不再改善时乘 0.5    | **需要 `run_validation: true`** |

   `ReduceLROnPlateau` 靠"指标停止改善"触发降学习率。关掉验证
   （`training.run_validation: false`）时，引擎传给它的是**训练 loss**，而训练
   loss 在模型还在学习时会一直缓慢下降——实测 10 个 epoch 里学习率一次都没降过。
   这不是彻底失效（真正在训练集上收敛后 loss 会平掉，那时才会降），但调度会远没有
   "plateau" 这个名字暗示的灵敏。这种组合能通过校验，只在构造配置时记一条 WARNING；
   想要确定的退火曲线，请用 `cosine` / `step`，或者打开验证。

2. **在您的训练脚本中加载它**:

   框架的 `Config` 类提供了 `from_yaml` 方法来从 YAML 文件加载配置. 您可以在自己的训练脚本中调用此方法:

    ```python
    from llm.training.core.config import Config

    # 从 YAML 文件加载配置
    config = Config.from_yaml("path/to/your/config.yaml")

    # CLI 参数会覆盖 YAML 配置 (如上所示)
    # 直接在命令行中使用 --lr, --batch-size 等参数覆盖 YAML 中的值
    ```

   *注意: `llm train` 使用 `--config-path` 参数加载 YAML 配置. 如果您希望在自己的 Python 脚本中加载配置, 可以显式调用 `Config.from_yaml()`.*

### 开关与它的子参数

`training` 里有五个走"总开关 + 子参数"形态的微调/量化选项。子参数**只在总开关为
真时才会被读取**（实现上就是 `if use_xxx:` 分支里读），所以：

| 总开关              | 它的子参数                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_qat`           | `qat_bits` · `qat_quant_activation` · `qat_target_modules`                                                                                                                                              |
| `use_adalora`       | `adalora_init_rank` · `adalora_target_rank` · `adalora_alpha` · `adalora_orth_reg_weight` · `adalora_ema_alpha` · `adalora_tinit` · `adalora_tfinal` · `adalora_prune_every` · `adalora_target_modules` |
| `use_prefix_tuning` | `prefix_tuning_len`                                                                                                                                                                                     |
| `use_ia3`           | `ia3_init_scale` · `ia3_target_modules`                                                                                                                                                                 |
| `use_adapter`       | `adapter_bottleneck_dim` · `adapter_target_modules`                                                                                                                                                     |

```yaml
training:
  use_qat: false      # ← 关着
  qat_bits: 4         # ← 这行不会生效
```

这种写法会通过校验，只在配置构造时记一条 WARNING（形如
`use_qat is off, so the configured option(s) qat_bits have no effect`）。判定依据是
**你有没有显式写这个字段**（Pydantic 的 `model_fields_set`），不是值和默认值比——
默认值本身都合法，拿默认值报警只会制造噪音。YAML 载入的路径同样能识别显式字段。

## 方法三: 使用环境变量 (分布式训练)

在进行多节点分布式训练时, 通常使用环境变量来告知每个进程其角色和如何找到主进程.

```bash
# 节点 0 (主节点)
export MASTER_ADDR='10.1.1.1'
export MASTER_PORT='12355'
export NUM_NODES=2
export NODE_RANK=0
export GPUS_PER_NODE=8
llm train --task ...

# 节点 1
export MASTER_ADDR='10.1.1.1' # 与主节点相同
export MASTER_PORT='12355'  # 与主节点相同
export NUM_NODES=2
export NODE_RANK=1
export GPUS_PER_NODE=8
llm train --task ...
```

`DistributedManager` 会自动从环境变量中读取这些值来初始化 `torch.distributed`.
