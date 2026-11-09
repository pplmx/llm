# Architecture Assessment — 整体架构是否需要重构

**评估日期**: 2026-11-09
**范围**: `src/llm/` 全部 15 个层、222 个模块、47,469 行源码
**方法**: 客观指标（依赖矩阵、循环检测、层规模、耦合计数），非印象判断

## 结论

**不需要「整体重构」。「整体重构」在这里是错的工具。**

代码库不是腐坏到该翻新的结构。分层边界实际守着，注册表抽象统一，扩展点齐备，测试面（80,357 行，1.69× 源码）足以支撑任何局部改动。做一次"整体重构"会拿真实现有的可用性去换一个假想的结构，代价是几周不可回归的改动、大量已经锁定的回归测试失效，而收益是修掉下面列出的**两个类型放错位置**和**一个包初始化环**——这些用三次小改就能解决。

需要做的是**三处定点修复**，不是重构。其中第一项有真实的行为收益（消除反向依赖后，`serving` 和 `runtime` 不再需要 `training` 才能工作）。

## 证据：架构是健康的

### 分层依赖基本单向

```text
from\to        core models runtim   data genera traini servin export evalua quanti compat tokeni multim    cli  utils
core               60      .      3      .      .      .      .      .      .      .      .      .      .      .     12
models              6      .      .      .      .      .      .      .      .      .      .      .      .      .      1
runtime             1      1     11      .      .      2      .      .      .      .      .      3      .      .      .
data                .      .      5     41      .      2      .      .      .      .      .      7      .      .      .
generation          1      4      2      .     13      .      4      .      .      .      .      1      .      .      .
training           13      6     12     16      1     88      .      .      2      1      .      1      2      .      2
serving             8      3      2      .      5      2     56      .      .      .      .      1      .      .      5
export              1      1      3      .      .      1      .     25      .      .      1      .      .      .      .
evaluation          .      .      1      1      .      .      .      .     14      .      .      1      .      .      .
quantization        .      .      .      .      .      .      .      .      .     36      .      .      .      .      1
compat              1      4      1      .      .      .      .      .      .      .      4      .      .      .      1
tokenization        .      .      .      .      .      .      .      .      .      .      .      1      .      .      .
multimodal          .      2      3      1      .      1      .      .      .      .      .      .     22      .      .
cli                 .      .      1      .      .      4      1      .      .      5      .      .      .     12      1
utils               .      1      2      .      1      .      .      .      .      .      .      1      .      .      1
```

`core` 完全不依赖 `models` / `training` / `serving` / `generation` / `export` / `evaluation` / `quantization` —— 最底层是干净的。`models` 只依赖 `core` 和 `utils`。这层设计守住了。

`core → runtime`（3 次）不是违规：`runtime.Registry` 是通用注册表基础设施，`core/registry.py` 和 `core/peft/registry.py` 正确地复用它，而不是各写一份。

`generation → serving`（矩阵里的 4 次）经核对为函数内局部导入，运行时不成环。

### 循环依赖只有 1 个，且是包初始化顺序问题

222 个模块里，运行期循环只有：

- `runtime/__init__.py ↔ runtime/bootstrap.py`：`__init__` 导入 `bootstrap`，`bootstrap` 又 `from llm.runtime import model_factory`，重入包初始化。能工作是因为导入顺序恰好，属于脆弱设计。

（另两个初筛出的"环"是误报：`training` 那个由 `TYPE_CHECKING` 保护、运行期不成环；`serving.cli` 那条是检测脚本把模块名和同名包名混淆了。）

### 注册表抽象统一，没有平行实现

- `Registry[T]` 单一定义，被 10 处复用（attention / MLP / norm / model / peft / backend / export / source / task / modality）
- 5 个 setuptools entry-point 组：`llm.models`、`llm.generation_backends`、`llm.data_sources`、`llm.export_backends`、`llm.peft_methods`
- 数据层继承链清晰：`BaseDataModule` → `MapDataModule`/`StreamDataModule` → `SamplerMapDataModule` → `TokenizedMapDataModule`，9 个具体 DataModule 复用，无平行基类

PEFT 那 8 个方法各带 `apply_*` / `merge_*` / `unmerge_*`（共 48 个函数）看起来像重复，实际是 8 种不同算法（LoRA 的 A/B 乘积 vs Adapter 的 bottleneck residual vs IA³ 的乘性缩放），且已由 `PEFT_REGISTRY` 统一分发。不是重复，是功能差异。

### 仓库在持续偿还技术债

审计文档的 findings 已大面积落地：`Finding G`（`training/core/utils.py` 是 299 行杂物抽屉）现在 **34 行**；`Finding H`（`serving/api.py` 混合多职责）已拆至 **292 行**并分出 routers/middleware/errors；CI 的 ty 类型检查、覆盖率门槛、安全问题、marker 分类都已就位。一个需要整体重构的代码库不会呈现这种趋势。

## 三处该修的（不是重构）

### 1. `ModelConfig` 放错层 —— 唯一有行为收益的一项

**事实**：`ModelConfig` 描述的是模型架构（`hidden_size`、`num_heads`、`attn_impl`、`num_kv_heads`…），却定义在 `training/core/config.py` 里（一个 1,393 行、混装 8 个配置类的文件）。

**后果**：`runtime/model_factory.py` 和 `serving/loader.py` 为了拿这个类型，必须 `from llm.training.core` —— 这两层在**任何路径**上都被迫引入 `training`。`serving` 只是要加载一个模型并起服务，不需要训练栈。

**收益**：把 `ModelConfig` 挪到 `models/config.py`（或 `runtime/config.py`），可消除 5 处跨层反向依赖。

**成本**：10 个源文件提及、52 个测试文件导入它。**这一项不能靠 `git mv` 完成** —— 需要在 `training/core/config.py` 保留向后兼容的重新导出，逐文件迁移，跑全量门禁。估计一次提交的量。

**注意**：`core/registry.py` 和 `core/attn/*` 里出现的 `ModelConfig` 只是注释提及，无实际导入，不受影响。

### 2. `data → training.rlhf` 反向依赖 ✅ 已修复

`data/modules/aifeedback.py` 和 `rejection_sample.py` 从 `training/rlhf/` 导入判分与采样函数。数据层不该依赖训练层。

**方向**：这两个函数（`PreferenceJudge`、`prefer_batch`、`rejection_sample`）本质是**数据处理逻辑**，不是训练逻辑。

**已执行**：`training/rlhf/aifeedback.py` → `data/preference_labeling.py`，`training/rlhf/rejection_sampling.py` → `data/rejection_sampling.py`。`training/rlhf/aligner_benchmark.py`（上层）改为向下引用 data。`data → training` 依赖计数归零，`data` 层现在只依赖 `runtime`（Registry）、`tokenization`、自身。

### 3. `runtime/__init__ ↔ bootstrap` 包初始化环 ✅ 已修复

`bootstrap.py` 里的 `from llm.runtime import model_factory` 改为直接引用子模块（`from llm.runtime.model_factory import MODEL_REGISTRY, build_decoder, build_regression_mlp`），不再经过包 `__init__`，环消失。

## 明确不建议做的

| 想法                                      | 为什么不该做                                                                                                                      |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 拆小 `training/core/config.py`（1393 行） | 8 个 `*Config` 类同属配置职责，拆成 8 个文件只会增加导入噪音。真问题是 `ModelConfig` **内容**放错了层（见上），不是文件长         |
| 拆小 `training/core/engine.py`（1242 行） | 训练循环本身就是内聚的；它已经用 callbacks / distributed / checkpoint 分离了职责。按行数拆只会把一条执行流切成碎块                |
| 引入 DDD / 六边形架构                     | 当前"分层 + 注册表"的设计已经表达了同样的边界（core 纯净、runtime 是插件内核、registry 是端口），换成六边形是术语迁移而非结构改进 |
| 统一 9 个层的注册表                       | 已经统一了（`Registry[T]`）                                                                                                       |
| 重写 PEFT 的 48 个函数                    | 是 8 种算法，不是重复                                                                                                             |

## 执行状态

| 项                    | 状态      | 验证                                                                      |
| --------------------- | --------- | ------------------------------------------------------------------------- |
| #3 包初始化环         | ✅ 已修复 | `MODEL_REGISTRY` 正常注册，57 个 models 测试通过                          |
| #2 `data → training`  | ✅ 已修复 | `grep 'from llm.training' src/llm/data/` 为空；data/training 相关测试通过 |
| #1 `ModelConfig` 上移 | ⬜ 未做   | 10 个源文件 + 52 个测试文件受影响，需单独一个提交                         |

两项修复后跑了完整 CPU 门禁（`GITHUB_ACTIONS=true CI=true CUDA_VISIBLE_DEVICES=""`）：**0 失败**。

### #1 为什么留到下一轮

`ModelConfig` 上移需要：新建 `models/config.py`、在 `training/core/config.py` 保留向后兼容重导出、逐文件迁移 10 处源码 + 52 处测试导入、跑全量门禁。这是一次独立提交的量，且必须一次成型（半途会留下双份定义）。前两项的收益（`data` 层解耦、包环消除）已经拿到。

### 剩下的反向依赖里哪些是合理的

修复后仍有 12 条"下层引用上层"，逐条判定：

| 依赖                            | 判定                                                                       |
| ------------------------------- | -------------------------------------------------------------------------- |
| `core → runtime` (3)            | **合理**。`runtime.Registry` 是通用注册表基础设施，core 复用它而非自写一份 |
| `runtime → models/tokenization` | **合理**。`model_factory`/`tokenizer_factory` 的职责就是构造这些类型       |
| `utils → *` (5)                 | **合理**。`utils` 是横切工具层，被各层共用是它的用途                       |
| `generation → serving` (4)      | **函数内局部导入**，运行期不成环                                           |
| `runtime → training` (2)        | **不合理**，但正是 #1 要修的：`ModelConfig` 放错层                         |
| `export → training` (1)         | 同源于 `ModelConfig`                                                       |
| `multimodal → training` (1)     | 同源于 `ModelConfig`                                                       |

即：**只要修完 #1，剩下 3 条不合理依赖全部消失**，反向依赖从 12 降到 0（按"合理 vs 不合理"划分）。

## 附：本次评估的局限

- 只做了静态结构与依赖分析，没有 profile 运行时性能热点
- 没有评估 `training/core/engine.py` 中"standard loop + custom loop"双模式隐藏契约（审计 `Finding F`），那需要读执行路径而非依赖图
- 没有逐文件审查业务逻辑正确性
