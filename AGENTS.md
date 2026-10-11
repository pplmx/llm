# Agent 指南

本文件是 AI Agent 的**决策入口**：只保留高频路径与约束，细节读链接文档。

## 快速索引

| 文件 | 何时读 |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | 当前优先级 P0–P3、阶段完成度 |
| [docs/reference/architecture.md](docs/reference/architecture.md) | 分层设计、数据流、Registry 机制 |
| [docs/development/guide-extending.md](docs/development/guide-extending.md) | 加 task / callback / scheduler / metric |
| [docs/adr/](docs/adr/) | 已锁定架构决策（GQA、KV cache、Paged Attn、QLoRA） |
| [CHANGELOG.md](CHANGELOG.md) | 近期功能与 breaking changes |
| [pytest.toml](pytest.toml) | pytest markers、config |
| [docs/development/test-architecture.md](docs/development/test-architecture.md) | 测试架构、GPU-less 测试策略、CI 拆分 |
| [Justfile](Justfile) | 所有 `just` 目标定义 |

## 当前焦点 (P0)

预训练 **productization**（流式骨架已完成，见 `stream_lm` task）：

- C4 / Pile / RedPajama **preset**（复用 `SOURCE_REGISTRY` + `HFStreamTextSource`）
- 数据 dedup / 过滤（`TextSource` 装饰器或 pipeline）
- 主路径文档与 e2e（`llm train --task stream_lm`，非仅 `scripts/`）

完整 backlog 见 [ROADMAP.md § 下一步探索方向](ROADMAP.md#下一步探索方向-).

## 源码分层

| 层 | 职责 | 扩展点 |
| --- | --- | --- |
| `core/` | Attention、MLP、Norm、KV cache | `ATTENTION_REGISTRY` / `MLP_REGISTRY` / `NORM_REGISTRY` |
| `models/` | `DecoderModel` | `ModelFactory` · entry point `llm.models` |
| `data/` | Dataset、DataModule、流式 source | `SOURCE_REGISTRY` · entry point `llm.data_sources` |
| `training/` | Engine、Task、DDP/FSDP | `TASK_REGISTRY` · `training/tasks/builtin.py` |
| `generation/` | 采样、推理 backend | `BACKEND_REGISTRY` · entry point `llm.generation_backends` |
| `runtime/` | Registry 基础设施、plugins | `registry.py` / `plugins.py` |
| `serving/` | FastAPI、continuous batching | `ServingConfig` / `batch_engine.py` |
| `evaluation/` | 离线评估 | `EvaluationRunner` · `eval_tasks/` |
| `compat/` | HF 权重加载、safetensors read | `hf_loader.py` |
| `export/` | ONNX 等导出 | `export/onnx.py`（尚无 export registry） |

### 架构边界（违反前需新 ADR）

| 约束 | 说明 |
| --- | --- |
| `attn_impl=mla` | 支持 KV cache (linear + paged)。当前是 placeholder 实现 (learnable latent queries + uniform-mean 输出)；DeepSeek-V2-style latent-compressed K, V 是单独的 follow-up，见 [Tier 3 #31 ticket](docs/audits/2026-07-12-tickets/31-mla-kv-cache.md) |
| Paged Attention | 已全链路支持：[ADR-004](docs/adr/004-paged-attention-serving.md) — prefix cache ✅，model forward 走 `PagedKVCache` |
| 多模态 / 3D 并行 | 无 registry；需先设计 `MultimodalDataModule` 等，勿硬改 `DecoderModel` |
| compat shim | Wave 3 已删；新代码走 `runtime/` / factory，不恢复旧路径 |

## 扩展地图

| 目标 | 主要改动位置 |
| --- | --- |
| 新训练 task | `training/tasks/` + `TASK_REGISTRY.register` in `builtin.py` |
| 新 DataModule | `data/modules/` + 与 task 配对注册 |
| 新流式数据源 | `data/sources.py` + `llm.data_sources` entry point |
| 新注意力/MLP/Norm | `core/` + `@register_*` in `core/registry.py` |
| 新推理 backend | `generation/` + `BACKEND_REGISTRY` + entry point |
| 新评估指标/任务 | `evaluation/metrics/` / `evaluation/eval_tasks/` |
| 新 CLI 能力 | 优先扩 `llm train` / `llm serve`，脚本仅作 demo |

### 内置训练 task

| CLI `--task` | 用途 |
| --- | --- |
| `lm` | Map-style 语言建模 |
| `stream_lm` | 流式大规模预训练 |
| `sft` / `dpo` / `reward` / `ppo` | 对齐流程 |
| `regression` | 合成回归 demo |

```bash
uv run llm train --task stream_lm --help
uv run llm serve
```

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `just init` | 首次：依赖 + pre-commit |
| `just sync` / `just dev` | 默认 groups / 全部 groups |
| `just test` | 全量（所有 marker，含 GPU/heavy/e2e；GPU 主机跑） |
| `just test-fast` | CPU-clean 日间闸门：排除 `heavy`、`e2e`、全部 GPU marker |
| `just test-gpu` | 仅 GPU marker（`gpu`/`multi_gpu`/`need_gpu(n)`/`full_cluster`） |
| `just test-cov` | 全量 + 覆盖率（scheduled 深闸门，`--cov-fail-under=77`） |
| `just test-cov-fast` | CPU 核心套件 + 覆盖率（per-push CI 闸门，`--cov-fail-under=77`） |
| `just test-e2e` | 仅 e2e |
| `just ruff` | format + lint（--fix） |
| `just check` | 静态 lint gate（uv lock --check + ruff --check + ty + examples） |
| `just ty` | 类型检查 |
| `uv sync --extra streaming` | HF `datasets` 流式依赖 |
| `uv lock --check` | 验证 lock 与 pyproject 同步 |

**改动后验证顺序**: `just test` → `just ruff` → `just test`

## uv 约定

- 唯一包管理器：`uv`（不新增 `requirements.txt` / pip workflow）
- `default-groups = ["test"]` → 普通 `uv sync` 含 pytest
- 改依赖后：`uv lock` + 提交 `uv.lock`
- CI / Docker：`uv sync --frozen`；Docker builder 排除 test/docs/streaming group
- **一律用 `uv run`（或 `uv run python`）执行 Python / 脚本**，不用系统裸 `python3`：本仓库 `requires-python = ">=3.14"`，系统 `python3` 常是 3.12，两者语法语义不同。
- **Python 3.14（PEP 758）：`except A, B:`（不带括号）是合法且推荐的写法**。`ruff format` 会把 `except (A, B):` 规范化为不带括号形式，这是正确行为——不要把它当成「损坏的 ruff」或「Python 2 残留语法」去改回去。注意：在 3.12 及更早这是语法错误，所以只用 3.14（`uv run`）验证。

## 测试约定

```
tests/
  support/          # 纯 Python  helper（无 pytest）
  conftest.py       # device, tiny_*, stub_tokenizer, model_and_tokenizer
  models/conftest.py
  data/conftest.py
```

| 场景 | 做法 |
| --- | --- |
| stub tokenizer | `stub_tokenizer` / `line_tokenizer` fixture |
| DecoderModel 参数 | `decoder_model_kwargs(**overrides)` 或 `models/conftest` |
| 弱断言 | 避免纯 `isinstance` / `is not None`；断言具体行为或数值 |
| mock | 默认不用；直接跑真实模块 |

Markers（`pyproject.toml`）：`quick` · `slow` · `heavy` · `e2e` · `gpu` · `multi_gpu`

## 代码与协作规范

- 路径：`pathlib.Path`
- 注释 / 日志：**英文**；与用户交流：**中文**
- 遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- Commit 工作流：`commit_message.txt` → `git commit -F` → 删除
- **Commit 粒度（最小完备）**：
  - 每个 commit 对应一个逻辑上完整、独立的变更（代码 + 其配套测试同仓）
  - 多个**非常小**且各自最小完备的改动，可以合并进**同一个 commit**；不必为每个小改动单独开 commit
  - 不要为了拆而拆：宁可合并同类小改动，也不要产生一堆碎片 commit
- Commit message：**bullet 式**（body 一个要点一行）、条理清晰、**不冗余赘述**；只写人类作者的意图与变更，不写过程废话
- **仅在用户明确要求时 commit**；禁止 `--no-verify`、force push `main`
- **禁止 `Co-Authored-By` AI 代理署名**（如 `Co-Authored-By: Claude <noreply@anthropic.com>`）；commit message 中不要追加 AI 身份行，commit body 应当只描述人类作者的意图与变更
- Bug 修复：评估是否补回归测试（测行为，非测实现细节）

## 反模式（避免）

- 内联 `_Tok` / `_LineTokenizer`（用 `tests/support/tokenizers.py`）
- 在 `DecoderModel` 里为多模态打补丁
- 恢复已删除的 compat shim 或 duplicate registry
- 为覆盖率堆 demo 测试或重复 e2e
- 未跑 `just test` 就声称完成

## Agent skills

### Issue tracker

Issues and PRDs for this repo live as GitHub issues, tracked via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
