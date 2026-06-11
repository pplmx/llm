# Agent 指南

参与本项目的 AI Agent 快速上下文入口。细节以链接文档为准，此处只保留高频决策点。

## 快速索引

| 文件 | 用途 |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | 项目状态、优先级、已知架构边界 |
| [docs/reference/architecture.md](docs/reference/architecture.md) | 分层架构、Registry、扩展点 |
| [docs/development/guide-extending.md](docs/development/guide-extending.md) | 添加 task / callback / scheduler 食谱 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |
| [pyproject.toml](pyproject.toml) | 依赖、pytest markers、entry points |

## 源码分层 (`src/llm/`)

| 层 | 职责 | 扩展方式 |
| --- | --- | --- |
| `core/` | Attention、MLP、Norm、KV cache 等纯模块 | `ATTENTION_REGISTRY` / `MLP_REGISTRY` / `NORM_REGISTRY` |
| `models/` | `DecoderModel` 组装 | `ModelFactory` + `llm.models` entry point |
| `data/` | Dataset、DataModule、流式 source | `SOURCE_REGISTRY` + `StreamDataModule` |
| `training/` | Engine、Task、DDP/FSDP | `TaskRegistry` + `llm.training.tasks` |
| `generation/` | 采样、eager/batched 推理 | `BACKEND_REGISTRY` |
| `runtime/` | Registry、bootstrap、plugins | setuptools entry points |
| `serving/` | FastAPI、continuous batching | `ServingConfig` |
| `evaluation/` | 离线评估 runner | `BaseTask` / `BaseMetric` |

**架构边界**（改代码前确认）:

- `attn_impl=mla` 不支持 KV cache
- Paged Attention 为 partial 实现（prefix cache ✅，forward 全链路待接，见 [ADR-004](docs/adr/004-paged-attention-serving.md)）
- 多模态 / 3D 并行尚无 registry，勿硬塞进 `DecoderModel`

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `make init` | 首次：`uv sync` + pre-commit hooks |
| `make sync` | 同步依赖（含默认 test group） |
| `make dev` | 全部 dependency groups（streaming、docs 等） |
| `make test` | 全量测试（608+） |
| `make test-fast` | 排除 heavy / e2e |
| `make ruff` | format + lint |
| `uv sync --group streaming` | HF 流式预训练额外依赖 |
| `uv lock --check` | CI 用：验证 lock 与 pyproject 一致 |

## 环境与依赖 (uv)

- **包管理**: 只用 `uv`，不要引入 pip/requirements.txt
- **默认 group**: `[tool.uv] default-groups = ["test"]`，`uv sync` 后可直接 `make test`
- **可选 groups**: `streaming`（datasets）、`docs`（mkdocs）
- **锁文件**: 改 `pyproject.toml` 后运行 `uv lock`；CI/Docker 用 `uv sync --frozen`
- **Docker**: builder 仅装运行时依赖；validator stage 再 sync test group 跑测试

## 代码质量

- 路径处理用 `pathlib.Path`
- 改动后验证: `make test` → `make ruff` → `make test`
- 注释与日志用**英文**；与用户交流用**中文**
- 测试优先调真实代码，避免无意义 mock
- 测试数目标: 保持全绿，不堆弱断言（`is not None`、纯 `isinstance`）

## 测试架构 (`tests/`)

```
tests/
  conftest.py          # device, tiny_config, stub_tokenizer, model_and_tokenizer
  support/             # 纯 Python 工具（无 pytest 依赖）
    tokenizers.py      # StubTokenizer, LineTokenizer, CharBoundTokenizer
    corpus.py          # 共享语料
    data.py            # DummyLMDataModule
    models.py          # decoder_model_kwargs()
  models/conftest.py   # DecoderModel 参数化 fixture
  data/conftest.py     # sample_text_tokenizer, line_tokenizer
```

- stub tokenizer → 用 `stub_tokenizer` / `line_tokenizer` fixture，勿内联 `_Tok`
- 构造 DecoderModel → `decoder_model_kwargs(**overrides)` 或 `models/conftest` fixture
- pytest markers: `quick` / `slow` / `heavy` / `e2e` / `gpu`（见 `pyproject.toml`）

## 提交规范

- [Conventional Commits](https://www.conventionalcommits.org/)
- 工作流: 写 `commit_message.txt` → `git commit -F commit_message.txt` → 删除
- 小而聚焦；**仅在用户明确要求时 commit**
- 不要 `--no-verify`、不要 force push main

## 用户偏好

- 使用**中文**交流
- 代码注释使用**英文/半角标点**
- 不使用 mock，直接调用真实代码测试
- 遇到 bug 时考虑是否需要添加测试覆盖
