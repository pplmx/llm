# AGENTS.md

本文件是仓库对 AI Agent 的**决策入口与规范**：只保留高频路径、硬约束与验证门槛，细节走链接（渐进式披露）。规范语言遵循 **RFC 2119**：**MUST**（必须）· **SHOULD**（应当）· **NEVER / MUST NOT**（禁止）。

- 与用户交流用**中文**；代码注释 / 日志 / commit message / 文档技术内容用**英文**。
- 所有结论 MUST 有实据（代码 / 工具 / 测试 / 文档）；未观测到 = `[INFERENCE]`。
- 不擅自扩大范围，也不静默缩小范围；缺信息先明说，再做能做完的部分。

## 1. 先读（高频索引）

| 主题                                                | 入口                                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------ |
| 当前优先级与阶段完成度                              | [ROADMAP.md](ROADMAP.md)                                                       |
| 分层设计、数据流、Registry                          | [docs/reference/architecture.md](docs/reference/architecture.md)               |
| 加 task / callback / scheduler / metric             | [docs/development/guide-extending.md](docs/development/guide-extending.md)     |
| 已锁定架构决策（GQA、KV cache、Paged Attn、QLoRA…） | [docs/adr/](docs/adr/)                                                         |
| 近期功能与 breaking changes                         | [CHANGELOG.md](CHANGELOG.md)                                                   |
| 测试架构：marker 分类、GPU-less 策略、CI 拆分       | [docs/development/test-architecture.md](docs/development/test-architecture.md) |
| 依赖 group / entry points                           | [pyproject.toml](pyproject.toml)                                               |
| pytest markers / 配置                               | [pytest.toml](pytest.toml)                                                     |
| 全部 `just` 命令                                    | [Justfile](Justfile)                                                           |

## 2. 仓库地图

### 2.1 源码分层（层 | 职责 | 扩展点）

| 层              | 职责                                      | 扩展点                                                                    |
| --------------- | ----------------------------------------- | ------------------------------------------------------------------------- |
| `core/`         | Attention、MLP、Norm、KV cache、PEFT      | `ATTENTION_REGISTRY` / `MLP_REGISTRY` / `NORM_REGISTRY` / `PEFT_REGISTRY` |
| `models/`       | `DecoderModel`                            | `ModelFactory` · entry point `llm.models`                                 |
| `data/`         | Dataset、DataModule、流式 source          | `SOURCE_REGISTRY` · entry point `llm.data_sources`                        |
| `training/`     | Engine、Task、DDP/FSDP                    | `TASK_REGISTRY` · `training/tasks/builtin.py`                             |
| `generation/`   | 采样、推理 backend                        | `BACKEND_REGISTRY` · entry point `llm.generation_backends`                |
| `runtime/`      | Registry 基础设施、plugins、model factory | `registry.py` / `plugins.py`                                              |
| `serving/`      | FastAPI、continuous batching              | `ServingConfig` / `batch_engine.py`                                       |
| `evaluation/`   | 离线评估                                  | `EvaluationRunner` · `eval_tasks/`                                        |
| `quantization/` | GPTQ / AWQ / SmoothQuant / FP8 / 混合精度 | 算法实现（无 registry）                                                   |
| `tokenization/` | tokenizer 与 BPE 训练                     | `tokenizer.py` / `train_bpe.py`                                           |
| `compat/`       | HF 权重加载、safetensors read             | `hf_loader.py`                                                            |
| `export/`       | ONNX / TorchScript / GGUF 导出            | `EXPORT_REGISTRY` · entry point `llm.export_backends`                     |

**Registries via entry points**：`llm.models` · `llm.generation_backends` · `llm.data_sources` · `llm.export_backends` · `llm.peft_methods`。

### 2.2 内置训练 task（`TASK_REGISTRY`）

| CLI `--task`                                        | 用途                                           |
| --------------------------------------------------- | ---------------------------------------------- |
| `lm`                                                | Map-style 语言建模                             |
| `stream_lm`                                         | 流式大规模预训练（C4 / Pile / RedPajama 预设） |
| `sft` / `dpo` / `simpo` / `reward` / `ppo` / `grpo` | 对齐流程                                       |
| `distill`                                           | 知识蒸馏                                       |
| `multimodal`                                        | 多模态（modal-fusion 前缀）                    |
| `regression`                                        | 合成回归 demo                                  |

### 2.3 架构边界（**跨界前 MUST 先落 ADR**）

| 约束             | 说明                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attn_impl=mla`  | 支持 KV cache（linear + paged），当前为 placeholder 实现；latent-compressed K/V 是独立 follow-up（`docs/audits/2026-07-12-tickets/31-mla-kv-cache.md`） |
| Paged Attention  | 已全链路支持 [ADR-004](docs/adr/004-paged-attention-serving.md)                                                                                         |
| 多模态 / 3D 并行 | 无 registry；先设计 `MultimodalDataModule`，勿硬改 `DecoderModel`                                                                                       |
| compat shim      | Wave 3 已删；新代码走 `runtime/` / factory，不恢复旧路径                                                                                                |

## 3. 当前焦点（P0）与 CLI

统一 CLI：`llm <subcommand>`（`train` · `serve` · `quantize` · `prune` · `decompose` · `migrate-ckpt`），`llm --version` / `-V` = `llm 0.0.5`；`python -m llm` 等价。

预训练 **productization**（流式骨架已完成，`stream_lm`）：C4 / Pile / RedPajama **preset**、数据 dedup / 过滤、主路径文档与 e2e。完整 backlog 见 [ROADMAP § 下一步探索方向](ROADMAP.md#下一步探索方向-)。

## 4. 工作流（开发闭环）

1. **Scope**：先读相关 skill（见 §9）；多文件工作先规划。
2. **研究后动手**：读相关段落、复用既有模式；改导出符号前 MUST 用 LSP 查引用；不猜文件名/路径。
3. **Implement**：优先改现有文件；改动**最小完备**（见 §6）。
4. **Verify**：跑 §5 门槛，看真实输出（测试、CLI、UI 各自的验证方式）。
5. **Cleanup**：更新文档 / CHANGELOG，删脚手架与临时脚本。

## 5. 验证门槛（Definition of Done）

**改完 MUST 按序验证**：`just test` → `just ruff` → `just test`（全量 → 格式化/lint → 复跑确认）。

- 逐一跑过的命令才算证明；**没跑 suite 不得声称完成**。
- Bug 修复：复现 → 修根因 → 确认；补回归测试（测行为，非实现）。
- Feature/API：更新受影响的契约测试，用临时脚本证明新行为。
- **NEVER**：删测试 / 跳测试 / 降低断言或阈值 / 注释失败用例来伪造「通过」。无法可靠修复则回滚。
- 失败即根因分析 → 修根因 → 重验，不绕行、不淡化。

### 常用命令（详见 [Justfile](Justfile)，测试均 `uv run --frozen`）

| 命令                                                       | 说明                                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------- |
| `just test`                                                | 全量（所有 marker，含 GPU/heavy/e2e；GPU 主机）            |
| `just test-fast`                                           | CPU-clean 日间闸门：排除 `heavy` / `e2e` 与全部 GPU marker |
| `just test-gpu` / `test-heavy` / `test-e2e` / `test-quick` | 只跑对应 marker                                            |
| `just test-cov-fast`                                       | CPU 核心 + 覆盖率（per-push 闸门，`--cov-fail-under`）     |
| `just test-doc`                                            | doctest（src/llm 内 `///` 示例）                           |
| `just check`                                               | 静态闸门：`uv lock --check` + ruff + ty + examples-compile |
| `just ruff` / `just ty`                                    | format + lint / 类型检查                                   |
| `just docs`                                                | 文档站 mkdocs --strict 构建                                |
| `just build` / `just image`                                | wheel / Docker 镜像                                        |

## 6. Commit 规范（业界约定）

- **Conventional Commits**；**message 用英文**；subject 简洁（`type(scope): summary`）。
- body 用 **bullet 一条条列出**（一个要点一行）、条理清晰、**不冗余赘述**——套话会让信息失去检索价值。
- **Commit 粒度（最小完备）**：
    - 一个 commit = 一个逻辑完整独立的变更（代码 + 配套测试同仓）。
    - **一行改完的改动不单独开 commit**，并入相关小改动，避免碎片化。
    - **不把几千 / 上万行塞进单个 commit**；除非深入评估后确属天生最小完备（机械重构、代码生成、整体迁移），例外需在 message 说明。
    - 不为拆而拆；宁可合并同类小改动。
- 流程：`commit_message.txt` → `git commit -F` → 删除；**仅在用户明确要求时 commit**。
- **NEVER** `--no-verify`、force push `main`、追加 `Co-Authored-By` AI 署名。
- **RIL 图谱簿记批量合并**：graph.json 簿记累积约 10–50 条（灵活指导、非硬上限）后合并提交一次 `chore(ril): <summary>`，不入代码提交。详见 `graph-engineering` skill §LEARN。

## 7. 约定与质量底线

- **uv**：唯一包管理器；一律 `uv run` / `uv run python`（本仓 `requires-python = ">=3.14"`，系统裸 `python3` 常是 3.12，二者语法语义不同）。改依赖后 `uv lock` + 提交 `uv.lock`；CI / Docker 用 `uv sync --frozen`。
- 可选依赖组：`dev` · `test` · `docs` · `security` · `eval` · `logging` · `perf` · `compat` · `dvc` · `streaming`（Docker builder 排除 test/docs/streaming）。
- Python 3.14（PEP 758）：`except A, B:` 不带括号是合法推荐写法，`ruff format` 规范化为该形式——正确行为，勿当成损坏改回。
- 路径用 `pathlib.Path`；注释 / 日志英文。
- **测试**：markers 在 `pytest.toml`（`quick` · `slow` · `heavy` · `e2e` · `integration` · `gpu` · `multi_gpu` · `full_cluster` · `need_gpu(n)`）。fixture `stub_tokenizer` / `line_tokenizer`、`decoder_model_kwargs(**overrides)`；避免纯 `isinstance` 弱断言；默认不 mock，跑真实模块；确定性、隔离、全量安全。
- **测试桩**：内联 tokenizer 一律改用 `tests/support/tokenizers.py`。

## 8. 反模式（NEVER）

- 内联 `_Tok` / `_LineTokenizer`（用 `tests/support/tokenizers.py`）
- 在 `DecoderModel` 里为多模态打补丁
- 恢复已删除的 compat shim 或 duplicate registry
- 为覆盖率堆 demo 测试或重复 e2e
- 未跑 `just test` 就声称完成

## 9. Agent skills 与领域文档

- **长期自主循环**：`graph-engineering`（OBSERVE→…→LEARN，RIL 图谱作为跨 session 记忆；图谱簿记批量提交见其 §LEARN）
- **测试 / CI / 排障**：`run-ci` · `write-test` · `debug-inference` · `diagnose` · `diagnosing-bugs`
- **工程**：`code-review` · `codebase-design` · `modern-python-tooling`（uv/ruff/ty/commitizen，pyproject 单一配置源）· `tdd` · `to-prd` · `to-issues`
- **协作流程**：issue tracker（`gh` CLI）· triage labels · domain 契约，见 `docs/agents/`（domain 详见 `docs/agents/domain.md`）
