# Gemini 工作区

此文件为 Gemini Agent 提供项目上下文信息.

## 项目概览

本项目是一个名为 "llm" 的 Python 应用程序, 旨在构建一个现代、高效的 Decoder-only Transformer 架构.
使用 `uv` 进行包管理, 项目配置详见 `pyproject.toml`.

### 核心架构特性

- **Decoder-only Transformer**: 基础架构.
- **SwiGLU 激活函数**: 提升 MLP 层的表现.
- **Grouped Query Attention (GQA)**: 兼顾性能与显存效率.
- **统一 QKV 投影**: 优化内存布局与计算吞吐.
- **混合精度支持**: 包含自动 BF16/FP16 检测.
- **BPE 分词器**: 高效的子词词元化.

## 工具链

- **Linting 和格式化**: `ruff` 用于 linting 和格式化. 配置位于 `ruff.toml`. 遵循 `pathlib` 最佳实践.
- **类型检查**: `mypy` 用于静态类型检查. 配置位于 `pyproject.toml`.
- **测试**: `pytest` 用于运行测试. 配置位于 `pyproject.toml`.
- **包管理**: `uv` 用于管理依赖.
- **构建**: 详见 `pyproject.toml` 的 build-system 配置.

运行 `make help` 查看所有可用的 Makefile 命令.

## 开发规范

### 代码风格与质量

- **类型注解**: 所有函数签名必须包含完整的类型注解
- **命名规范**:
    - 变量/函数使用 `snake_case`
    - 类使用 `PascalCase`
    - 常量使用 `UPPER_CASE`
    - 私有成员使用前导下划线 `_private`
- **导入顺序**: 标准库 → 第三方库 → 本地模块, 各组之间空一行
- **现代化优先**:
    - 使用 `pathlib` 替代 `os.path`
    - 优先使用 f-string 和新式语法
    - 避免冗余注释, 代码应自解释
- **标点符号**: 代码注释和文档中使用英文/半角标点符号 (如 `,` `(` `)`), 避免全角标点
- **函数长度**: 单个函数建议不超过 50 行, 超过则考虑拆分
- **复杂度控制**: 避免嵌套超过 3 层的逻辑

### 错误处理

- 使用具体的异常类型而非裸 `except`
- 自定义异常应继承自内置异常并放在 `exceptions.py`
- 关键路径必须有错误处理 (文件 I/O, 网络请求, 外部调用)
- 错误信息应包含上下文 (例如: 失败的文件路径, 参数值)

### 性能与优化

- 避免在循环内重复创建对象
- 大规模数据处理优先使用 Generator
- 使用 `__slots__` 优化高频实例化的类
- 关键路径考虑添加性能基准测试 (`tests/benchmarks/`)
- 架构一致性: 任何对 MHA 或 MLP 的修改必须考虑 GQA 和 SwiGLU 的兼容性

### 测试哲学

- **Functional over Mock**: 优先使用轻量级真实对象 (`tiny_model`, `tiny_config`) 进行测试, 避免过度 Mock 内部逻辑. 直接调用代码验证功能性.
- **Infrastructure**: 使用 `tests/conftest.py` 集中管理共享 Fixtures. 避免在每个测试文件中重复 setup.
- **E2E Focus**: 关键训练流程 (Gradient Accumulation, Resume, Checkpointing) 必须有 E2E 测试覆盖 (`tests/e2e/`).
- **Refactoring**: 发现由于拷贝粘贴导致的重复测试逻辑时, 必须积极重构 (如提取 `tests/dummies.py`).
- **错误驱动测试**: 遇到 bug 或边界情况时, 考虑是否需要添加测试覆盖 (Regression Test).

### 文档规范

- **Docstring 风格**: 使用 Google 风格
- **必须文档化**:
    - 所有 public API
    - 非显而易见的算法实现
    - 复杂的配置项
- **避免冗余文档**:
    - 不要重复参数名 (`Args: x: x parameter` ❌)
    - 代码自解释时不强制要求注释
- **Comments policy**: 仅在必要时编写高价值注释, 避免通过注释与用户对话

### 依赖管理

- 新增依赖必须在 Commit Message 中说明理由
- 优先使用标准库
- 避免引入重量级依赖解决小问题
- 定期审查 `pyproject.toml` 清理未使用依赖
- 主要依赖项详见 `pyproject.toml`

### 安全基线

- 不提交敏感信息 (API keys, 密码, 私钥)
- 使用环境变量或配置文件管理敏感配置
- 外部输入必须验证 (文件路径, 用户配置)

## Git 工作流

### 提交信息规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 规范. 使用以下类型:

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 仅文档更改
- `style`: 不影响代码含义的更改 (例如: 空白、格式、缺少分号等)
- `refactor`: 既不修复 Bug 也不添加功能的代码更改
- `perf`: 提高性能的代码更改
- `test`: 添加缺失的测试或更正现有测试
- `build`: 影响构建系统或外部依赖的更改 (示例范围: pip, docker, npm)
- `ci`: 更改 CI 配置文件和脚本
- `chore`: 不修改 `src` 或 `test` 文件的其他更改
- `revert`: 撤销之前的提交

**注意**: `description` (主题行) 通常应以小写字母开头.

### 提交工作流

为确保提交信息的一致性和正确格式, 请遵循以下工作流:

1. **创建临时文件**: 将完整的提交信息 (包括主题和正文) 写入项目根目录下的临时文件 `commit_message.txt`.
2. **从文件提交**: 使用命令 `git commit -F commit_message.txt` 创建提交.
3. **清理**: 成功创建提交后, 删除 `commit_message.txt` 文件.

### Git 最佳实践

- **分支命名**: `<type>/<short-description>` (如 `feat/gqa-support`)
- **Commit 粒度**: 保持小而聚焦的 commit, 除非功能无法合理拆分. 每个 commit 应该是可独立运行的状态.
- **避免**:
    - 提交调试代码 (`print`, `breakpoint()`)
    - 提交 IDE 配置文件 (已在 `.gitignore`)
    - 混合不相关的修改在单个 commit

### 验证流程

每次 Plan 实施结束后, 必须执行以下验证流程:

```bash
make test  # 或 uv run pytest
make ruff
make test  # 再次确认
```

## 文档结构规范

项目的文档组织在 `docs/` 目录下, 遵循标准的 GitHub 项目实践. 关键规范包括:

- **根目录的 `CONTRIBUTING.md`**: 提供高层次的贡献指南.
- **`docs/` 目录作为主要文档中心**:
    - **根目录的 `README.md` 是所有文档的主要入口**, 它直接链接到 `docs/` 目录下的各个顶级文档文件.
    - 大多数文档文件名使用小写, **多单词文件名使用连字符 `-` 分隔** (例如 `development.md`, `tutorial-cpu-llm.md`, `guide-extending.md`), `README.md` 是例外.
    - 相关文档组织在特定子目录中 (例如 `docs/training/` 用于所有训练框架文档).
    - 子目录可以包含自己的 `README.md` 作为入口.
    - 项目级别的 `docs/troubleshooting.md` 用于通用问题排查.

## 用户偏好

- 用户偏好我使用中文进行回复.
- 用户极其看重代码美感和现代化.

---

## 开发路线图

项目的详细开发路线图请参见 [ROADMAP.md](ROADMAP.md).
