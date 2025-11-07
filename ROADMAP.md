# LLM 项目完整生命周期路线图

这是一个动态更新的路线图，用于追踪整个项目的进展。

### 阶段一：基础建设 (Foundations & DevOps)

- [x] **核心依赖与环境**: 使用 `uv` 管理 Python 依赖。
- [x] **构建与打包**: 使用 `hatchling` (`pyproject.toml`)。
- [x] **代码质量**: `ruff` (格式化), `mypy` (类型检查), `.editorconfig`。
- [x] **测试体系**: `pytest` (单元/集成测试), `pytest-cov` (覆盖率)。
- [x] **自动化 (CI/CD)**: GitHub Actions (`.github/workflows`) 自动执行测试、构建和发布流程。
- [ ] **环境管理**:
    - [x] `Dockerfile` 和 `compose.yml` 已存在。
    - [ ] 完善 Docker 环境，确保开发、测试和生产环境的一致性。

### 阶段二：数据工程 (Data Engineering)

- [x] **数据加载**: `src/llm/data/loader.py` 用于读取和初步处理数据。
- [x] **数据抽象**: `DataModule` (`data_module.py`) 封装数据集和数据加载器。
- [ ] **数据预处理**: 增强预处理流程，支持更大规模的数据集和流式处理。
- [ ] **数据版本控制**: 引入 `DVC` 或类似工具，对数据集和预处理脚本进行版本管理，确保实验可复现。

### 阶段三：模型架构 (Model Architecture)

- [x] **Transformer 基础**: `Embedding`, `PositionalEncoding`, `LayerNorm`, `MLP` 等核心模块。
- [x] **注意力机制**: `DotProductAttn`, `MHA` (多头), `MLA` (多查询)。
- [x] **完整模型**: `DecoderModel` (`decoder.py`) 组装成一个完整的解码器架构。
- [ ] **高级架构探索**:
    - [x] `MoE` (专家混合) 模块已实现。
    - [ ] 将 `MoE` 集成到主模型中，并进行对比实验。
    - [ ] 研究并实现其他高级结构，如 `GQA` (分组查询注意力), `SwiGLU` 等。

### 阶段四：分词 (Tokenization)

- [x] **原型分词器**: `SimpleCharacterTokenizer` 已实现，用于快速验证。
- [ ] **高级分词器**: 替换为生产级的 Subword 分词器。
    - [ ] **步骤 1**: 集成 Hugging Face `tokenizers` 库。
    - [ ] **步骤 2**: 提供脚本，用于在自定义语料上训练 BPE 或 WordPiece 分词器。
    - [ ] **步骤 3**: 将框架中的所有部分（数据、模型、推理）更新为使用新分词器。

### 阶段五：模型训练 (Model Training)

- [x] **训练框架**: `training/core/engine.py` 提供了灵活的训练和验证循环。
- [x] **配置驱动**: `config.py` 支持通过配置来定义和调整实验。
- [x] **任务抽象**: `BaseTask` 允许轻松定义新的训练目标。
- [ ] **大规模训练**:
    - [ ] **分布式训练**: 实现 `DDP` (数据并行) 或 `FSDP` (完全分片数据并行) 以支持多机多卡训练。
    - [ ] **混合精度**: 实现 `torch.cuda.amp` 以加速训练并减少显存占用。
- [ ] **训练策略**:
    - [ ] **预训练 (Pre-training)**: 设计并执行在大型通用语料库上的预训练流程。
    - [ ] **指令微调 (Instruction Fine-tuning)**: 设计并执行在特定指令数据集上的微调流程。

### 阶段六：模型评估 (Model Evaluation)

- [ ] **评估指标**:
    - [ ] **语言模型指标**: 实现 `Perplexity` (困惑度) 的计算。
    - [ ] **下游任务指标**: 根据微调任务，实现相应的评估指标 (如准确率、F1 分数等)。
- [ ] **评估框架**: 建立一个标准化的评估流程，可以在多个数据集和基准上自动运行评估。
- [ ] **人工评估**: 设计人工评估流程，用于评估模型的生成质量、相关性和无害性。

### 阶段七：推理与部署 (Inference & Deployment) - **当前重点**

- [ ] **推理API**:
    - [x] **基础实现**: `inference.py` 已创建。
    - [ ] **核心功能**: 实现 `Greedy Search`, `KV Cache`, `Top-k/Top-p` 采样。
- [ ] **模型服务化**:
    - [ ] 使用 `FastAPI` 或 `Flask` 将推理功能封装成一个 REST API 服务。
    - [ ] 支持流式输出，以实现类似打字机的效果。
- [ ] **性能优化**:
    - [ ] **模型量化**: 研究并应用 `int8` 或 `GPTQ` 等量化技术以压缩模型并加速推理。
    - [ ] **编译优化**: 使用 `torch.compile` 或 `ONNX Runtime` 优化推理性能。

### 阶段八：测试与质量保证 (Testing & QA)

- [x] **单元测试**: 已覆盖大部分核心模块。
- [ ] **集成测试**:
    - [ ] 完善 `MoE` 等复杂模块的集成测试。
    - [ ] 为推理流程 (`inference.py`) 编写完整的测试用例。
- [ ] **端到端 (E2E) 测试**: 创建一个脚本，自动完成“训练-评估-推理”的全流程，作为最终的冒烟测试。

### 阶段九：文档与社区 (Documentation & Community)

- [x] **基础文档**: `README`, `CONTRIBUTING` 等已存在。
- [ ] **开发者文档**: 完善代码内文档 (docstrings)，并使用 `Sphinx` 或 `MkDocs` 生成 API 文档网站。
- [ ] **用户文档**: 编写详细的用户指南和教程，解释如何使用框架进行训练、评估和推理。
- [ ] **示例**: 提供 `Jupyter Notebook` 示例，展示框架的核心功能。
