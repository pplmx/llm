# 项目故障排查指南

本文件提供了在使用本项目时可能遇到的常见问题及其解决方案。如果您在这里找不到解决方案，请查阅相关文档或提交一个问题。

## 常见问题

- **问题: `make init` 或 `make sync` 失败，或遇到依赖冲突。**
  - **解决方案: **
    1.  确保您的 Python 版本符合 `pyproject.toml` 中 `requires-python` 的要求（当前为 3.13+）。
    2.  尝试清理 `uv` 缓存: `uv clean`。
    3.  检查 `pyproject.toml` 和 `uv.lock` 文件，手动解决潜在的依赖冲突。
    4.  确保您的网络连接正常，可以访问 PyPI。

- **问题: `make` 命令无法执行，提示“command not found”。**
  - **解决方案: ** 确保您的系统安装了 `make` 工具。
    -   **Linux/macOS:** 通常预装。
    -   **Windows:** 可以通过 Chocolatey (`choco install make`) 或 Scoop (`scoop install make`) 安装，或者安装 Git for Windows (它通常包含 `make`)。

- **问题: 训练时遇到 `torch.cuda.is_available()` 返回 `False`，即使有 GPU。**
  - **解决方案: **
    1.  确保您安装了正确版本的 PyTorch，并且它与您的 CUDA 驱动版本兼容。
    2.  检查您的 CUDA 驱动是否已正确安装并更新到最新版本。
    3.  确认您的 GPU 设备已正确识别并启用。
    4.  如果使用 Docker，确保容器以 `--gpus all` 或类似方式运行。

- **问题: 训练过程中出现内存不足 (OOM) 错误。**
  - **解决方案: **
    1.  **减小 `batch_size`**: 这是最直接有效的方法。
    2.  **减小模型大小**: 尝试减小 `hidden_size` 或 `num_layers`。
    3.  **启用自动混合精度 (AMP)**: 在 `config.py` 中设置 `optimization.use_amp = True`，或在命令行中不使用 `--no-amp`。AMP 可以显著减少显存占用。
    4.  **启用 `torch.compile`**: 在 `config.py` 中设置 `optimization.use_compile = True`，或在命令行中不使用 `--no-compile`。虽然主要用于加速，但有时也能帮助优化内存。
    5.  **梯度累积**: 如果您的任务支持，可以通过增大 `gradient_accumulation_steps` 来模拟更大的批次大小，同时保持较小的实际 `batch_size`。

- **问题: 分词器抛出 `KeyError`，提示字符不在词汇表中。**
  - **解决方案: ** 当前的 `SimpleCharacterTokenizer` 是字符级别的，并且词汇表是根据初始化时提供的语料库构建的。确保您尝试编码的文本只包含在初始化分词器时语料库中存在的字符。如果需要处理更广泛的字符集，您可能需要更新分词器或其初始化语料。

## 提交问题

如果您在这里找不到解决方案，请在我们的 GitHub 仓库上提交一个问题: [GitHub Issues](https://github.com/pplmx/llm/issues)。请提供详细的错误信息、复现步骤和您的环境配置。
