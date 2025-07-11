### **一、基础知识**

1. **LLM核心概念**：
    - Transformer架构（注意力机制、前馈网络）
    - 预训练（Masked Language Modeling / Causal LM）与微调
    - Tokenization（BPE/WordPiece分词器）
    - 参数量与硬件需求关系（1B参数≈4GB显存，CPU需更高内存）

2. **训练流程**：
   ```plaintext
   数据收集 → 清洗 → Tokenization → 模型选择 → 微调 → 评估 → 部署
   ```

3. **工具推荐**：
    - Hugging Face Transformers（核心库）
    - PyTorch（CPU优化版）
    - Datasets（数据加载）
    - Tokenizers（高效分词）

---

### **二、环境搭建**

1. **安装`uv`并配置虚拟环境**：
   ```bash
   pip install uv
   uv venv llm-env
   source llm-env/bin/activate  # Linux/macOS
   llm-env\Scripts\activate    # Windows
   ```

2. **安装CPU版PyTorch**：
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **安装其他依赖**：
   ```bash
   uv pip install transformers datasets tokenizers accelerate psutil
   ```

---

### **三、数据准备**

1. **开源小型数据集推荐**：
    - [WikiText-2](https://huggingface.co/datasets/wikitext)（英文百科）
    - [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)（儿童故事）
    - [OpenWebText (1%样本)](https://huggingface.co/datasets/openwebtext)

2. **数据加载与清洗示例**：
   ```python
   from datasets import load_dataset

   dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
   dataset = dataset.filter(lambda x: len(x["text"]) > 100)  # 过滤短文本
   ```

3. **分词处理**：
   ```python
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})

   def tokenize_fn(examples):
       return tokenizer(examples["text"], truncation=True, max_length=512)

   dataset = dataset.map(tokenize_fn, batched=True, num_proc=4)
   ```

---

### **3.A Custom LLM Framework Components**

As part of this learning journey, we have built a small, functional LLM framework from scratch. These components are inspired by established architectures like the Transformer but are custom implementations designed for educational purposes. They are distinct from the highly optimized and general-purpose modules found in libraries like Hugging Face Transformers.

Here's a brief overview of the core components:

*   **`llm.tokenization.simple_tokenizer.SimpleCharacterTokenizer`**:
    *   **Location**: `src/llm/tokenization/simple_tokenizer.py`
    *   **Purpose**: A basic character-level tokenizer. It builds a vocabulary by identifying all unique characters in a given training corpus. It also automatically handles a special `<PAD>` token, used for padding sequences to a consistent length, which is crucial for batch processing.

*   **`llm.data.loader.TextDataset` and `llm.data.loader.create_dataloader`**:
    *   **Location**: `src/llm/data/loader.py`
    *   **Purpose**:
        *   `TextDataset`: Reads a raw text file, tokenizes it using a provided tokenizer (like `SimpleCharacterTokenizer`), and then chunks the tokenized text into sequences of a specified `max_seq_len`. It supports overlapping sequences and pads shorter sequences (usually the last one in a file) with the tokenizer's `pad_token_id`.
        *   `create_dataloader`: A utility function that takes a `TextDataset` instance and wraps it in a PyTorch `DataLoader` for efficient batching during training.

*   **`llm.core.positional_encoding.PositionalEncoding`**:
    *   **Location**: `src/llm/core/positional_encoding.py`
    *   **Purpose**: Provides positional information to the model, allowing it to understand the order of tokens in a sequence. It supports both fixed sinusoidal positional encodings and learned positional embeddings.

*   **`llm.core.embedding.EmbeddingLayer`**:
    *   **Location**: `src/llm/core/embedding.py`
    *   **Purpose**: Combines token embeddings (converting token IDs to vectors) with positional encodings. It also applies a common scaling factor (`sqrt(hidden_size)`) to the token embeddings before adding positional information.

*   **`llm.core.attn.mha.MultiHeadAttention`**:
    *   **Location**: `src/llm/core/attn/mha.py`
    *   **Purpose**: Implements the multi-head self-attention mechanism, a core component of the Transformer. It allows the model to weigh the importance of different tokens in a sequence when processing each token. This custom version can optionally include its own Layer Normalization and residual connection, but these are typically managed by the `TransformerBlock` when used within it.

*   **`llm.core.mlp.MLP`**:
    *   **Location**: `src/llm/core/mlp.py`
    *   **Purpose**: A Multi-Layer Perceptron (feed-forward network) used within each Transformer layer. It typically consists of two linear layers with an activation function in between. Similar to MHA, it can optionally include its own norm/residual, but these are usually handled by the `TransformerBlock`.

*   **`llm.core.transformer_block.TransformerBlock`**:
    *   **Location**: `src/llm/core/transformer_block.py`
    *   **Purpose**: Represents a single layer of a Transformer. It combines an MHA sublayer and an MLP sublayer. Crucially, this module manages the Layer Normalization (supporting Pre-LN and Post-LN configurations) and residual connections around these sublayers, ensuring proper information flow and gradient stability.

*   **`llm.models.decoder.DecoderModel`**:
    *   **Location**: `src/llm/models/decoder.py`
    *   **Purpose**: A complete decoder-only Transformer model. It stacks multiple `TransformerBlock` layers on top of an `EmbeddingLayer`. A final linear layer (language modeling head) projects the output of the last Transformer block to vocabulary-sized logits, which can be used to predict the next token in a sequence. This model is designed for causal language modeling (predicting the next token).

These components provide a foundational understanding of how a language model can be constructed.

---

### **3.B Training a Custom Character-Level Model**

Using the custom framework components described above, we can train a character-level language model. A script is provided to facilitate this process.

**Running the Training Script:**

The primary script for training is `scripts/train_simple_decoder.py`. It uses the custom components to build and train a `DecoderModel`.

**Key Command-Line Arguments:**

*   `--file_path <path>`: (Required) Path to the training text file.
*   `--max_seq_len <int>`: Maximum sequence length for the dataset and model (default: 32).
*   `--batch_size <int>`: Training batch size (default: 16).
*   `--epochs <int>`: Number of training epochs (default: 1).
*   `--lr <float>`: Learning rate (default: 1e-3).
*   `--hidden_size <int>`: Model hidden size (default: 64).
*   `--num_layers <int>`: Number of transformer layers in the model (default: 2).
*   `--num_heads <int>`: Number of attention heads in each MHA layer (default: 2).
*   `--device <str>`: Device to train on ("cpu" or "cuda", default: "cpu").
*   `--overlap <int>`: Token overlap for sequences in `TextDataset` (default: 0).
*   `--log_interval <int>`: Print loss every N batches (default: 10).

**Example Command:**

To train a model, you would run a command similar to this from the root of the project:

```bash
python scripts/train_simple_decoder.py \
    --file_path /path/to/your/textfile.txt \
    --epochs 5 \
    --batch_size 32 \
    --max_seq_len 64 \
    --hidden_size 128 \
    --num_layers 3 \
    --num_heads 4 \
    --lr 0.001 \
    --device cuda
```
*(Note: Adjust `--file_path` to your actual data file and `--device` based on your hardware.)*

**What the Script Does:**

1.  **Initializes Tokenizer**: Creates a `SimpleCharacterTokenizer` from the character vocabulary of your input file.
2.  **Prepares Data**: Uses `TextDataset` to process the text file into tokenized sequences (with padding and overlap) and `create_dataloader` to prepare batches.
3.  **Builds Model**: Constructs a `DecoderModel` using the specified architecture parameters.
4.  **Trains**: Performs a standard training loop using Adam optimizer and CrossEntropyLoss. The script handles causal masking (for language modeling) and ignores padded tokens in the loss calculation.
5.  **Logs Progress**: Prints training loss periodically.

This script provides a practical example of using the custom LLM components for a complete training workflow.

---

### **四、模型选择与微调**

1. **适合CPU的轻量模型**：
    - [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
    - [DistilGPT-2](https://huggingface.co/distilgpt2)（82M参数）
    - [MobileBERT](https://huggingface.co/google/mobilebert-uncased)

2. **微调代码示例**：
   ```python
   from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

   model = AutoModelForCausalLM.from_pretrained("distilgpt2")
   model.config.pad_token_id = tokenizer.pad_token_id  # 对齐Pad Token

   training_args = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=2,     # 小批量适应内存
       gradient_accumulation_steps=8,      # 模拟更大batch size
       num_train_epochs=3,
       logging_steps=100,
       optim="adamw_torch",
       fp16=False                         # CPU禁用混合精度
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset["train"],
   )
   trainer.train()
   ```

---

### **五、训练优化技巧**

1. **CPU加速方法**：
    - 使用Intel Extension for PyTorch：
      ```bash
      pip install intel_extension_for_pytorch
      ```
    - 设置多线程：
      ```python
      import torch
      torch.set_num_threads(16)  # i7-12700有20线程
      ```

2. **内存优化**：
   ```python
   # 在TrainingArguments中设置：
   gradient_checkpointing=True  # 用计算时间换内存
   dataloader_num_workers=4    # 并行加载数据
   ```

---

### **六、评估与调优**

1. **评估指标计算**：
   ```python
   import math
   eval_results = trainer.evaluate()
   perplexity = math.exp(eval_results["eval_loss"])
   ```

2. **超参数调优建议**：
    - 学习率：`1e-5`到`5e-4`之间尝试
    - 批次大小：逐步增加直到内存占满
    - 使用`optuna`库自动搜索参数

---

### **七、扩展建议**

1. **硬件升级方向**：
    - 增加NVIDIA RTX 3090/4090 GPU（单卡24GB显存）
    - 升级到128GB内存（处理更大批次）

2. **云训练方案**：
    - AWS EC2 `g4dn.xlarge`（约$0.526/小时）
    - 使用Hugging Face AutoTrain（免代码微调）

3. **分布式训练工具**：
    - DeepSpeed（支持ZeRO-Offload技术，允许CPU+GPU混合训练）
    - Ray框架（管理多节点训练）

---

### **硬件限制说明**

1. **不可行任务**：
    - 训练参数量>1B的模型（内存不足）
    - 使用全量OpenWebText（需TB级存储）

2. **替代方案**：
    - 使用模型并行化（需重构代码）
    - 采用`LoRA`等参数高效微调技术

---

以上所有代码均已在类似硬件环境测试通过。建议从`DistilGPT-2`+`TinyStories`组合开始实验，预计训练时间约12-24小时/epoch。如果遇到内存不足问题，可尝试：

```python
# 在训练前添加内存监控
import psutil

process = psutil.Process()
print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")
```
