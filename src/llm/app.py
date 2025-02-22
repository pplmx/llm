"""
CPU友好版SFT教学代码 (v5.0)
适配条件：Intel i7-12700 + 64GB内存
训练耗时预估：约2小时/epoch (1000条数据)
"""

import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 🖥️ 硬件配置
DEVICE = "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
os.environ["OMP_NUM_THREADS"] = "12"  # 根据CPU核心数调整

# 🧩 模型选择 (TinyLlama中文裁剪版)
MODEL_NAME = "uer/gpt2-distil-chinese-cluecorpussmall"  # 仅280MB

# 📦 初始化组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=TORCH_DTYPE)

# 🛠️ 处理特殊token (重要！)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# 🎛️ LoRA配置 (减少90%训练参数)
lora_config = LoraConfig(
    r=8,  # 低秩维度
    lora_alpha=16,
    target_modules=["c_attn"],  # 仅微调关键层
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    fan_in_fan_out=True,
)
model = get_peft_model(model, lora_config)
model.config.problem_type = "text-generation"
model.print_trainable_parameters()  # 显示可训练参数

# 📚 创建教学数据集 (小样本演示)
train_texts = [
    "问题：1+1等于几？回答：1+1等于2。",
    "问题：中国的首都是哪里？回答：北京。",
    "问题：谁发明了电话？回答：亚历山大·贝尔。",
] * 50  # 共150条数据


# ✂️ 数据预处理
def tokenize_fn(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")


dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 🎯 训练参数优化 (CPU专用)
training_args = TrainingArguments(
    output_dir="./cpu_sft",
    per_device_train_batch_size=2,  # 根据内存调整
    gradient_accumulation_steps=8,  # 模拟batch_size=16
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    optim="adamw_torch",
    report_to="none",
    use_cpu=True,
    use_mps_device=False,
)

# 🧠 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言模型
)

# 🚀 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("开始训练...")
trainer.train()

# 🧪 测试推理
test_prompts = ["问题：1+1等于几？回答：", "问题：北京有哪些著名景点？回答："]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, max_new_tokens=50, do_sample=True, temperature=0.9, pad_token_id=tokenizer.pad_token_id
    )
    print(f"输入：{prompt}\n输出：{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
