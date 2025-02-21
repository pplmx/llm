import torch
from transformers import pipeline

print("PyTorch version:", torch.__version__)
model = pipeline("text-generation", model="distilgpt2")
print(model("Hello, how are you?"))
