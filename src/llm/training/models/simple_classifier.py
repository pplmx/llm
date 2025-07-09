import torch.nn as nn

from llm.training.core.config import ModelConfig


class SimpleClassifier(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        # Use the same MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.classifier_head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits

    def count_parameters(self):  # Helper
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
