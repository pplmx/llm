import torch.nn as nn

from llm.training.core.config import ModelConfig


class SimpleMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers = []
        input_size = config.hidden_size
        for i in range(config.num_layers):
            is_last_layer = i == config.num_layers - 1
            output_size = config.hidden_size if is_last_layer else config.ffn_hidden_size
            layers.append(nn.Linear(input_size, output_size))
            if not is_last_layer:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
            input_size = output_size
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
