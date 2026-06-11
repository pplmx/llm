"""
Value Model for PPO critic.

Predicts per-token value estimates from hidden states for GAE and value loss.
"""

import torch
import torch.nn as nn

from llm.models.decoder import DecoderModel


class ValueModel(nn.Module):
    """
    Critic model = Base LM + per-position value head.

    Returns a scalar value estimate at each token position.
    """

    def __init__(self, base_model: DecoderModel):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.hidden_size
        self.value_head = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute value estimates for each token position.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1 for valid, 0 for padding

        Returns:
            values: [batch_size, seq_len]
        """
        hidden_states = self._get_hidden_states(input_ids, attention_mask)
        return self.value_head(hidden_states).squeeze(-1)

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract hidden states from base model."""
        x = self.base_model.embedding_layer(input_ids)

        for block in self.base_model.transformer_blocks:
            x = block(x, attn_mask=attention_mask)

        if self.base_model.final_norm is not None:
            x = self.base_model.final_norm(x)
        return x
