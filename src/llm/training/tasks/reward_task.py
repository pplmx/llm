"""
Reward Model Task for RLHF.

Trains a reward model that scores responses given prompts.
Uses Bradley-Terry loss for preference comparison.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.models.decoder import DecoderModel
from llm.training.tasks.base_task import TrainingTask


class RewardModel(nn.Module):
    """
    Reward Model = Base LM + Value Head.

    Maps the last token's hidden state to a scalar reward.
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
        Compute reward for each sequence in the batch.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1 for valid, 0 for padding

        Returns:
            rewards: [batch_size] scalar reward per sequence
        """
        # Get hidden states from base model
        # We need to access the hidden states, not logits
        hidden_states = self._get_hidden_states(input_ids, attention_mask)

        # Get the last non-padding token's hidden state
        if attention_mask is not None:
            # Find the index of the last valid token for each sequence
            seq_lengths = attention_mask.sum(dim=1).long() - 1  # [batch_size]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]  # [batch_size, hidden_size]
        else:
            # No mask, use the last position
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Pass through value head
        rewards = self.value_head(last_hidden).squeeze(-1)  # [batch_size]
        return rewards

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract hidden states from base model."""
        # Access the transformer blocks and get pre-lm_head hidden states
        x = self.base_model.embedding_layer(input_ids)

        for block in self.base_model.transformer_blocks:
            x = block(x, attn_mask=attention_mask)

        if self.base_model.final_norm is not None:
            x = self.base_model.final_norm(x)
        return x


class RewardTask(TrainingTask):
    """
    Task for training a Reward Model.

    Uses Bradley-Terry loss: L = -log(sigmoid(r_chosen - r_rejected))
    """

    def __init__(self, config: Any, data_module: Any):
        super().__init__(config, data_module)

    def build_model(self) -> nn.Module:
        """Build the RewardModel."""
        from llm.models.decoder import DecoderModel

        base_model = DecoderModel(
            vocab_size=self.config.model.vocab_size,
            hidden_size=self.config.model.hidden_size,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            max_seq_len=self.config.data.max_seq_len,
            dropout=getattr(self.config.model, "dropout", 0.1),
            num_kv_heads=getattr(self.config.model, "num_kv_heads", None),
            attn_impl=getattr(self.config.model, "attn_impl", "mha"),
            mlp_impl=getattr(self.config.model, "mlp_impl", "standard"),
            mlp_ratio=getattr(self.config.model, "mlp_ratio", 4.0),
        )

        return RewardModel(base_model)

    def build_criterion(self) -> nn.Module:
        """Criterion is embedded in train_step, return dummy."""
        return nn.Identity()

    def train_step(
        self,
        batch: dict[str, Any],
        model: nn.Module,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """
        Training step for Reward Model.

        Args:
            batch: Contains chosen_input_ids, chosen_attention_mask,
                   rejected_input_ids, rejected_attention_mask
            model: RewardModel
            criterion: Unused (loss computed inline)

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        # Compute rewards
        chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # Metrics
        reward_diff = chosen_rewards - rejected_rewards
        accuracy = (reward_diff > 0).float().mean()

        metrics = {
            "loss": loss.item(),
            "reward_chosen": chosen_rewards.mean().item(),
            "reward_rejected": rejected_rewards.mean().item(),
            "reward_margin": reward_diff.mean().item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics

    def validation_step(
        self,
        batch: dict[str, Any],
        model: nn.Module,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, dict]:
        """Validation step."""
        loss, metrics = self.train_step(batch, model, criterion)
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        return loss, val_metrics
