"""Reward Model DataModule for RLHF."""

from torch.utils.data import random_split

from llm.data.datasets.reward import RewardDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class RewardDataModule(TokenizedMapDataModule):
    """DataModule for Reward Model training with DDP-compatible loaders."""

    def setup(self, stage: str | None = None):
        self.setup_tokenizer()
        data_config = self.config.data

        if not data_config.dataset_path:
            return

        full_dataset = RewardDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )

        val_path = data_config.val_dataset_path
        if val_path:
            self.train_dataset = full_dataset
            self.val_dataset = RewardDataset(
                file_path=val_path,
                tokenizer=self.tokenizer,
                max_seq_len=data_config.max_seq_len,
            )
            return

        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if val_size > 0:
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
