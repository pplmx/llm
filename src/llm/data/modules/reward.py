"""Reward Model DataModule for RLHF."""

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

        self.assign_train_val_datasets(
            full_dataset,
            val_path=data_config.val_dataset_path,
            build_val_dataset=lambda path: RewardDataset(
                file_path=path,
                tokenizer=self.tokenizer,
                max_seq_len=data_config.max_seq_len,
            ),
        )
