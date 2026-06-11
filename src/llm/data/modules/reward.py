"""Reward Model DataModule for RLHF."""

from llm.data.datasets.reward import RewardDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class RewardDataModule(TokenizedMapDataModule):
    """DataModule for Reward Model training with DDP-compatible loaders."""

    def setup(self, stage: str | None = None):
        self.setup_tokenized_file_dataset(RewardDataset, stage)
