"""
Reward Model DataModule for RLHF.
"""

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from llm.data.reward_dataset import RewardDataset
from llm.tokenization.tokenizer import BaseTokenizer


class RewardDataModule:
    """
    DataModule for Reward Model training.

    Wraps RewardDataset and provides train/val DataLoaders.
    """

    def __init__(
        self,
        train_file: str | Path,
        tokenizer: BaseTokenizer,
        val_file: str | Path | None = None,
        max_seq_len: int = 1024,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file) if val_file else None
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset: RewardDataset | None = None
        self._val_dataset: RewardDataset | None = None

    def setup(self):
        """Initialize datasets."""
        self._train_dataset = RewardDataset(
            file_path=self.train_file,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

        if self.val_file:
            self._val_dataset = RewardDataset(
                file_path=self.val_file,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self._train_dataset is None:
            self.setup()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]] | None:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
