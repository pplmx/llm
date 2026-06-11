from llm.data.datasets.sft import SFTDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class SFTDataModule(TokenizedMapDataModule):
    """DataModule for Supervised Fine-Tuning (SFT) using SFTDataset."""

    def setup(self, stage: str | None = None):
        self.setup_tokenizer()

        data_config = self.config.data
        if not data_config.dataset_path:
            return

        full_dataset = SFTDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )
        self.train_dataset, self.val_dataset = self.split_train_val(full_dataset)
