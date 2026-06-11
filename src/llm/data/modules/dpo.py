from llm.data.datasets.dpo import DPODataset
from llm.data.modules.map_base import TokenizedMapDataModule


class DPODataModule(TokenizedMapDataModule):
    """DataModule for Direct Preference Optimization (DPO)."""

    def setup(self, stage: str | None = None):
        self.setup_tokenizer()

        data_config = self.config.data
        if not data_config.dataset_path:
            return

        full_dataset = DPODataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )
        self.train_dataset, self.val_dataset = self.split_train_val(full_dataset)
