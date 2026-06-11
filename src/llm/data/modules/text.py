from llm.data.datasets.text import TextDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class TextDataModule(TokenizedMapDataModule):
    """DataModule for Language Modeling using TextDataset."""

    def setup(self, stage: str | None = None):
        self.setup_tokenizer()

        data_config = self.config.data
        if not data_config.dataset_path:
            return

        full_dataset = TextDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )
        self.train_dataset, self.val_dataset = self.split_train_val(full_dataset)
