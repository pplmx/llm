import pytest

from llm.data.text_data_module import TextDataModule
from llm.tokenization.tokenizer import HFTokenizer
from llm.training.core.config import Config


class TestDataAbstraction:
    def test_hf_tokenizer_loading(self):
        # Using gpt2 as it is small and standard
        try:
            tokenizer = HFTokenizer.from_pretrained("gpt2")
        except OSError:
            pytest.skip("GPT2 tokenizer download failed (network issue?)")

        assert tokenizer.vocab_size > 0
        tokens = tokenizer.encode("Hello world")
        assert len(tokens) > 0
        text = tokenizer.decode(tokens)
        assert "Hello world" in text

    def test_text_data_module_setup(self, tmp_path):
        # Create dummy text file
        data_file = tmp_path / "data.txt"
        data_file.write_text("hello world " * 100, encoding="utf-8")

        # Config
        config = Config()
        config.data.tokenizer_type = "hf"
        config.data.tokenizer_path = "gpt2"
        config.data.dataset_path = str(data_file)
        config.data.max_seq_len = 10
        config.training.batch_size = 2
        config.optimization.num_workers = 0  # Avoid fork warning in tests

        dm = TextDataModule(config)
        try:
            dm.setup()
        except OSError:
            pytest.skip("GPT2 tokenizer download failed during setup")

        assert dm.tokenizer is not None
        assert isinstance(dm.tokenizer, HFTokenizer)
        assert dm.train_dataset is not None
        assert len(dm.train_dataset) > 0

        # Test Dataloader
        loader, _ = dm.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert batch["input_ids"].shape == (2, 10)
