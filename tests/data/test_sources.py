"""Tests for pluggable text sources."""

import pytest

from llm.data.sources import LocalLineTextSource, build_text_source
from llm.training.core.config import Config


def test_local_line_text_source(tmp_path):
    text_file = tmp_path / "data.txt"
    text_file.write_text("alpha\n\nbeta\n", encoding="utf-8")

    texts = list(LocalLineTextSource(text_file).iter_texts())
    assert texts == ["alpha", "beta"]


def test_build_text_source_local():
    config = Config()
    config.data.data_source = "local"
    config.data.dataset_path = __file__

    source = build_text_source(config.data)
    assert isinstance(source, LocalLineTextSource)


def test_build_text_source_hf_requires_dataset_name():
    config = Config()
    config.data.data_source = "hf"

    with pytest.raises(ValueError, match="dataset_name"):
        build_text_source(config.data)


def test_hf_stream_text_source_class():
    from llm.data.sources import HFStreamTextSource

    source = HFStreamTextSource("wikitext", dataset_config="wikitext-2-raw-v1", text_column="text")
    assert source.dataset_name == "wikitext"
    assert source.text_column == "text"
