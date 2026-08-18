"""Tests for :class:`llm.data.datasets.prompt.PromptDataset`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.data.datasets.prompt import PromptDataset


def _write_jsonl(path, lines: list[dict]):
    with Path(path).open("w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")


class TestPromptDataset:
    def test_loads_prompt_field(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "hello world"}, {"prompt": "foo bar"}])
        ds = PromptDataset(data_path)
        assert len(ds) == 2
        assert ds[0] == {"prompt": "hello world"}
        assert ds[1] == {"prompt": "foo bar"}

    def test_falls_back_to_instruction_field(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"instruction": "summarize this"}])
        ds = PromptDataset(data_path)
        assert len(ds) == 1
        assert ds[0] == {"prompt": "summarize this"}

    def test_falls_back_to_text_field(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"text": "raw text prompt"}])
        ds = PromptDataset(data_path)
        assert len(ds) == 1
        assert ds[0] == {"prompt": "raw text prompt"}

    def test_prefers_prompt_over_instruction(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "use this", "instruction": "not this"}])
        ds = PromptDataset(data_path)
        assert ds[0] == {"prompt": "use this"}

    def test_prefers_prompt_over_text(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "use prompt", "text": "not text"}])
        ds = PromptDataset(data_path)
        assert ds[0] == {"prompt": "use prompt"}

    def test_skips_blank_lines(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "first"}])
        with data_path.open("a") as f:
            f.write("\n")
            f.write(json.dumps({"prompt": "second"}) + "\n")
        ds = PromptDataset(data_path)
        assert len(ds) == 2
        assert ds[0] == {"prompt": "first"}
        assert ds[1] == {"prompt": "second"}

    def test_skips_lines_without_recognized_key(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"other": "no prompt"}])
        with pytest.raises(ValueError, match="No prompts found"):
            PromptDataset(data_path)

    def test_coerces_non_string_prompt_to_str(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": 12345}])
        ds = PromptDataset(data_path)
        assert len(ds) == 1
        assert ds[0] == {"prompt": "12345"}

    def test_coerces_instruction_to_string(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"instruction": 42}])
        ds = PromptDataset(data_path)
        assert ds[0] == {"prompt": "42"}

    def test_raises_when_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptDataset(tmp_path / "nonexistent.jsonl")

    def test_malformed_row_raises_value_error_naming_file(self, tmp_path):
        """A malformed JSONL row must surface as a ValueError naming the file
        (aligned with the SFT/DPO/Reward datasets, RIL ISS-201), not a raw
        JSONDecodeError (round-78 TASK-193 / ISS-231)."""
        data_path = tmp_path / "broken.jsonl"
        data_path.write_text('{"prompt": "ok"}\n{"broken}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON in prompt file"):
            PromptDataset(data_path)

    def test_accepts_string_path(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "hello"}])
        ds = PromptDataset(str(data_path))
        assert len(ds) == 1
        assert ds[0] == {"prompt": "hello"}

    def test_len_and_index_match(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        items = [{"prompt": f"prompt_{i}"} for i in range(5)]
        _write_jsonl(data_path, items)
        ds = PromptDataset(data_path)
        assert len(ds) == 5
        for i in range(5):
            assert ds[i] == {"prompt": f"prompt_{i}"}

    def test_item_out_of_bounds(self, tmp_path):
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "only"}])
        ds = PromptDataset(data_path)
        with pytest.raises(IndexError):
            _ = ds[1]
        with pytest.raises(IndexError):
            _ = ds[-2]
