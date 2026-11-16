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

    def test_skips_non_dict_scalar_rows(self, tmp_path):
        """A scalar JSON row (bare int/string) must be skipped, not crash with
        a raw AttributeError on ``item.get``."""
        data_path = tmp_path / "prompts.jsonl"
        with data_path.open("w", encoding="utf-8") as f:
            for line in [
                '{"prompt": "first"}',
                "42",
                '"just a bare string"',
                '{"instruction": "second"}',
            ]:
                f.write(line + "\n")
        ds = PromptDataset(data_path)
        assert len(ds) == 2
        assert ds[0] == {"prompt": "first"}

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
        (aligned with the SFT/DPO/Reward datasets), not a raw
        JSONDecodeError."""
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


class TestPromptDatasetReportsDroppedRows:
    """A dropped row must never be silent.

    ``PromptDataset`` is the rollout corpus for PPO, so a row it skips is
    training data the user does not get. It used to drop malformed rows and
    rows with no usable prompt without logging anything, while every sibling
    dataset (SFT / DPO / Reward) warned per row. Measured on a five-row file
    where three were unusable, the dataset reported nothing at all and PPO
    would have trained on 40% of the corpus.

    Both tests pin the WARNING text AND the resulting count, so a future change
    cannot keep the message while starting to accept the rows (or the reverse).
    """

    def test_malformed_rows_are_reported(self, tmp_path, caplog):
        data_path = tmp_path / "prompts.jsonl"
        data_path.write_text(
            "\n".join(['{"prompt": "good"}', '"a bare string"', '{"no_prompt_key": 1}', '{"prompt": ""}']) + "\n",
            encoding="utf-8",
        )

        with caplog.at_level("WARNING"):
            ds = PromptDataset(data_path)

        assert len(ds) == 1, "only the one well-formed row is usable"
        messages = [record.message for record in caplog.records]
        assert any("not a JSON object" in m for m in messages), f"non-object row not reported: {messages}"
        empty_reports = [m for m in messages if "no non-empty prompt" in m]
        assert len(empty_reports) == 2, f"both prompt-less rows must be reported, saw {messages}"

    def test_every_drop_has_its_own_message(self, tmp_path, caplog):
        """One summary line is not enough: the reader needs to know WHICH row.

        The warning carries the offending ``item``, which is what lets someone
        find the bad line in a large JSONL.
        """
        data_path = tmp_path / "prompts.jsonl"
        data_path.write_text('{"prompt": "ok"}\n{"missing": "prompt"}\n', encoding="utf-8")

        with caplog.at_level("WARNING"):
            PromptDataset(data_path)

        assert any("missing" in str(record.args) for record in caplog.records if record.args), (
            "the warning must include the offending row, not just a count"
        )

    def test_a_clean_file_logs_no_warnings(self, tmp_path, caplog):
        """Counterpart: nothing dropped means nothing to report.

        Without this, a change that warned on every row would satisfy the tests
        above while flooding the logs of a healthy run.
        """
        data_path = tmp_path / "prompts.jsonl"
        _write_jsonl(data_path, [{"prompt": "a"}, {"prompt": "b"}, {"instruction": "c"}])

        with caplog.at_level("WARNING"):
            ds = PromptDataset(data_path)

        assert len(ds) == 3
        assert not [r for r in caplog.records if r.levelname == "WARNING"], (
            f"clean file produced warnings: {[r.message for r in caplog.records]}"
        )
