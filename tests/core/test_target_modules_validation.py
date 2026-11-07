"""Blank ``target_modules`` patterns must fail, not silently match everything.

``target_modules`` is a list of module-name SUBSTRING patterns, matched with
``any(pattern in name for pattern in patterns)``. In Python ``"" in name`` is
always True, so a single empty string bypasses the filter entirely — measured,
``target_modules=[""]`` wrapped EVERY layer exactly like ``None``, while
``["  "]`` matched nothing. Two visually-identical "blank" inputs produced
opposite behaviour, with no hint that the filter had been ignored.

Seven modules shared the same matching idiom (lora / qlora / adalora / adapter /
bitfit / ia3 / prefix_tuning), so the guard lives in one place and every entry
point calls it.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from llm.core.peft.types import reject_blank_target_patterns


class TestTheHelper:
    """``reject_blank_target_patterns`` unit behaviour."""

    @pytest.mark.parametrize("patterns", [None, [], ["qkv_proj"], ["qkv_proj", "out_proj"]])
    def test_valid_patterns_pass(self, patterns):
        reject_blank_target_patterns(patterns, "LoRA")  # must not raise

    @pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
    def test_blank_patterns_rejected(self, blank):
        with pytest.raises(ValueError, match="blank pattern"):
            reject_blank_target_patterns([blank], "LoRA")

    def test_mixed_list_rejected(self):
        """One typo among good patterns must still fail, not be silently dropped."""
        with pytest.raises(ValueError, match="blank pattern"):
            reject_blank_target_patterns(["qkv_proj", ""], "LoRA")


def _linear() -> nn.Linear:
    return nn.Linear(8, 8)


class TestEveryEntryPointRejectsBlank:
    """Each PEFT method must reject the pattern at its own entry point."""

    def test_lora(self):
        from llm.core.lora import apply_lora

        with pytest.raises(ValueError, match="blank pattern"):
            apply_lora(_linear(), target_modules=[""])

    def test_qlora(self):
        from llm.core.qlora import apply_qlora

        with pytest.raises(ValueError, match="blank pattern"):
            apply_qlora(_linear(), target_modules=[""])

    def test_adalora(self):
        from llm.core.adalora import apply_adalora

        with pytest.raises(ValueError, match="blank pattern"):
            apply_adalora(_linear(), target_modules=[""])

    def test_adapter(self):
        from llm.core.adapter import apply_adapter

        with pytest.raises(ValueError, match="blank pattern"):
            apply_adapter(_linear(), target_modules=[""])

    def test_bitfit(self):
        from llm.core.bitfit import apply_bitfit

        with pytest.raises(ValueError, match="blank pattern"):
            apply_bitfit(_linear(), target_modules=[""])

    def test_ia3(self):
        from llm.core.ia3 import apply_ia3

        with pytest.raises(ValueError, match="blank pattern"):
            apply_ia3(_linear(), target_modules=[""])

    def test_prefix_tuning(self):
        from llm.core.prefix_tuning import apply_prefix_tuning

        with pytest.raises(ValueError, match="blank pattern"):
            apply_prefix_tuning(_linear(), prefix_len=4, target_modules=[""])


def test_valid_filters_still_select_the_right_layers():
    """The guard must not disturb real filtering (vacuity check)."""
    import torch

    from llm.models.decoder import DecoderModel

    config = {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 16}

    def wrapped(patterns):
        from llm.core.ia3 import apply_ia3

        torch.manual_seed(0)
        model = DecoderModel(**config)
        apply_ia3(model, target_modules=patterns)
        return sum(1 for module in model.modules() if type(module).__name__ == "IA3Linear")

    assert wrapped(None) > wrapped(["qkv_proj"]) > 0, "filtering no longer narrows the selection"
