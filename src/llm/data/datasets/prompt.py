"""Prompt dataset for RLHF / PPO rollouts."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Dataset of prompt strings loaded from JSONL."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.prompts: list[str] = []
        with self.file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("instruction") or item.get("text")
                if prompt:
                    self.prompts.append(str(prompt))

        if not self.prompts:
            raise ValueError(f"No prompts found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"prompt": self.prompts[idx]}
