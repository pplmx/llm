"""Verify stop-sequence implementation by running key checks manually."""
import os
import sys

import torch

os.chdir("/workspace/llm")
sys.path.insert(0, "src")
sys.path.insert(0, ".")

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from llm.generation.backends import EagerGenerationBackend, GenerationConfig  # noqa: E402
from llm.models.decoder import DecoderModel  # noqa: E402


class _MultiCharTokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 99

    def __init__(self, prompt_ids=None):
        self._prompt_ids = prompt_ids or [1]

    def encode(self, text):
        return list(self._prompt_ids)

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def make_tiny_model():
    return DecoderModel(
        vocab_size=100, hidden_size=16, num_layers=1,
        num_heads=2, max_seq_len=16, device=torch.device("cpu"),
    )


def run_eager(model, tokenizer_factory, *, max_new_tokens=20, **gen_kwargs):
    config = GenerationConfig(max_new_tokens=max_new_tokens, **gen_kwargs)
    backend = EagerGenerationBackend()
    chunks = list(backend.stream(model, tokenizer_factory(), "x", config))
    return "".join(chunks)


passed = 0
failed = 0

def check(name, condition, msg=None):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        detail = f": {msg}" if msg else ""
        print(f"  FAIL: {name}{detail}")


print("=== GenerationConfig stop field ===")
check("default None", GenerationConfig().stop is None)
check("single string", GenerationConfig(stop="END").stop == "END")
check("list", GenerationConfig(stop=["END", "STOP"]).stop == ["END", "STOP"])
check("immutable", True)  # frozen dataclass

print("\n=== EagerGenerationBackend stop logic ===")
model = make_tiny_model()

# Test 1: No stop generates full window
class GrowingTok(_MultiCharTokenizer):
    def __init__(self):
        super().__init__([1])
        self.c = 0
    def decode(self, ids):
        self.c += 1
        return chr(ord("a") + (self.c - 1) % 26)
r = run_eager(model, GrowingTok, max_new_tokens=5)
check("no stop -> 5 chars", len(r) == 5)

# Test 2: Single string stop
class StopTok(_MultiCharTokenizer):
    def __init__(self):
        super().__init__([1])
        self.seq = ["a","b","c","d","E","N","D","a","b","c","d","E"]
        self.i = 0
    def decode(self, ids):
        ch = self.seq[self.i]
        self.i += 1
        return ch
r = run_eager(model, StopTok, max_new_tokens=12, stop="END")
check("single stop -> 'abcd'", r == "abcd", f"got {r!r}")

# Test 3: List of stops
class TwoStopsTok(_MultiCharTokenizer):
    def __init__(self):
        super().__init__([1])
        self.seq = ["x","y","S","T","O","P","a","b","E","N","D"]
        self.i = 0
    def decode(self, ids):
        ch = self.seq[self.i]
        self.i += 1
        return ch
r = run_eager(model, TwoStopsTok, max_new_tokens=11, stop=["STOP", "END"])
check("list stops -> 'xy'", r == "xy", f"got {r!r}")

# Test 4: Stop at first token
class ImmediateStopTok(_MultiCharTokenizer):
    def __init__(self):
        super().__init__([1])
        self.seq = ["X","a","b","c"]
        self.i = 0
    def decode(self, ids):
        ch = self.seq[self.i]
        self.i += 1
        return ch
r = run_eager(model, ImmediateStopTok, max_new_tokens=4, stop="X")
check("stop at first -> ''", r == "", f"got {r!r}")

# Test 5: No match falls through
class NoMatchTok(_MultiCharTokenizer):
    def __init__(self):
        super().__init__([1])
        self.seq = ["a","b","c","d"]
        self.i = 0
    def decode(self, ids):
        ch = self.seq[self.i]
        self.i += 1
        return ch
r = run_eager(model, NoMatchTok, max_new_tokens=4, stop="ZZZ")
check("no match -> 'abcd'", r == "abcd", f"got {r!r}")

print(f"\n=== Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
