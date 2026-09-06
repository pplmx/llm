"""Smoke tests that run the shipped ``examples/*.py`` scripts end-to-end.

CI only ``py_compile``s the examples (circle 0646889-era gap), so a broken
runtime idiom such as ``activation="swiglu"`` (not a ``get_activation_layer``
name) or ``output_moe, load = moe(x)`` (``MoE.forward`` returns a single
tensor, not a tuple) sailed through for commits. These tests execute each
CPU-runnable example as a subprocess — exactly as a user would — and assert
exit 0 with no traceback.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    # name, needs_openai_sdk
    ("inference_demo.py", False),
    ("kv_cache_demo.py", False),
    ("moe_vs_mlp_demo.py", False),
    ("qlora_finetuning_demo.py", False),
]

# `openai_client_demo.py` requires a running llm-serve + the openai SDK, so
# it is deliberately excluded — a live-server smoke belongs in tests/serving.


@pytest.mark.slow
@pytest.mark.parametrize(("script", "_unused"), EXAMPLES)
def test_example_runs_cleanly(script: str, _unused: bool) -> None:
    """Run ``examples/<script>`` in a subprocess and require exit 0."""
    example = REPO_ROOT / "examples" / script
    proc = subprocess.run(  # noqa: S603 — script from a hardcoded constant tuple above
        [sys.executable, str(example)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"examples/{script} failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )
    # A crash that happens to exit 0 (e.g. caught in an exception handler)
    # still leaves a traceback behind — treat it as a failure.
    assert "Traceback" not in proc.stderr, f"examples/{script} printed a traceback:\n{proc.stderr[-2000:]}"
