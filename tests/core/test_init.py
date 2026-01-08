import subprocess
import sys

import pytest


@pytest.mark.quick
def test_import_llm_registers_components():
    """
    Run a subprocess to verify that `import llm` automatically populates the registry.
    This protects against the issue where scripts fail because registries are empty.
    """
    code = """
import llm
from llm.core.registry import ATTENTION_REGISTRY, MLP_REGISTRY

# Check if registries are populated
attn_keys = list(ATTENTION_REGISTRY._registry.keys())
mlp_keys = list(MLP_REGISTRY._registry.keys())

print(f"ATTENTION: {attn_keys}")
print(f"MLP: {mlp_keys}")

if 'mha' not in attn_keys:
    exit(1)
if 'mlp' not in mlp_keys:
    exit(1)
    """

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(f"Subprocess failed verification.\nOutput: {result.stdout}\nError: {result.stderr}")
