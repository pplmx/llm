# Expose version
# Expose version

__version__ = "0.1.0"

# Import core components to ensure they are registered via the Registry pattern
# This ensures that when 'llm' is imported, the registry is populated.
# Core components (Import to trigger registration)
import llm.core.attn.mha  # noqa: F401
import llm.core.mlp  # noqa: F401
import llm.core.moe  # noqa: F401
