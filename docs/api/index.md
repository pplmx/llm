# API Reference

Auto-generated API reference for the ``llm`` package. Pages are built
from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/) — if something looks
stale or missing, the canonical source is the Python file, not this
page.

## Layout

The package is organised by responsibility; each subpackage owns one
slice of the framework.

| Subpackage | Responsibility | Page |
|---|---|---|
| `llm.core` | Building blocks (attention, MLP, norms, KV cache, embeddings) | [core.md](core.md) |
| `llm.runtime` | Plugin registries + model factory | [runtime.md](runtime.md) |
| `llm.models` | High-level model definitions (decoder, regression MLP) | [models.md](models.md) |
| `llm.generation` | Sampling, generation backends, registry | [generation.md](generation.md) |
| `llm.serving` | FastAPI app, routers, batched engine, metrics | [serving.md](serving.md) |
| `llm.training` | Trainer, configs, callbacks, RLHF | [training.md](training.md) |

## Stability

The public surface is whatever is exported through ``llm/__init__.py``
and the per-subpackage ``__init__.py`` files. Anything reachable
through `from llm.X import Y` is part of the supported API;
under-the-hood helpers prefixed with ``_`` are not.
