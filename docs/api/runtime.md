# `llm.runtime` — Plugin Registries and Model Factory

The runtime layer wires extension points (model architectures,
attention, MLP, norms, generation backends) via the generic
:class:`~llm.runtime.registry.Registry` kernel. The model factory
turns a ``ModelConfig`` into an instantiated ``DecoderModel``.

## Registries

::: llm.runtime.registry

## Model Factory

::: llm.runtime.model_factory

## Plugin Loader

::: llm.runtime.plugins
