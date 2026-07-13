# `llm.training` — Training Loop, Configs, Callbacks

The training entry point is `uv run llm-train`. Most users will only
interact with the configs and the public callback hooks.

> **Note:** The training subpackage uses namespace packages
> (`__init__.py`-less directories). Some submodules require optional
> dependencies (tensorboard via the `logging` group; onnx via the
> `test` group) which may not be installed in every docs build
> environment. The CLI entry point `llm.training.train` additionally
> requires `tensorboard`.

For an overview of the training data flow, see
[Training Flow](../development/training-flow.md). For deep dives on
the callback bridge and extending the trainer, see the
[Development guides](../development/README.md).

The source for the RLHF PPO trainer and the core training engine
lives in `src/llm/training/` — browse it directly for the
auto-generated API reference until namespace-package support
stabilises upstream.
