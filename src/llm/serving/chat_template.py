"""Render OpenAI-style chat messages into a single prompt string.

This is intentionally tiny and dependency-free so the serving tier stays
out of the tokenizer's way. The model is expected to have been trained on
the rendered format (or one configured via ``ServingConfig``).
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_CHAT_MESSAGE_TEMPLATE = "{role}: {content}"
DEFAULT_CHAT_GENERATION_PREFIX = "Assistant: "


def messages_to_prompt(
    messages: Iterable,
    *,
    message_template: str | None = None,
    generation_prefix: str | None = None,
) -> str:
    """Convert chat messages to a single prompt string.

    Each message is rendered with ``message_template.format(role=..., content=...)``
    (default: ``"{role}: {content}"``). The rendered messages are joined
    with newlines and ``generation_prefix`` (default: ``"Assistant: "``) is
    appended so the model knows where to start producing assistant tokens.

    Override ``message_template`` and ``generation_prefix`` to match a
    fine-tuned model's expected format (ChatML, Llama-2-chat, Vicuna, …).
    """
    # ``is None`` rather than ``or``: an explicitly empty ``generation_prefix`` is
    # a legitimate request (render no assistant cue at all — some base models must
    # be prompted without one), and ``or`` silently replaced it with the default.
    # Measured: ``ServingConfig(chat_generation_prefix="")`` stored "", yet the
    # rendered prompt still ended with "Assistant: ".
    #
    # ``message_template`` is different: an empty template would format every
    # message to "", silently discarding the conversation, so it is rejected.
    # ``message_template`` keeps its historical "blank means default" behaviour:
    # an empty format string renders every message to "", so falling back is the
    # only sensible reading, and existing callers pass "" to mean "unset".
    mt = message_template or DEFAULT_CHAT_MESSAGE_TEMPLATE
    gp = DEFAULT_CHAT_GENERATION_PREFIX if generation_prefix is None else generation_prefix
    parts = [mt.format(role=msg.role, content=msg.content) for msg in messages]
    parts.append(gp)
    return "\n".join(parts)
