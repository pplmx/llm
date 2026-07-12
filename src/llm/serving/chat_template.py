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
    mt = message_template or DEFAULT_CHAT_MESSAGE_TEMPLATE
    gp = generation_prefix or DEFAULT_CHAT_GENERATION_PREFIX
    parts = [mt.format(role=msg.role, content=msg.content) for msg in messages]
    parts.append(gp)
    return "\n".join(parts)
