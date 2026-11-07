"""Unit tests for ``_messages_to_prompt`` (Tier-2 audit item #5, Finding L).

The chat endpoint used to hardcode ``[System]: ...\n[User]: ...\n[Assistant]: ...``.
This module pins the default behavior and verifies that ``ServingConfig``
overrides propagate through.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm.serving.api import (
    DEFAULT_CHAT_GENERATION_PREFIX,
    DEFAULT_CHAT_MESSAGE_TEMPLATE,
    _messages_to_prompt,
)


@dataclass
class _Msg:
    role: str
    content: str


def test_default_message_template_format():
    """The built-in default is the simple ``{role}: {content}`` form."""
    assert DEFAULT_CHAT_MESSAGE_TEMPLATE == "{role}: {content}"
    assert DEFAULT_CHAT_GENERATION_PREFIX == "Assistant: "


def test_default_render_simple_conversation():
    """Three messages + Assistant prefix, joined by newlines."""
    messages = [
        _Msg("system", "be brief"),
        _Msg("user", "hi"),
    ]
    out = _messages_to_prompt(messages)
    assert out == "system: be brief\nuser: hi\nAssistant: "


def test_render_with_assistant_message_keeps_prefix():
    """An assistant message is rendered the same way; the trailing prefix still appears."""
    messages = [
        _Msg("user", "hi"),
        _Msg("assistant", "hello!"),
        _Msg("user", "how are you?"),
    ]
    out = _messages_to_prompt(messages)
    assert out == ("user: hi\nassistant: hello!\nuser: how are you?\nAssistant: ")


def test_custom_message_template_chatml_style():
    """Override message template to use ChatML-style tags."""
    messages = [
        _Msg("system", "be brief"),
        _Msg("user", "hi"),
    ]
    out = _messages_to_prompt(
        messages,
        message_template="<|im_{role}|>\n{content}<|im_end|>",
    )
    assert out == ("<|im_system|>\nbe brief<|im_end|>\n<|im_user|>\nhi<|im_end|>\nAssistant: ")


def test_custom_generation_prefix():
    """Override only the generation prefix (e.g., for a Vicuna-style finish)."""
    messages = [_Msg("user", "hi")]
    out = _messages_to_prompt(
        messages,
        generation_prefix="ASSISTANT:",
    )
    assert out == "user: hi\nASSISTANT:"


def test_custom_message_template_and_prefix():
    """Both overrides work together."""
    messages = [_Msg("user", "hi")]
    out = _messages_to_prompt(
        messages,
        message_template="### {role}:\n{content}\n",
        generation_prefix="### Response:\n",
    )
    assert out == "### user:\nhi\n\n### Response:\n"


def test_template_with_extra_unused_placeholders_is_safe():
    """If the user-defined template references no extra keys, format() still works."""
    # We don't error if the user's template uses only {role}/{content}.
    out = _messages_to_prompt(
        [_Msg("user", "hi")],
        message_template="{role} says: {content}",
    )
    assert out == "user says: hi\nAssistant: "


def test_empty_messages_still_appends_prefix():
    """An empty conversation renders to just the generation prefix."""
    out = _messages_to_prompt([])
    assert out == "Assistant: "


def test_serving_config_default_is_legacy_format():
    """``ServingConfig.chat_message_template`` defaults to None → legacy default."""
    from llm.serving.config import ServingConfig

    cfg = ServingConfig()
    assert cfg.chat_message_template is None
    assert cfg.chat_generation_prefix is None


def test_serving_config_overrides_via_env(monkeypatch):
    """Chat template fields can be set via env vars (LLM_SERVING_CHAT_*)."""
    from llm.serving.config import ServingConfig

    monkeypatch.setenv("LLM_SERVING_CHAT_MESSAGE_TEMPLATE", "<|im_{role}|>\n{content}<|im_end|>")
    monkeypatch.setenv("LLM_SERVING_CHAT_GENERATION_PREFIX", "<|im_start|>assistant\n")
    cfg = ServingConfig()
    assert cfg.chat_message_template == "<|im_{role}|>\n{content}<|im_end|>"
    assert cfg.chat_generation_prefix == "<|im_start|>assistant\n"


# --- An explicit empty generation prefix must be honoured ---


def test_empty_generation_prefix_renders_no_assistant_cue():
    """``generation_prefix=""`` must mean "append nothing", not "use the default".

    ``or`` conflates an explicit empty string with "unset", so a user who
    configured ``chat_generation_prefix: ""`` — some base models must be prompted
    without an ``Assistant:`` cue — still got the default appended. Measured via
    ``ServingConfig``: the config stored "", yet the rendered prompt still ended
    with "Assistant: ", with no warning.
    """
    messages = [_Msg(role="user", content="hi")]

    with_default = _messages_to_prompt(messages)
    assert with_default.endswith(DEFAULT_CHAT_GENERATION_PREFIX), "the default cue should still be present"

    without_cue = _messages_to_prompt(messages, generation_prefix="")
    assert not without_cue.endswith(DEFAULT_CHAT_GENERATION_PREFIX), (
        "an explicit empty prefix was replaced by the default"
    )
    assert without_cue.rstrip().endswith("hi"), "the message body itself should still render"
    assert without_cue != with_default


def test_none_generation_prefix_still_uses_the_default():
    """``None`` remains the "unset" sentinel, so the default cue still appears."""
    messages = [_Msg(role="user", content="hi")]
    assert _messages_to_prompt(messages, generation_prefix=None) == _messages_to_prompt(messages)


def test_custom_generation_prefix_is_used_verbatim():
    messages = [_Msg(role="user", content="hi")]
    rendered = _messages_to_prompt(messages, generation_prefix="<|assistant|>")
    assert rendered.endswith("<|assistant|>")
