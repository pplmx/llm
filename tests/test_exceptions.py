"""Tests for :mod:`llm.exceptions`.

Covers the ``LLMError`` base class (message + details handling, ``__str__``
formatting) and the seven domain-specific subclasses. These had 0 % coverage
previously — the ``__str__`` formatting branch and the ``details or {}``
default were never exercised.
"""

from __future__ import annotations

import pytest

from llm.exceptions import (
    ConfigError,
    DataError,
    InferenceError,
    LLMError,
    ModelError,
    ServingError,
    TokenizationError,
    TrainingError,
    ValidationError,
)

# --------------------------------------------------------------------------- #
# Core LLMError behaviour
# --------------------------------------------------------------------------- #


def test_llm_error_message_no_details():
    """Message with no details -> __str__ returns just the message."""
    err = LLMError("something went wrong")
    assert err.message == "something went wrong"
    assert err.details == {}
    assert str(err) == "something went wrong"


def test_llm_error_message_with_details():
    """Message with details dict -> __str__ includes ``(details: {...})``."""
    err = LLMError("config bad", details={"key": "value"})
    assert err.message == "config bad"
    assert err.details == {"key": "value"}
    assert str(err) == "config bad (details: {'key': 'value'})"


def test_llm_error_details_defaults_to_empty_dict():
    """Omitting *details* gives ``{}``, not ``None``."""
    err = LLMError("no details given")
    assert err.details == {}
    assert err.details is not None


def test_llm_error_details_none_explicit():
    """Passing ``details=None`` explicitly also yields ``{}``."""
    err = LLMError("explicit none", details=None)
    assert err.details == {}


def test_llm_error_empty_dict_details():
    """An empty dict is *not* truthy -> ``__str__`` omits the details suffix."""
    err = LLMError("empty dict", details={})
    assert str(err) == "empty dict"


# --------------------------------------------------------------------------- #
# Domain subclasses (parametrised over all eight subclasses)
# --------------------------------------------------------------------------- #

# Each tuple: (subclass, __name__)
_SUBCLASSES = [
    (ConfigError, "ConfigError"),
    (DataError, "DataError"),
    (InferenceError, "InferenceError"),
    (LLMError, "LLMError"),
    (ModelError, "ModelError"),
    (ServingError, "ServingError"),
    (TokenizationError, "TokenizationError"),
    (TrainingError, "TrainingError"),
    (ValidationError, "ValidationError"),
]


@pytest.mark.parametrize(("subclass", "expected_name"), _SUBCLASSES)
def test_subclass_inherits_from_llm_error(subclass, expected_name):  # noqa: ARG001
    """Every domain subclass is an LLMError subclass."""
    assert issubclass(subclass, LLMError)
    err = subclass("domain error")
    assert isinstance(err, LLMError)
    assert isinstance(err, Exception)
    assert err.message == "domain error"
    assert err.details == {}


@pytest.mark.parametrize(("subclass", "expected_name"), _SUBCLASSES)
def test_subclass_str_without_details(subclass, expected_name):  # noqa: ARG001
    err = subclass("plain message")
    assert str(err) == "plain message"


@pytest.mark.parametrize(("subclass", "expected_name"), _SUBCLASSES)
def test_subclass_str_with_details(subclass, expected_name):  # noqa: ARG001
    err = subclass("with context", details={"layer": 3})
    assert str(err) == "with context (details: {'layer': 3})"


@pytest.mark.parametrize(("subclass", "expected_name"), _SUBCLASSES)
def test_subclass_name(subclass, expected_name):
    assert subclass.__name__ == expected_name
