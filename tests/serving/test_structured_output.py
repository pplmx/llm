"""Tests for the Responses API's structured output (``text.format``).

Scope is stated honestly in ``llm.serving.json_grammar``: the JSON *grammar* is
implemented and cross-validated against ``json.loads``, and the schema check runs
after generation. The grammar is NOT yet wired as a hard logit mask (that needs a
per-step hook in the generation loop), so a schema violation is REPORTED rather
than prevented. These tests pin both halves of that split so the boundary cannot
silently move.
"""

from __future__ import annotations

import json

import pytest

# --- the JSON grammar, cross-validated against the standard library ---------


def _accepts(text: str) -> bool:
    from llm.serving.json_grammar import JSONGrammar

    grammar = JSONGrammar()
    return grammar.feed(text) and grammar.accept()


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        '{"a": 1}',
        '{"a": "bc"}',
        '{"a": [1, 2]}',
        "[1, 2, 3]",
        '"s"',
        "123",
        "true",
        "null",
        '{"a": {"b": null}}',
        '{"a": 1, "b": 2}',
        "[[1], [2]]",
        '{"a": 1.5e3}',
        '{"a": [], "b": {}}',
        '{"k": "hello world"}',
        "[-1.5e-3, true, null]",
        '{"a": [{"b": [1, {"c": "d"}]}]}',
    ],
)
def test_valid_json_is_accepted(text):
    assert _accepts(text), f"the grammar rejected valid JSON: {text!r}"
    assert json.loads(text) is not None or text in ("null",)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": }',
        "[1,]",
        "abc",
        '{"a" 1}',
        '"unterminated',
        '{"a":1} extra',
        "{,}",
        "[1 2]",
        "",
        '{"a": 1,,}',
    ],
)
def test_invalid_json_is_refused(text):
    assert not _accepts(text), f"the grammar accepted invalid JSON: {text!r}"


def test_the_grammar_agrees_with_json_loads_over_a_large_random_sample():
    """Cross-validation against the standard library, not just hand-picked cases.

    A hand-written automaton is exactly the kind of code that is right on the
    examples you thought of and wrong on the one you did not. ``json.dumps``
    generates the valid side and targeted mutations generate the invalid side; a
    disagreement in EITHER direction fails here.
    """
    import random

    def random_json(rng, depth=0):
        choices = ["int", "float", "str", "bool", "null"] + (["obj", "arr"] if depth < 3 else [])
        kind = rng.choice(choices)
        if kind == "int":
            return rng.randint(-1000, 1000)
        if kind == "float":
            return rng.uniform(-1e3, 1e3)
        if kind == "str":
            return "".join(rng.choice("abcXYZ 019_-.!") for _ in range(rng.randint(0, 8)))
        if kind == "bool":
            return rng.choice([True, False])
        if kind == "null":
            return None
        if kind == "obj":
            return {f"k{i}": random_json(rng, depth + 1) for i in range(rng.randint(0, 4))}
        return [random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]

    rng = random.Random(1234)  # noqa: S311 - fixed seed for reproducible fixtures
    checked = 0
    for _ in range(120):
        value = random_json(rng)
        for text in (json.dumps(value), json.dumps(value, separators=(",", ":"))):
            assert _accepts(text), f"grammar rejected json.dumps output: {text[:80]!r}"
            checked += 1

    # The invalid side: truncations and appended junk must be refused.
    for _ in range(60):
        value = random_json(rng)
        text = json.dumps(value)
        if len(text) < 2:
            continue
        for broken in (text[:-1], text + " x", text + ",", "{" + text):
            try:
                json.loads(broken)
                valid = True
            except ValueError, TypeError:
                valid = False
            assert _accepts(broken) == valid, f"grammar disagrees with json.loads on {broken[:60]!r}"
            checked += 1

    assert checked > 200, f"only {checked} cases ran; the sweep is too thin"


# --- schema validation ------------------------------------------------------


_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_a_conforming_value_has_no_problems():
    from llm.serving.json_grammar import validate_against_schema

    assert validate_against_schema({"name": "Ann", "age": 3}, _SCHEMA) == []


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ({"name": "Ann"}, "age"),
        ({"name": "Ann", "age": "3"}, "integer"),
        ({"name": "Ann", "age": 3, "z": 1}, "z"),
        ({"age": 3}, "name"),
    ],
)
def test_violations_are_reported_with_their_path(value, fragment):
    from llm.serving.json_grammar import validate_against_schema

    problems = validate_against_schema(value, _SCHEMA)
    assert problems, f"no problem reported for {value!r}"
    assert any(fragment in problem for problem in problems), problems


def test_a_boolean_does_not_satisfy_an_integer():
    """``True`` IS an ``int`` in Python; JSON keeps them distinct.

    Checking ``isinstance(value, int)`` alone would let ``{"age": true}`` pass a
    schema that says integer.
    """
    from llm.serving.json_grammar import validate_against_schema

    problems = validate_against_schema({"name": "Ann", "age": True}, _SCHEMA)
    assert problems, "a boolean passed an integer schema"
    assert any("integer" in problem for problem in problems), problems


# --- request plumbing -------------------------------------------------------


def test_text_format_json_schema_is_parsed():
    from llm.serving.schemas import ResponsesRequest

    request = ResponsesRequest(
        input="hi",
        text={"format": {"type": "json_schema", "name": "Person", "schema": _SCHEMA}},
    )
    assert request.text.format.type == "json_schema"
    assert request.text.format.name == "Person"
    assert request.text.format.schema_ == _SCHEMA


def test_text_format_is_not_refused_as_unsupported():
    """It used to be on the unsupported-knob list; structured output is real now."""
    from llm.serving.schemas import ResponsesRequest

    request = ResponsesRequest(input="hi", text={"format": {"type": "json_object"}})
    assert request.text.format.type == "json_object"


def test_structured_output_instruction_reaches_the_prompt():
    """The schema must be rendered into the prompt the model actually reads."""
    from llm.serving.chat_template import messages_to_prompt
    from llm.serving.routers.responses import _to_chat_messages
    from llm.serving.schemas import ResponsesRequest

    request = ResponsesRequest(
        input="hi",
        text={"format": {"type": "json_schema", "name": "Person", "schema": _SCHEMA}},
    )
    from llm.serving.routers.responses import _structured_output_spec

    instruction, schema, name = _structured_output_spec(request)
    assert schema == _SCHEMA
    assert name == "Person"
    assert instruction is not None

    messages = [
        *_to_chat_messages(request),
        __import__("llm.serving.schemas", fromlist=["ChatMessage"]).ChatMessage(role="system", content=instruction),
    ]
    rendered = messages_to_prompt(messages)
    assert "Person" in rendered
    assert "JSON Schema" in rendered


def test_plain_text_mode_adds_no_instruction():
    from llm.serving.routers.responses import _structured_output_spec
    from llm.serving.schemas import ResponsesRequest

    instruction, schema, name = _structured_output_spec(ResponsesRequest(input="hi"))
    assert (instruction, schema, name) == (None, None, None)


def test_json_object_mode_asks_for_json_but_carries_no_schema():
    from llm.serving.routers.responses import _structured_output_spec
    from llm.serving.schemas import ResponsesRequest

    instruction, schema, _name = _structured_output_spec(
        ResponsesRequest(input="hi", text={"format": {"type": "json_object"}})
    )
    assert schema is None
    assert instruction is not None
    assert "JSON" in instruction


# --- the report, not the prevention -----------------------------------------


def test_a_non_conforming_reply_is_reported():
    """The server must be able to say a structured reply did not validate.

    This is the honest half of the feature: the grammar is not (yet) a hard logit
    mask, so the contract is "we check", not "we cannot produce a violation". If
    this ever becomes unnecessary because the mask landed, this test should be
    replaced by one asserting the violation cannot be produced at all — not
    deleted.
    """
    from llm.serving.routers.responses import _check_structured_output

    assert _check_structured_output('{"name": "Ann", "age": 3}', _SCHEMA) == []

    problems = _check_structured_output('{"name": "Ann"}', _SCHEMA)
    assert problems
    assert any("age" in problem for problem in problems), problems

    not_json = _check_structured_output("Sure, the name is Ann.", _SCHEMA)
    assert not_json
    assert "not valid JSON" in not_json[0], not_json


def test_the_mask_builder_admits_only_grammatical_tokens():
    """The logit-mask half exists and is correct, even though it is not wired yet.

    Kept tested so that wiring it into the generation loop later is a plumbing
    change rather than a rewrite: the mapping from grammar state to admissible
    token ids is already verified here.
    """
    from llm.serving.json_grammar import JSONGrammar, build_logit_mask

    # A tiny character-level vocabulary: digits, quotes, braces, comma, colon.
    id_to_token = {
        0: "{",
        1: "}",
        2: '"',
        3: ":",
        4: ",",
        5: "a",
        6: "1",
        7: " ",
        8: "x",  # never legal as the first character of a JSON document
    }
    grammar = JSONGrammar()
    mask = build_logit_mask(grammar, id_to_token)
    assert mask is not None
    assert 0 in mask, "an object may start with '{'"
    assert 8 not in mask, "'x' cannot start a JSON document"
    assert 1 not in mask, "'}' cannot be the first character"
