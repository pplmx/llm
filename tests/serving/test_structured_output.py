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


# --- constrained decoding (the logits processor) ----------------------------


def _toy_vocab() -> dict[int, str]:
    """A de-duplicated character vocabulary that can express any JSON."""
    chars = list(dict.fromkeys('{}[]",:0123456789- truefalsn'))
    return dict(enumerate(chars))


def _legal_chars(processor, context: str, vocab_size: int) -> set[str]:
    import torch

    out = processor(torch.zeros(vocab_size), context)
    allowed = out.isfinite()
    return {processor._id_to_token[i] for i in allowed.nonzero().flatten().tolist()}


def test_the_processor_masks_illegal_continuations():
    """A logits processor must actually remove the impossible next characters.

    This is the half that turns structured output from "we check the reply" into
    "the reply cannot be malformed", so each state is asserted on the characters
    that must survive AND on one that must not.
    """
    from llm.serving.json_grammar import GrammarConstrainedProcessor

    vocab = _toy_vocab()
    processor = GrammarConstrainedProcessor(vocab, eos_id=None, vocab_size=len(vocab))

    first = _legal_chars(processor, "", len(vocab))
    assert "{" in first
    assert '"' in first
    assert "x" not in first, "a bare 'x' cannot start a JSON document"

    after_object = _legal_chars(processor, "{", len(vocab))
    assert '"' in after_object, "an object starts with a key"
    assert "1" not in after_object, "an object cannot start with a bare number"

    after_key = _legal_chars(processor, '{"a"', len(vocab))
    assert ":" in after_key
    assert "1" not in after_key, "a key must be followed by ':'"


def test_leading_whitespace_is_refused_while_the_document_is_empty():
    """JSON allows leading blanks; a decoder must not.

    Measured before this rule: a constrained run emitted 40 tokens of pure space
    and never reached the document, because space stays legal at position 0 and
    the next step is again unconstrained.
    """
    from llm.serving.json_grammar import GrammarConstrainedProcessor

    vocab = _toy_vocab()
    processor = GrammarConstrainedProcessor(vocab, eos_id=None, vocab_size=len(vocab))
    first = _legal_chars(processor, "", len(vocab))
    assert " " not in first
    assert "\n" not in first


def test_accept_is_pure_and_does_not_finish_the_document():
    """``accept()`` must report without advancing the automaton.

    An earlier version called ``_value_completed()`` here, so asking mid-run
    marked the document finished: measured, a constrained generation emitted "9",
    the next ``accept()`` flipped the state to done, every further token became
    illegal, and the processor gave up and let an unconstrained tail through.
    """
    from llm.serving.json_grammar import JSONGrammar

    grammar = JSONGrammar()
    assert grammar.feed("9")
    before = (grammar.state, grammar.done)

    assert grammar.accept() is True  # "9" IS a complete document
    assert grammar.accept() is True, "asking twice must not change the answer"
    assert (grammar.state, grammar.done) == before, "accept() mutated the automaton"

    # The document can still grow: "9" may become "99".
    assert grammar.step("9")
    assert grammar.state != "done"


def test_a_complete_document_admits_only_eos():
    """Once the JSON is finished the only legal move is to stop.

    Without this the decoder keeps sampling after a complete document, producing
    trailing data ("69e" + newline, measured) that ``json.loads`` rejects.
    """
    from llm.serving.json_grammar import JSONGrammar, build_logit_mask

    vocab = _toy_vocab()
    eos = max(vocab) + 1
    grammar = JSONGrammar()
    assert grammar.feed("null")
    assert grammar.accept()

    mask = build_logit_mask(grammar, vocab, vocab_size=eos + 1, eos_id=eos)
    assert mask == {eos}, f"a finished document should allow only EOS, got {mask}"

    # The processor keeps its own id->text map, so an out-of-vocabulary EOS id is
    # unknown to ``_legal_chars``; assert through the mask builder instead, which
    # is the layer that decides.
    assert build_logit_mask(grammar, vocab, vocab_size=eos + 1, eos_id=eos) == {eos}


def test_the_processor_is_optional_and_changes_nothing_when_absent():
    """A plain request must take exactly the unconstrained path.

    The hook lives in the generation loop, so a regression that applied it
    unconditionally would silently constrain every request on the server.
    """
    from llm.generation.backends import GenerationConfig

    assert GenerationConfig().logits_processor is None


# --- response_format on /v1/chat/completions -------------------------------
#
# Chat spells structured output differently from the Responses API: it NESTS the
# spec under ``json_schema``. Both are OpenAI's own shapes, so both are lowered
# to one (instruction, schema, name) triple and share the grammar and processor.


def test_chat_response_format_json_schema_is_parsed():
    from llm.serving.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "P", "schema": _SCHEMA}},
    )
    assert request.response_format.type == "json_schema"
    assert request.response_format.json_schema.name == "P"
    assert request.response_format.json_schema.schema_ == _SCHEMA


def test_chat_response_format_is_not_refused_as_unsupported():
    """It used to be on the unsupported list; structured output is real now."""
    from llm.serving.schemas import ChatCompletionRequest

    for fmt in ({"type": "json_object"}, {"type": "text"}):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], response_format=fmt)
        assert request.response_format.type == fmt["type"]


@pytest.mark.parametrize(
    ("fmt", "expect_instruction", "expect_schema"),
    [
        (None, False, False),
        ({"type": "text"}, False, False),
        ({"type": "json_object"}, True, False),
        ({"type": "json_schema", "json_schema": {"name": "P", "schema": {"type": "object"}}}, True, True),
    ],
)
def test_chat_lowers_response_format_to_one_spec(fmt, expect_instruction, expect_schema):
    """Every mode must resolve to the same triple the Responses API uses.

    ``text`` is a no-op, ``json_object`` asks for JSON without a schema, and
    ``json_schema`` carries both — if these diverged, the two endpoints would
    constrain differently for the same user intent.
    """
    from llm.serving.routers.chat import _response_format_spec
    from llm.serving.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        **({} if fmt is None else {"response_format": fmt}),
    )
    instruction, schema, _name = _response_format_spec(request)
    assert (instruction is not None) is expect_instruction
    assert (schema is not None) is expect_schema


def test_chat_and_responses_agree_on_the_instruction():
    """The same schema must produce the SAME instruction text on both endpoints.

    They differ only in where the schema sits in the request; the model sees one
    prompt either way, and two renderings would mean two behaviours.
    """
    from llm.serving.routers.chat import _response_format_spec
    from llm.serving.routers.responses import _structured_output_spec
    from llm.serving.schemas import ChatCompletionRequest, ResponsesRequest

    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}

    chat_instruction, chat_schema, chat_name = _response_format_spec(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_schema", "json_schema": {"name": "P", "schema": schema}},
        )
    )
    responses_instruction, responses_schema, responses_name = _structured_output_spec(
        ResponsesRequest(input="hi", text={"format": {"type": "json_schema", "name": "P", "schema": schema}})
    )

    assert chat_schema == responses_schema == schema
    assert chat_name == responses_name == "P"
    assert chat_instruction == responses_instruction


def test_the_json_object_instruction_is_shared_between_endpoints():
    """``json_object`` must render identically too."""
    from llm.serving.routers.chat import _response_format_spec
    from llm.serving.routers.responses import _structured_output_spec
    from llm.serving.schemas import ChatCompletionRequest, ResponsesRequest

    chat_instruction, _, _ = _response_format_spec(
        ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], response_format={"type": "json_object"})
    )
    responses_instruction, _, _ = _structured_output_spec(
        ResponsesRequest(input="hi", text={"format": {"type": "json_object"}})
    )
    assert chat_instruction == responses_instruction
