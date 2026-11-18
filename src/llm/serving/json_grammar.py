"""Constrained decoding for the Responses API's structured-output (``text.format``).

A JSON Schema handed to ``text.format`` is a *contract*: the caller parses the
output as JSON and expects it to validate. Prompting for JSON and hoping is not
that contract — the model can emit trailing prose, a markdown fence, or an
object that misses a required key, and the caller's parser fails at runtime.

This module implements the lexical half of constrained decoding: a character
automaton that tracks how much of a JSON document has been produced and reports
which characters may legally come next. Prefixes are then matched against the
vocabulary to build a per-step logit mask.

Scope, stated plainly:

* it enforces JSON **syntax** exactly (a masked step cannot emit text that makes
  the output non-JSON);
* it does NOT enforce the schema's structural constraints (types, required keys,
  enum values). Doing that needs the schema compiled into the automaton, which
  is a substantially larger piece of work and is called out as the remaining
  gap in the ADR rather than approximated here.

The automaton is character-based on purpose: tokenizers here can be
character-level (``SimpleCharacterTokenizer``) or subword (HF BPE), and a
character-level state machine composes with either — the vocabulary scan decides
which *tokens* are admissible given their text.

Grammar accepted (the JSON the response should contain):

    value := object | array | string | number | true | false | null
    ws    := ' ' | '\t' | '\n' | '\r'

The machine is a small hand-written stack rather than a regex because JSON
nesting is unbounded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Every character valid JSON can contain. Used to bound the vocabulary scan: a
# token whose text has a character outside this set can never be emitted once
# the constraint is active.
# Sentinel returned when EVERY character is admissible (a JSON string body).
# ``allowed_characters`` otherwise returns the concrete set a limit imposes.
ANY_CHARACTER = "\x00<any>"

JSON_CHARS = frozenset('{}[],:"\\ \t\n\r0123456789+-.eEtruefalsn/bfu')

# States of the automaton. ``VALUE`` means "a value may start here";
# ``AFTER_VALUE`` means "a value just ended, expect a separator or a closer".
VALUE = "value"
AFTER_VALUE = "after_value"
STRING = "string"
STRING_ESCAPE = "string_escape"
STRING_UNICODE = "string_unicode"
NUMBER = "number"
LITERAL = "literal"
EXPECT_COLON = "expect_colon"
DONE = "done"
FAILED = "failed"

_TRUE, _FALSE, _NULL = "true", "false", "null"


@dataclass
class _Frame:
    """One level of nesting: an object or an array plus its progress."""

    kind: str  # "object" | "array"
    # object: expect_key -> expecting a key string; expect_colon -> after a key;
    #         expect_value -> after the colon; expect_comma_or_end -> after a pair
    phase: str = "start"
    count: int = 0
    # Set after a ',' so a trailing comma cannot be closed by '}' / ']'.
    saw_comma: bool = False


@dataclass
class JSONGrammar:
    """Character automaton tracking a JSON document's prefix.

    Feed it the text produced so far (or step it character by character); ask it
    which characters are admissible next. ``done`` becomes True once a complete
    top-level value has been produced — everything after that is rejected, so the
    model cannot append prose.
    """

    state: str = VALUE
    stack: list[_Frame] = field(default_factory=list)
    literal_target: str = ""
    literal_index: int = 0
    unicode_remaining: int = 0
    started: bool = False
    done: bool = False

    # -- public API --------------------------------------------------------

    # When True, leading whitespace at the very start is refused even though JSON
    # permits it. A model free to emit leading spaces emits them forever (measured:
    # 40 constrained tokens of pure space, never reaching the document), because
    # space is always legal there and the next step is again unconstrained.
    # Constrained decoding for structured output wants the document, not a prefix
    # of blanks.
    forbid_leading_whitespace: bool = True

    def allowed_characters(self) -> frozenset[str] | str:
        """Characters that may legally follow the current prefix."""
        state = self.state
        if state == DONE:
            return frozenset()
        if state == STRING:
            # A string body is unbounded: any character except the two that need
            # special handling (a bare backslash starts an escape; a quote closes
            # it). Enumerating a fixed set here wrongly excluded ordinary letters.
            return ANY_CHARACTER
        if state == EXPECT_COLON:
            return frozenset(": \t\n\r")
        if state == STRING_ESCAPE:
            return frozenset('"\\/bfnrtu')
        if state == STRING_UNICODE:
            return frozenset("0123456789abcdefABCDEF")
        if state == LITERAL:
            return frozenset({self.literal_target[self.literal_index]})
        if state == NUMBER:
            # A number is ended by whatever may legally FOLLOW it: a closing
            # bracket, a comma, whitespace. Restricting this set to digits made
            # every numeric value unclosable ("{", '"', 'a', '"', ':', '1', '}'
            # failed at the '}').
            if self.stack:
                top = self.stack[-1]
                followers = frozenset("}") if top.kind == "object" else frozenset("]")
                followers |= frozenset(", \t\n\r")
            else:
                followers = frozenset(" \t\n\r")
            return frozenset("0123456789+-.eE") | followers
        if state == VALUE:
            values = frozenset('{["tfn-0123456789')
            if not self.stack:
                if not self.started and self.forbid_leading_whitespace:
                    # Nothing produced yet: only a real value may start the
                    # document. JSON allows leading blanks; a decoder must not.
                    return values
                return values | frozenset(" \t\n\r")
            top = self.stack[-1]
            if top.kind == "object" and top.phase == "expect_key":
                # '}' closes an empty object; after a trailing comma it would be
                # illegal, which is why the closer is gated on the pair count.
                return (frozenset('"}') if not top.saw_comma else frozenset('"')) | frozenset(" \t\n\r")
            if top.kind == "array" and top.phase in ("expect_value", "after_comma"):
                # An empty array closes immediately: ']' is legal before any item.
                closer = frozenset("]") if (top.phase == "expect_value" and not top.count) else frozenset()
                return values | closer | frozenset(" \t\n\r")
            return values | frozenset(" \t\n\r")
        if state == AFTER_VALUE:
            if not self.stack:
                return frozenset(" \t\n\r")  # trailing whitespace only
            top = self.stack[-1]
            if top.phase != "expect_comma_or_end":
                return frozenset()
            # After a value inside a container, a comma may start the next pair /
            # item and the closer may end it. Omitting the comma made every
            # container hold at most one element ('[[1], [2]]' failed at its comma).
            closer = "}" if top.kind == "object" else "]"
            return frozenset(", \t\n\r" + closer)
        return frozenset()

    def step(self, char: str) -> bool:
        """Advance by one character. Returns False when the char is illegal."""
        allowed = self.allowed_characters()
        # ``ANY_CHARACTER`` marks a position (a string body) where the JSON
        # grammar imposes no restriction, so the membership test is skipped.
        if allowed is not ANY_CHARACTER and char not in allowed:
            self.state = FAILED
            return False
        self._advance(char)
        return True

    def feed(self, text: str) -> bool:
        """Advance over ``text``. Returns False as soon as it becomes illegal."""
        return all(self.step(char) for char in text)

    def accept(self) -> bool:
        """True when the text produced SO FAR is a complete JSON document.

        ``feed`` alone cannot decide this: nothing terminates a top-level number,
        so the end of input is the terminator and the caller must ask.

        PURE: it reports without advancing. An earlier version called
        ``_value_completed()`` here, so asking mid-generation marked the document
        finished — measured: a constrained run emitted "9", the next ``accept()``
        flipped the state to ``done``, every further token became illegal, and the
        processor gave up and let an unconstrained tail through.
        """
        if self.state == DONE:
            return True
        if self.state == NUMBER and not self.stack:
            return True
        return bool(self.state == AFTER_VALUE and not self.stack)

    def clone(self) -> JSONGrammar:
        """A copy that can be advanced without disturbing this one."""
        return JSONGrammar(
            state=self.state,
            stack=[_Frame(frame.kind, frame.phase, frame.count, frame.saw_comma) for frame in self.stack],
            literal_target=self.literal_target,
            literal_index=self.literal_index,
            unicode_remaining=self.unicode_remaining,
            started=self.started,
            done=self.done,
        )

    # -- transitions -------------------------------------------------------

    def _advance(self, char: str) -> None:
        state = self.state

        if state == EXPECT_COLON:
            if char == ":":
                self.stack[-1].phase = "expect_value"
                self.state = VALUE
            return

        if state == STRING:
            if char == '"':
                self.state = AFTER_VALUE if not self.stack else VALUE
                self._finish_string()
            elif char == "\\":
                self.state = STRING_ESCAPE
            return

        if state == STRING_ESCAPE:
            if char == "u":
                self.unicode_remaining = 4
                self.state = STRING_UNICODE
            else:
                self.state = STRING
            return

        if state == STRING_UNICODE:
            self.unicode_remaining -= 1
            if self.unicode_remaining == 0:
                self.state = STRING
            return

        if state == LITERAL:
            self.literal_index += 1
            if self.literal_index >= len(self.literal_target):
                self._value_completed()
            return

        if state == NUMBER:
            if char in "0123456789+-.eE":
                return
            # A number ends the moment a non-number character arrives.
            self._value_completed()
            self._advance(char)
            return

        if state == VALUE:
            self._start_value(char)
            return

        if state == AFTER_VALUE:
            self._after_value(char)
            return

    def _in_object_key_slot(self) -> bool:
        """True when the next token must be an object KEY (not a value)."""
        if not self.stack:
            return False
        top = self.stack[-1]
        return top.kind == "object" and top.phase == "expect_key"

    def _start_value(self, char: str) -> None:
        if not self.stack and not self.started and char in " \t\n\r":
            return  # leading whitespace at the very top level
        if char in " \t\n\r":
            return  # whitespace before a value or a closer
        if char in "}]" and self.stack:
            # Closing an EMPTY container (or one right after a comma, which is
            # illegal and must fail): the closer arrives while a value is still
            # expected, so it is resolved here rather than in AFTER_VALUE.
            top = self.stack[-1]
            if top.saw_comma:
                self.state = FAILED
                return
            if (char == "}" and top.kind == "object") or (char == "]" and top.kind == "array"):
                self.stack.pop()
                self._value_completed()
                return
            self.state = FAILED
            return
        self.started = True

        if char == "{":
            self.stack.append(_Frame("object", "expect_key"))
            self.state = VALUE
            return
        if char == "[":
            self.stack.append(_Frame("array", "expect_value"))
            self.state = VALUE
            return
        if char == '"':
            # Remember whether this string is an object KEY or a VALUE: the
            # closing quote resolves differently, and treating every string as a
            # value made the object's mandatory ':' illegal.
            self.string_is_key = self._in_object_key_slot()
            self.state = STRING
            return
        if char == "t":
            self._begin_literal(_TRUE)
            return
        if char == "f":
            self._begin_literal(_FALSE)
            return
        if char == "n":
            self._begin_literal(_NULL)
            return
        if char in "-0123456789":
            self.state = NUMBER
            return
        if char in " \t\n\r":
            return  # whitespace inside a container, before a value
        self.state = FAILED

    def _begin_literal(self, target: str) -> None:
        self.literal_target = target
        self.literal_index = 1
        if self.literal_index >= len(target):
            self._value_completed()
        else:
            self.state = LITERAL

    def _finish_string(self) -> None:
        """A closing quote landed; resolve whether it was a key or a value."""
        if self.string_is_key and self.stack:
            top = self.stack[-1]
            top.phase = "expect_colon"
            self.state = EXPECT_COLON
            return
        self._value_completed()

    def _value_completed(self) -> None:
        if not self.stack:
            self.done = True
            self.state = DONE
            return
        top = self.stack[-1]
        top.phase = "expect_comma_or_end"
        top.count += 1
        top.saw_comma = False
        self.state = AFTER_VALUE

    def _after_value(self, char: str) -> None:
        if char in " \t\n\r":
            return
        if not self.stack:  # pragma: no cover - guarded by allowed_characters
            self.state = DONE
            return
        top = self.stack[-1]
        if char == ",":
            top.saw_comma = True
            if top.kind == "object":
                top.phase = "expect_key"
            else:
                top.phase = "after_comma"
            self.state = VALUE
            return
        if char == "}" and top.kind == "object":
            self.stack.pop()
            self._value_completed()
            return
        if char == "]" and top.kind == "array":
            self.stack.pop()
            self._value_completed()
            return
        self.state = FAILED


def schema_to_instruction(name: str, schema: dict, description: str | None = None) -> str:
    """Render a JSON Schema as an instruction the model can follow.

    Used as the *soft* half of structured output: the grammar guarantees valid
    JSON syntax, this text steers the model toward the requested SHAPE. Both are
    needed — a syntactically valid object with the wrong keys still fails the
    caller's validation.
    """
    parts = [
        f"Respond with a single JSON value only, matching this JSON Schema named {name!r}.",
        "Do not wrap it in markdown fences and do not add any prose before or after it.",
    ]
    if description:
        parts.append(description)
    parts.append("JSON Schema:")
    parts.append(json.dumps(schema, indent=2, sort_keys=True))
    return "\n".join(parts)


def validate_against_schema(value: object, schema: dict) -> list[str]:
    """Best-effort structural validation of ``value`` against ``schema``.

    Deliberately minimal and dependency-free: it checks the constraints that are
    cheap and that a caller most often relies on (``type``, ``required``,
    ``properties`` recursion, ``enum``, ``items``). Anything else is ignored
    rather than guessed at. ``jsonschema`` is not a project dependency, and
    pulling one in for this would be a heavier decision than the feature warrants.

    Returns a list of human-readable problems; empty means it passed.
    """
    problems: list[str] = []
    _validate(value, schema, path="$", problems=problems)
    return problems


def _validate(value: object, schema: dict, *, path: str, problems: list[str]) -> None:
    if not isinstance(schema, dict):
        return

    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(value, expected):
        problems.append(f"{path}: expected {expected}, got {type(value).__name__}")
        return

    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{path}: missing required property {key!r}")  # noqa: PERF401
        for key, sub_schema in properties.items():
            if key in value:
                _validate(value[key], sub_schema, path=f"{path}.{key}", problems=problems)
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    problems.append(f"{path}: unexpected property {key!r}")  # noqa: PERF401

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, path=f"{path}[{index}]", problems=problems)


def _type_matches(value: object, expected: str) -> bool:
    # ``bool`` is a subclass of ``int`` in Python; JSON treats them as distinct,
    # so check booleans first or True would satisfy "integer".
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True  # unknown type names are not guessed at


# --- mask construction -----------------------------------------------------


def build_logit_mask(
    grammar: JSONGrammar,
    id_to_token: dict[int, str],
    *,
    vocab_size: int | None = None,
    eos_id: int | None = None,
) -> set[int] | None:
    """Token ids the grammar admits next, or ``None`` meaning "no constraint".

    ``None`` is returned when the mask would be empty: at least one token must
    always be allowed or the step can never be taken. An empty mask means the
    vocabulary cannot express a legal continuation (a character-level vocabulary
    with no digit, say), and the honest response is to stop constraining rather
    than to produce a token that breaks the JSON.

    ``eos_id`` is admitted whenever the grammar accepts the current text as
    complete, so generation can end cleanly instead of being forced to pad.
    """
    allowed = grammar.allowed_characters()
    unlimited = allowed is ANY_CHARACTER
    if not unlimited and not allowed:
        # No character at all is admissible. When the document is already
        # complete this is the DESIRED state: the only legal move is to stop, so
        # return EOS alone and generation ends on a valid document. Without EOS
        # in the vocabulary the caller cannot be forced to stop, and it falls
        # back to unconstrained (``build_logit_mask`` returns None) rather than
        # fabricating a token.
        if grammar.accept():
            return {eos_id} if eos_id is not None else None
        return set()

    size = vocab_size if vocab_size is not None else (max(id_to_token) + 1 if id_to_token else 0)
    legal: set[int] = set()
    for token_id, text in id_to_token.items():
        if token_id >= size:
            continue
        if not text:
            continue
        if unlimited or all(char in allowed for char in text):
            legal.add(token_id)

    if grammar.accept() and eos_id is not None:
        legal.add(eos_id)
    return legal if legal else None


# --- the constrained-decoding processor ------------------------------------


class GrammarConstrainedProcessor:
    """A ``logits_processor`` that masks every token the JSON grammar refuses.

    This is the hard half of structured output. It is a callable with the shape
    ``stream_generate`` expects — ``(logits, text_so_far) -> logits`` — and it
    keeps ONE piece of state: the grammar advanced over the text produced so far.

    The grammar is rebuilt from ``text_so_far`` rather than advanced incrementally
    because the generation loop may hand over any prefix (it is a pure function of
    the text, so re-deriving it is always correct); ``_reuse`` keeps the common
    case — the text only grew by the last chunk — cheap by cloning the previous
    state and feeding just the new suffix.

    Masking is expressed as ``-inf`` on illegal ids. When the vocabulary cannot
    express ANY legal continuation (no digit in a character-level vocabulary, say)
    the processor stops constraining and records ``gave_up`` instead of forcing a
    token that breaks the contract — the honest degradation, since the alternative
    is silent corruption.
    """

    def __init__(self, id_to_token: dict[int, str], *, eos_id: int | None = None, vocab_size: int | None = None):
        self._id_to_token = id_to_token
        self._eos_id = eos_id
        self._vocab_size = vocab_size
        self._grammar = JSONGrammar()
        self._consumed = ""
        # True once the grammar could not be satisfied and masking was dropped.
        self.gave_up = False

    @property
    def grammar(self) -> JSONGrammar:
        """The grammar advanced over the text produced so far."""
        return self._grammar

    def _advance_to(self, text: str) -> None:
        if text == self._consumed:
            return
        if text.startswith(self._consumed):
            suffix = text[len(self._consumed) :]
            self._grammar.feed(suffix)
        else:  # pragma: no cover - the loop only ever appends
            self._grammar = JSONGrammar()
            self._grammar.feed(text)
        self._consumed = text

    def __call__(self, logits, text_so_far: str):
        """Mask ``logits`` in place and return them."""
        import torch

        if self.gave_up:
            return logits

        self._advance_to(text_so_far)
        if self._grammar.state == FAILED:
            # The prefix already broke the grammar; nothing can be salvaged by
            # masking, so stop constraining and let the caller's validator report.
            self.gave_up = True
            return logits

        allowed = build_logit_mask(
            self._grammar,
            self._id_to_token,
            vocab_size=self._vocab_size,
            eos_id=self._eos_id,
        )
        if allowed is None:
            # The grammar is satisfiable but this vocabulary cannot express the
            # continuation. Dropping the constraint is the honest move.
            self.gave_up = True
            return logits

        mask = torch.full_like(logits, float("-inf"))
        if allowed:
            indices = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
            mask[indices] = 0.0
        else:
            # No admissible token at all (e.g. the document is complete and there
            # is no EOS id to stop on). Leaving every logit at -inf would make
            # sampling NaN, so fall back to unconstrained rather than emit junk.
            self.gave_up = True
            return logits
        return logits + mask

    def is_complete(self, text: str) -> bool:
        """True when ``text`` is already a complete JSON document."""
        self._advance_to(text)
        return self._grammar.accept()
