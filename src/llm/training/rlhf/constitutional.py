"""Constitutional-AI slice: rule-based self-critique -> revision (TASK-235).

Constitutional AI (Bai et al. 2022) steers a model with a *constitution* — a
list of principles — plus a self-critique / self-revision loop that makes the
model's own output better comply with those principles. This slice is the
CPU-measurable, rule-based core of that loop (a stand-in for a real critique
model, mirroring how ``TargetTokenJudge`` stands in for a real preference
judge):

- a :class:`Constitution` of :class:`Principle` objects and a judge that scores
  a response by the fraction of principles it satisfies;
- a :func:`critique` that reports which principles a response violates;
- a :func:`revise` that deterministically rewrites a response to satisfy them.

The CPU e2e shows revised responses satisfy a higher fraction of the
constitution than the originals — reproducing the "critique -> revise ->
improved compliance" signal in a small, reproducible setting.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import torch


class Principle(abc.ABC):
    """A single constitution rule; a response either satisfies it or not."""

    name: str

    @abc.abstractmethod
    def is_satisfied(self, ids: Sequence[int]) -> bool:
        """True when a token sequence complies with this principle."""


class ForbiddenToken(Principle):
    """Principle: the response must not contain any of ``forbidden`` tokens."""

    def __init__(self, forbidden: set[int], *, name: str | None = None) -> None:
        self.forbidden = frozenset(int(t) for t in forbidden)
        if not self.forbidden:
            raise ValueError("ForbiddenToken requires a non-empty forbidden set")
        self.name = name or f"forbid-tokens-{sorted(self.forbidden)}"

    def is_satisfied(self, ids: Sequence[int]) -> bool:
        return not (self.forbidden & set(ids))


class EndsWithToken(Principle):
    """Principle: the response's last token must equal ``target``."""

    def __init__(self, target: int, *, name: str | None = None) -> None:
        self.target = int(target)
        self.name = name or f"end-with-token-{self.target}"

    def is_satisfied(self, ids: Sequence[int]) -> bool:
        return len(ids) > 0 and ids[-1] == self.target


class Constitution:
    """A list of :class:`Principle` rules + a score over them."""

    def __init__(self, principles: Sequence[Principle]) -> None:
        if not principles:
            raise ValueError("Constitution requires at least one principle")
        self.principles = list(principles)

    def satisfied_count(self, ids: Sequence[int]) -> int:
        return sum(1 for p in self.principles if p.is_satisfied(ids))

    def score(self, ids: Sequence[int]) -> float:
        """Fraction of principles a response satisfies, in ``[0, 1]``."""
        return self.satisfied_count(ids) / len(self.principles)

    def violations(self, ids: Sequence[int]) -> list[Principle]:
        return [p for p in self.principles if not p.is_satisfied(ids)]


def critique(ids: Sequence[int], constitution: Constitution) -> str:
    """Describe which constitution principles a response violates."""
    bad = constitution.violations(ids)
    if not bad:
        return "satisfies all principles"
    return "violates: " + ", ".join(p.name for p in bad)


def revise(ids: Sequence[int], constitution: Constitution, safe_token: int = 0) -> list[int]:
    """Deterministically rewrite ``ids`` to satisfy the constitution.

    Handles the two built-in principle kinds only:
    - :class:`ForbiddenToken`: every forbidden token is replaced by ``safe_token``;
    - :class:`EndsWithToken`: the last token is forced to the target.
    Other principle types raise :class:`NotImplementedError` so unsupported
    constitutions fail loud instead of silently returning unchanged text.
    """
    out = list(ids)
    for principle in constitution.principles:
        if isinstance(principle, ForbiddenToken):
            out = [safe_token if t in principle.forbidden else t for t in out]
        elif isinstance(principle, EndsWithToken):
            if out:
                out[-1] = principle.target
        else:
            raise NotImplementedError(f"revise does not support principle {type(principle).__name__}")
    return out


def constitutional_loop(
    responses: torch.Tensor,
    constitution: Constitution,
    safe_token: int = 0,
) -> dict[str, torch.Tensor | list[str] | list[list[int]]]:
    """Critique + revise a ``[B, L]`` batch, returning before/after scores."""
    if responses.ndim != 2:
        raise ValueError(f"constitutional_loop expects [B, L] responses, got shape {tuple(responses.shape)}")

    before = torch.zeros(responses.size(0), dtype=torch.float)
    after = torch.zeros(responses.size(0), dtype=torch.float)
    critiques: list[str] = []
    revisions: list[list[int]] = []
    for i in range(responses.size(0)):
        row = [int(t) for t in responses[i].tolist()]
        before[i] = constitution.score(row)
        critiques.append(critique(row, constitution))
        revised = revise(row, constitution, safe_token=safe_token)
        after[i] = constitution.score(revised)
        revisions.append(revised)

    return {
        "scores_before": before,
        "scores_after": after,
        "critiques": critiques,
        "revisions": revisions,
    }
