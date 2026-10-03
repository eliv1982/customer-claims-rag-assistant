"""Bounded, near-linear matching primitives for the deterministic risk rules.

Two ideas keep the safety rules both correct and cheap on adversarial input:

* **Proximity without wildcards.** A relation like "A somewhere before B" is expressed as an
  ordered chain of simple anchor patterns whose consecutive matches must lie within
  ``ANCHOR_WINDOW`` characters of each other on one line. Each anchor is searched
  independently (every anchor regex is a plain pattern with no ``.*``), and the chain is then
  resolved in Python with binary search. This replaces regexes such as ``A.*B.*C`` whose
  backtracking is polynomial in the message length.

* **Local masks instead of global vetoes.** An exclusion phrase is a *mask* over a region of
  the text. It cancels only the candidate matches that overlap it, so an unrelated phrase
  elsewhere in the message can never switch off a valid signal.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence

Span = tuple[int, int]

# Maximum gap, in characters, between two consecutive anchors of a proximity relation.
ANCHOR_WINDOW = 250

# By default a chain never crosses a line break; relations that must stay inside one sentence
# pass SENTENCE_BREAK instead.
LINE_BREAK = re.compile(r"\n")
SENTENCE_BREAK = re.compile(r"[.!?;\n]")


def spans(pattern: re.Pattern[str], text: str) -> list[Span]:
    """Return the non-overlapping match spans of a simple pattern, in text order."""
    return [match.span() for match in pattern.finditer(text)]


def chain_spans(
    text: str,
    anchors: Sequence[re.Pattern[str]],
    *,
    window: int = ANCHOR_WINDOW,
    stop: re.Pattern[str] = LINE_BREAK,
) -> list[Span]:
    """Return one span per occurrence of the first anchor that can be completed into a chain.

    The chain is ordered: after each anchor match ends, the next anchor must start within
    ``window`` characters and without a ``stop`` match in between (a line break by default).
    The returned span runs from the start of the first anchor to the end of the last anchor
    of the nearest completion.
    """
    if not anchors:
        return []
    per_anchor = [spans(anchor, text) for anchor in anchors]
    if any(not found for found in per_anchor):
        return []

    # tail[i] = (start of a match at the current level, end of the chain's last anchor)
    tail: list[Span] = list(per_anchor[-1])
    for level in range(len(anchors) - 2, -1, -1):
        next_starts = [start for start, _ in tail]
        completed: list[Span] = []
        for start, end in per_anchor[level]:
            index = bisect_left(next_starts, end)
            if index == len(tail):
                continue
            next_start, chain_end = tail[index]
            if next_start - end > window:
                continue
            if stop.search(text, end, next_start):
                continue
            completed.append((start, chain_end))
        if not completed:
            return []
        tail = completed
    return tail


class Masks:
    """Disjoint, sorted exclusion regions with O(log n) overlap queries."""

    __slots__ = ("_regions", "_starts")

    def __init__(self, found: Iterable[Span] = ()) -> None:
        merged: list[Span] = []
        for start, end in sorted(found):
            if merged and start < merged[-1][1]:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        self._regions: tuple[Span, ...] = tuple(merged)
        self._starts: tuple[int, ...] = tuple(start for start, _ in merged)

    def overlaps(self, span: Span) -> bool:
        """True when ``span`` intersects any masked region."""
        if not self._regions:
            return False
        start, end = span
        index = bisect_right(self._starts, end - 1) - 1
        if index < 0:
            return False
        return self._regions[index][1] > start


def has_unmasked(candidates: Iterable[Span], masks: Masks) -> bool:
    """True when at least one candidate span is not cancelled by a local mask."""
    return any(not masks.overlaps(candidate) for candidate in candidates)
