"""Deterministic citation marker extraction from model answers."""

from __future__ import annotations

import re

# Match [Sx] only when not wrapped in extra square brackets.
_CITATION_MARKER_PATTERN = re.compile(r"(?<!\[)\[S([1-9][0-9]*)\](?!\])")


def extract_citation_markers(answer: str) -> list[str]:
    """Return unique citation keys in first-appearance order from an answer."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_MARKER_PATTERN.finditer(answer):
        key = f"S{match.group(1)}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
