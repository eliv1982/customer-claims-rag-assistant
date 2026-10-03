"""Minimal script check: can the Russian deterministic rules assess this text at all?"""

from __future__ import annotations

# Cyrillic + Cyrillic Supplement blocks.
_CYRILLIC_FIRST = "Ѐ"
_CYRILLIC_LAST = "ԯ"

# Share of alphabetic characters that must be Cyrillic for the text to count as supported.
_MIN_CYRILLIC_SHARE = 0.5


def is_supported_language(text: str) -> bool:
    """Return True when the text is predominantly Cyrillic.

    Text with no alphabetic characters at all (digits, punctuation, emoji) is not
    supported either: the rules have nothing to assess.
    """
    letters = 0
    cyrillic = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        if _CYRILLIC_FIRST <= char <= _CYRILLIC_LAST:
            cyrillic += 1
    if letters == 0:
        return False
    return cyrillic / letters >= _MIN_CYRILLIC_SHARE
