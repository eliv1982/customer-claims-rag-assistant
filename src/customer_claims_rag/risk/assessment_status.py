"""Explicit status of a deterministic risk assessment."""

from __future__ import annotations

from enum import StrEnum


class RiskAssessmentStatus(StrEnum):
    """How the deterministic assessment came about.

    ``RULE_MATCH``            at least one deterministic rule matched; ``risk_floor`` is the
                              maximum level of the matched rules (this is the only status in
                              which a LOW floor can be an affirmative rule outcome).
    ``NO_SIGNAL``             supported-language text and no rule matched. This is the absence
                              of a deterministic signal, NOT a confirmation that the case is low
                              risk. ``risk_floor`` is LOW only as the neutral lower bound.
    ``UNSUPPORTED_LANGUAGE``  the text cannot be assessed by the Russian deterministic rules.
                              The case is not classified; manual review is required.
    """

    RULE_MATCH = "rule_match"
    NO_SIGNAL = "no_signal"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
