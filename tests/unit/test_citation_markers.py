"""Unit tests for citation marker extraction."""

from __future__ import annotations

from customer_claims_rag.generation.citation_markers import extract_citation_markers


def test_single_marker() -> None:
    assert extract_citation_markers("Ответ [S1].") == ["S1"]


def test_multiple_markers() -> None:
    assert extract_citation_markers("A [S1] and [S2].") == ["S1", "S2"]


def test_repeated_markers_deduplicated() -> None:
    assert extract_citation_markers("[S1] then [S1] again") == ["S1"]


def test_first_appearance_ordering() -> None:
    assert extract_citation_markers("[S2] before [S1]") == ["S2", "S1"]


def test_unknown_formats_ignored() -> None:
    text = "S1 [s1] [S0] [doc::1] [Sx] [S01]"
    assert extract_citation_markers(text) == []


def test_lower_case_marker_ignored() -> None:
    assert extract_citation_markers("[s1]") == []


def test_s0_ignored() -> None:
    assert extract_citation_markers("[S0]") == []


def test_no_markers() -> None:
    assert extract_citation_markers("plain answer") == []


def test_answer_not_modified() -> None:
    original = "Text [S1] stays."
    assert extract_citation_markers(original) == ["S1"]


def test_double_brackets_not_recognized() -> None:
    assert extract_citation_markers("[[S1]]") == []


def test_triple_brackets_not_recognized() -> None:
    assert extract_citation_markers("[[[S1]]]") == []


def test_unclosed_opening_brackets_not_recognized() -> None:
    assert extract_citation_markers("[[S1]") == []


def test_extra_closing_brackets_not_recognized() -> None:
    assert extract_citation_markers("[S1]]") == []


def test_malformed_marker_does_not_block_valid_marker() -> None:
    assert extract_citation_markers("[[S1]] valid [S2]") == ["S2"]


def test_adjacent_valid_markers() -> None:
    assert extract_citation_markers("[S1][S2]") == ["S1", "S2"]


def test_punctuation_around_marker() -> None:
    assert extract_citation_markers("text [S1].") == ["S1"]
