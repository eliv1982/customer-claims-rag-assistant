"""Tests for Unicode-safe token splitting."""

from __future__ import annotations

import pytest

from customer_claims_rag.token_counter import FakeTokenCounter, TiktokenCounter


@pytest.fixture
def counter() -> TiktokenCounter:
    return TiktokenCounter()


def test_empty_text(counter: TiktokenCounter) -> None:
    assert counter.split_by_token_window("", max_tokens=10) == []


def test_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        TiktokenCounter("nonexistent_encoding_xyz")


def test_russian_text_no_replacement_char(counter: TiktokenCounter) -> None:
    text = "привет" * 500
    parts = counter.split_by_token_window(text, max_tokens=50, overlap_tokens=10)
    assert len(parts) > 1
    for part in parts:
        assert "\ufffd" not in part
        assert counter.count(part) <= 50


def test_mixed_script_no_replacement_char(counter: TiktokenCounter) -> None:
    text = ("FoodFlow доставка emoji 🍕 " * 200).strip()
    parts = counter.split_by_token_window(text, max_tokens=40, overlap_tokens=0)
    assert len(parts) > 1
    reconstructed = "".join(parts)
    assert "\ufffd" not in reconstructed
    assert reconstructed == text


def _join_without_overlap(parts: list[str]) -> str:
    if not parts:
        return ""
    result = parts[0]
    for part in parts[1:]:
        max_len = min(len(result), len(part))
        overlap = 0
        for length in range(max_len, 0, -1):
            if part.startswith(result[-length:]):
                overlap = length
                break
        result += part[overlap:]
    return result


def test_overlap_prefix_matches_suffix(counter: TiktokenCounter) -> None:
    text = "абвгдежз " * 300
    parts = counter.split_by_token_window(text, max_tokens=60, overlap_tokens=12)
    assert len(parts) > 1
    for idx in range(1, len(parts)):
        prev, curr = parts[idx - 1], parts[idx]
        max_len = min(len(prev), len(curr))
        shared = 0
        for length in range(max_len, 0, -1):
            if curr.startswith(prev[-length:]):
                shared = length
                break
        assert shared > 0


def test_split_deterministic(counter: TiktokenCounter) -> None:
    text = "токен " * 400
    a = counter.split_by_token_window(text, max_tokens=70, overlap_tokens=10)
    b = counter.split_by_token_window(text, max_tokens=70, overlap_tokens=10)
    assert a == b


def test_fake_counter_overlap_tests() -> None:
    fake = FakeTokenCounter()
    text = " ".join(["word"] * 1500)
    parts = fake.split_by_token_window(text, max_tokens=800, overlap_tokens=80)
    assert len(parts) > 1
