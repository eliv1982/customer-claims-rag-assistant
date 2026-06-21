"""Injectable token counter for chunk sizing."""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Protocol for counting and splitting text by tokens."""

    def count(self, text: str) -> int: ...

    def split_by_token_window(
        self, text: str, max_tokens: int, overlap_tokens: int = 0
    ) -> list[str]: ...


class TiktokenCounter:
    """Token counter backed by tiktoken with Unicode-safe splitting."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for TiktokenCounter; install with: pip install tiktoken"
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ValueError(
                f"Unknown or unavailable tiktoken encoding: {encoding_name!r}"
            ) from exc
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def split_by_token_window(
        self, text: str, max_tokens: int, overlap_tokens: int = 0
    ) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        if not text:
            return []

        total = self.count(text)
        if total <= max_tokens:
            return [text]

        parts: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = self._max_char_boundary(text, start, text_len, max_tokens)
            if end <= start:
                raise ValueError(
                    "cannot split text within token limit at Unicode character boundary"
                )

            chunk = text[start:end]
            if not chunk:
                raise ValueError("empty chunk produced during token split")
            if "\ufffd" in chunk:
                raise ValueError("chunk contains replacement character U+FFFD")

            parts.append(chunk)
            if end >= text_len:
                break

            if overlap_tokens > 0:
                overlap_start = self._overlap_char_start(text, start, end, overlap_tokens)
                if overlap_start >= end:
                    start = end
                else:
                    start = overlap_start
            else:
                start = end

        return parts

    def _max_char_boundary(
        self, text: str, start: int, text_len: int, max_tokens: int
    ) -> int:
        """Binary search for largest end index with token count <= max_tokens."""
        lo = start + 1
        hi = text_len
        best = start
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[start:mid]
            if self.count(candidate) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _overlap_char_start(
        self, text: str, chunk_start: int, chunk_end: int, target_overlap: int
    ) -> int:
        """Find start index for next chunk so prefix overlaps previous suffix."""
        lo = chunk_start
        hi = chunk_end
        best = chunk_end
        while lo <= hi:
            mid = (lo + hi) // 2
            suffix = text[mid:chunk_end]
            suffix_tokens = self.count(suffix)
            if suffix_tokens <= target_overlap and suffix:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best


class FakeTokenCounter:
    """Deterministic counter for tests: ~1 token per whitespace-separated word."""

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(text.split())

    def split_by_token_window(
        self, text: str, max_tokens: int, overlap_tokens: int = 0
    ) -> list[str]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")

        words = text.split()
        if not words:
            return []
        if len(words) <= max_tokens:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            parts.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            if overlap_tokens > 0:
                start = max(start + 1, end - overlap_tokens)
            else:
                start = end
        return parts
