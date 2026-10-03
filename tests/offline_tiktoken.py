"""Deterministic offline stand-in for tiktoken encodings (tests only).

``tiktoken.get_encoding("cl100k_base")`` downloads its vocabulary on first use,
so a cold cache plus no network turns every test that chunks text into a
failure. The default suite therefore swaps the encoding for ``OfflineEncoding``.

``OfflineEncoding`` is NOT cl100k: it counts one token per word (words longer
than 12 characters are cut every 12) and one per punctuation mark. It is
deterministic, local and exercises the real ``TiktokenCounter`` splitting code,
but its counts differ from the real vocabulary, so it is not evidence for
production chunk topology: the release corpus chunks into 282 chunks here (333
under cl100k), and the doc12 overlay experiment even inverts (doc12 22 -> 13
chunks here, 23 -> 26 under cl100k). Golden values that only hold under real
cl100k tokenization (chunk counts, corpus fingerprints) must be marked
``real_tiktoken``; they run in the real-tokenizer lane described in
``tests/tokenizer_lanes.py``.

Production code is untouched: ``TiktokenCounter`` still uses real tiktoken.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

_TOKEN_PATTERN = re.compile(r"\w{1,12}|[^\w\s]")


class OfflineEncoding:
    """Minimal ``tiktoken.Encoding`` look-alike; ``TiktokenCounter`` only calls ``encode``."""

    name = "offline-test-encoding"

    def encode(self, text: str, **_kwargs: object) -> list[str]:
        return _TOKEN_PATTERN.findall(text)


@contextmanager
def offline_tiktoken() -> Iterator[None]:
    """Make every *known* tiktoken encoding name resolve to ``OfflineEncoding``.

    Unknown names still reach the real ``get_encoding``, which raises
    ``ValueError`` without touching the network.
    """
    import tiktoken

    real_get_encoding = tiktoken.get_encoding
    known_names = set(tiktoken.list_encoding_names())

    def get_encoding(encoding_name: str) -> object:
        if encoding_name in known_names:
            return OfflineEncoding()
        return real_get_encoding(encoding_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tiktoken, "get_encoding", get_encoding)
        yield
