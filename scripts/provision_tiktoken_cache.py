"""Provision the real ``cl100k_base`` vocabulary into an explicit tiktoken cache (CI setup step).

The vocabulary is deliberately not committed to the repository. The real-tokenizer test lane
(``python -m pytest --real-tiktoken -m real_tiktoken``) never downloads it; it needs the cache to
exist already. This script is the one place where that download happens, in a setup step of its
own, before test execution (and before the test network guard is installed):

* ``TIKTOKEN_CACHE_DIR`` must be set to an absolute path. There is no fallback to tiktoken's
  default temp directory, so the cache location is always known and is the same one the lane reads.
* The download is tiktoken's own supported mechanism (``tiktoken.get_encoding``). tiktoken checks
  the cached file against the vocabulary's SHA-256 baked into the installed version: a cache hit
  needs no network, while a missing, truncated or tampered file is discarded and fetched again.
* Afterwards the loaded encoding is checked against known identity values, and the cache contents
  are listed with their digests so the CI log shows exactly what the lane will read.

Exit status: 0 provisioned and verified, 2 unusable cache directory setting, 1 anything else.
Needs no credentials and writes nothing outside the cache directory.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ENCODING = "cl100k_base"
EXPECTED_VOCABULARY_SIZE = 100277
# The example from tiktoken's own documentation, as cl100k_base encodes it.
EXPECTED_TOKENS = {"tiktoken is great!": [83, 1609, 5963, 374, 2294, 0]}
CACHE_DIR_VARIABLE = "TIKTOKEN_CACHE_DIR"


def _cache_files(cache_dir: Path) -> dict[str, str]:
    if not cache_dir.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(cache_dir.iterdir())
        if path.is_file()
    }


def main() -> int:
    setting = os.environ.get(CACHE_DIR_VARIABLE, "")
    if not setting or not Path(setting).is_absolute():
        # An empty value would silently disable tiktoken's cache; a relative one depends on the cwd.
        print(f"{CACHE_DIR_VARIABLE} must be set to an absolute path (got {setting!r})", file=sys.stderr)
        return 2
    cache_dir = Path(setting)
    before = _cache_files(cache_dir)

    import tiktoken

    try:
        encoding = tiktoken.get_encoding(ENCODING)
    except Exception as exc:  # download or hash-check failure: say what failed, without a traceback
        print(f"could not provision {ENCODING} into {cache_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if encoding.name != ENCODING:
        print(f"expected encoding {ENCODING!r}, tiktoken returned {encoding.name!r}", file=sys.stderr)
        return 1
    if encoding.n_vocab != EXPECTED_VOCABULARY_SIZE:
        print(f"vocabulary size {encoding.n_vocab}, expected {EXPECTED_VOCABULARY_SIZE}", file=sys.stderr)
        return 1
    for text, tokens in EXPECTED_TOKENS.items():
        if encoding.encode(text) != tokens:
            print(f"{ENCODING} encodes {text!r} differently from the reference", file=sys.stderr)
            return 1

    after = _cache_files(cache_dir)
    if not after:
        print(f"{ENCODING} loaded but left no file in {cache_dir}: caching is not in effect", file=sys.stderr)
        return 1

    print(f"encoding={encoding.name} vocabulary_size={encoding.n_vocab} tiktoken={tiktoken.__version__}")
    print(f"cache_dir={cache_dir}")
    print(f"vocabulary_source={'cache' if after == before else 'download'}")
    for name, digest in after.items():
        print(f"cache_file={name} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
