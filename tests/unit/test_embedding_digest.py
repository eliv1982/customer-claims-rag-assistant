"""``embedding_digest`` serialization: half precision, not float64.

The release documents describe what this digest attests, so the serialization is pinned: a change
to it would silently invalidate every recorded ``embedding_digest`` (historical manifests, doc12
replay artifacts, embedding snapshots) and make the written description untrue again.
"""

from __future__ import annotations

import hashlib

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.retrieval.index_identity import compute_embedding_digest

# IEEE-754 binary16, big-endian: 1.0 -> 0x3c00, -2.0 -> 0xc000, 0.1 -> 0x2e66.
HALF_PRECISION_BYTES = bytes.fromhex("3c00c0002e66")


def test_embedding_digest_hashes_chunk_id_then_two_byte_half_precision_components() -> None:
    chunk = make_chunk_record(chunk_id="a::chunk-001")
    expected = hashlib.sha256(b"a::chunk-001\x00" + HALF_PRECISION_BYTES).hexdigest()
    assert compute_embedding_digest([chunk], [[1.0, -2.0, 0.1]]) == expected


def test_embedding_digest_cannot_see_changes_below_half_precision() -> None:
    chunk = make_chunk_record(chunk_id="a::chunk-001")
    base = compute_embedding_digest([chunk], [[1.0, -2.0, 0.1]])
    # float64 serialization would tell these apart; half precision (11 significant bits) cannot.
    assert compute_embedding_digest([chunk], [[1.0, -2.0, 0.1 + 1e-9]]) == base
    assert compute_embedding_digest([chunk], [[1.0, -2.0, 0.1005]]) != base
