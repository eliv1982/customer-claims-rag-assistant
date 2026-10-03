"""Local historical archives: the superseded Chroma indexes some experiments were measured on.

Two indexes may exist on a maintainer machine from before the canonical corpus: the
15-document expansion at ``data/04_index`` and the 10-document release v1 backup at
``data/04_index_backup_10docs_215chunks``. They are not production indexes and the release
posture no longer knows them; they matter only as evidence, and only the experiments' replay
tests (``test_hybrid_lexical_replay.py``, ``test_doc08_atomic_experiment.py``) read them.

What they must be is committed, in the frozen snapshot's manifest, so the expectations here are
read from it rather than repeated. The checks need the built Chroma indexes (made with live
OpenAI embeddings), so they stay ``local_artifact`` tests: absent -> skipped with the path,
present but different -> failed.
"""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.retrieval.manifest import load_manifest
from tests.local_artifacts import requires_local_artifacts

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = json.loads(
    (ROOT / "experiments" / "corpus" / "historical_pre_2d2_15doc_v1" / "snapshot_manifest.json").read_text(
        encoding="utf-8"
    )
)
EXPANDED_ARCHIVE = ROOT / "data" / "04_index"
RELEASE_V1_ARCHIVE = ROOT / "data" / "04_index_backup_10docs_215chunks"


def _assert_is_the_historical_index(index_dir: Path, identity_key: str) -> None:
    identity = SNAPSHOT["identities"][identity_key]
    manifest = load_manifest(index_dir)
    assert manifest.chunk_count == identity["chunk_count"]
    assert manifest.document_count == len(identity["document_ids"])
    assert manifest.corpus_fingerprint == identity["corpus_fingerprint"]
    if manifest.chunk_payload_digest is not None:
        assert manifest.chunk_payload_digest == identity["chunk_payload_digest"]


@requires_local_artifacts(
    EXPANDED_ARCHIVE / "manifest.json",
    why="the historical 15-document index (built with live OpenAI embeddings)",
)
def test_local_expanded_archive_is_the_historical_15_document_index() -> None:
    _assert_is_the_historical_index(EXPANDED_ARCHIVE, "expanded_15doc")


@requires_local_artifacts(
    RELEASE_V1_ARCHIVE / "manifest.json",
    why="the historical 10-document release v1 index (built with live OpenAI embeddings)",
)
def test_local_release_v1_archive_is_the_historical_10_document_index() -> None:
    _assert_is_the_historical_index(RELEASE_V1_ARCHIVE, "release_10doc_v1")
