"""Release validation recomputes what the index contains; a claim alone never passes.

Each test starts from a genuine staged release (Chroma store + index manifest + canonical corpus
manifest + descriptor that all agree) and damages exactly one thing, the way a wrong corpus, a
stale index, a hand-edited manifest or a modified store would damage a real deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest

from customer_claims_rag.exceptions import ReleasePostureError
from customer_claims_rag.release.index_integrity import recompute_store_identity
from customer_claims_rag.release.posture import (
    ResolvedReleaseTarget,
    load_release_posture_descriptor,
    resolve_release_target,
    validate_release_posture_for_production,
)
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.retrieval.index_identity import compute_chunk_payload_digest
from customer_claims_rag.retrieval.manifest import load_manifest
from tests.release_posture_helpers import (
    FAKE_MODEL,
    StagedRelease,
    make_test_chunks,
    stage_consistent_release,
    write_consistent_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
DOCS = ("01_service_overview", "02_delivery_rules")


def _release(tmp_path: Path) -> StagedRelease:
    return stage_consistent_release(tmp_path, document_ids=DOCS, chunks_per_document=2)


def _validate(staged: StagedRelease, *, with_store: bool = True):
    descriptor = load_release_posture_descriptor(staged.descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=staged.root)
    store = None
    if with_store:
        # the store the descriptor names, which a test may have pointed elsewhere
        store = ChromaVectorStore(
            index_dir=resolved.index_dir,
            collection_name="customer_claims",
            open_existing=True,
        )
    try:
        return validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=staged.root,
            vector_store=store,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        )
    finally:
        if store is not None:
            store.close()


def _collection(index_dir: Path):
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_collection("customer_claims")


def _edit_stored_text(index_dir: Path, chunk_id: str, text: str) -> None:
    """Change one stored document, keeping its vector (Chroma would otherwise re-embed)."""
    collection = _collection(index_dir)
    vector = collection.get(ids=[chunk_id], include=["embeddings"])["embeddings"][0]
    collection.update(ids=[chunk_id], documents=[text], embeddings=[list(vector)])


def _swap_in_index(staged: StagedRelease, chunks, name: str, **index_kwargs) -> Path:
    """Build another index and point the descriptor at it.

    A directory cannot be replaced in place on Windows while Chroma keeps its files open, and the
    descriptor (committed config) is what names the index anyway.
    """
    new_dir = staged.root / name
    write_consistent_index(new_dir, chunks, **index_kwargs)
    payload = json.loads(staged.descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["active"]["index_path"] = name
    staged.descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    return new_dir


def _rewrite_manifest(index_dir: Path, **fields) -> None:
    path = index_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(fields)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- the genuine release ----------------------------------------------------------------------


def test_a_genuine_release_validates_and_is_marked_store_recomputed(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    diagnostics = _validate(staged)
    assert diagnostics.integrity == "store_recomputed"
    assert diagnostics.chunk_count == 4
    assert diagnostics.document_count == 2


def test_recomputed_store_identity_equals_what_the_build_wrote(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    store = ChromaVectorStore(index_dir=staged.index_dir, collection_name="customer_claims", open_existing=True)
    try:
        identity = recompute_store_identity(store, embedding_model=FAKE_MODEL, rebuild_hint="rebuild")
    finally:
        store.close()
    manifest = load_manifest(staged.index_dir)
    assert identity.chunk_payload_digest == manifest.chunk_payload_digest == compute_chunk_payload_digest(staged.chunks)
    assert identity.corpus_fingerprint == manifest.corpus_fingerprint == staged.corpus_fingerprint
    assert identity.collection_content_digest == manifest.collection_content_digest
    assert identity.chunk_count == 4
    assert identity.document_ids == frozenset(DOCS)
    assert identity.vector_dimensions == frozenset({staged.vector_dimension})


# --- the store was changed after it was built --------------------------------------------------


def test_edited_stored_chunk_text_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    _edit_stored_text(staged.index_dir, staged.chunks[0].chunk_id, "silently edited text")
    with pytest.raises(ReleasePostureError, match="stored chunk payload digest mismatch"):
        _validate(staged)


def test_edited_stored_vector_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    collection = _collection(staged.index_dir)
    collection.update(ids=[staged.chunks[0].chunk_id], embeddings=[[0.5] * staged.vector_dimension])
    with pytest.raises(ReleasePostureError, match="stored collection content digest mismatch"):
        _validate(staged)


def test_edited_stored_metadata_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    _collection(staged.index_dir).update(
        ids=[staged.chunks[0].chunk_id],
        metadatas=[{"heading": "a heading nobody approved"}],
    )
    with pytest.raises(ReleasePostureError, match="stored chunk payload digest mismatch"):
        _validate(staged)


def test_an_added_chunk_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    extra = make_test_chunks(("01_service_overview",), 3)[2]
    store = ChromaVectorStore(index_dir=staged.index_dir, collection_name="customer_claims")
    store.add_chunks([extra], [[0.1] * staged.vector_dimension])
    store.close()
    with pytest.raises(ReleasePostureError, match="vector store chunk count mismatch"):
        _validate(staged)


def test_a_removed_chunk_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    _collection(staged.index_dir).delete(ids=[staged.chunks[0].chunk_id])
    with pytest.raises(ReleasePostureError, match="vector store chunk count mismatch"):
        _validate(staged)


def test_stored_vectors_of_another_dimension_are_detected_even_if_the_manifest_claims_the_right_one(
    tmp_path: Path,
) -> None:
    staged = _release(tmp_path)
    index_dir = _swap_in_index(staged, staged.chunks, "dim4-index", vector_dimension=4)
    _rewrite_manifest(index_dir, vector_dimension=staged.vector_dimension)
    with pytest.raises(ReleasePostureError, match="stored vector dimension mismatch"):
        _validate(staged)


# --- the manifest lies -------------------------------------------------------------------------


def _stale_index_with_forged_manifest(staged: StagedRelease, chunks) -> None:
    """Build the store from other chunks, then claim the approved corpus in the manifest."""
    index_dir = _swap_in_index(staged, chunks, "stale-index")
    _rewrite_manifest(
        index_dir,
        corpus_fingerprint=staged.corpus_fingerprint,
        chunk_payload_digest=staged.chunk_payload_digest,
    )


def test_a_stale_index_with_a_forged_manifest_passes_the_static_check_but_not_the_recompute(
    tmp_path: Path,
) -> None:
    """Why the store is re-read: the manifest alone can be made to say anything."""
    staged = _release(tmp_path)
    stale = [chunk.model_copy(update={"content": chunk.content + " (old wording)"}) for chunk in staged.chunks]
    _stale_index_with_forged_manifest(staged, stale)

    static = _validate(staged, with_store=False)
    assert static.integrity == "manifest_self_attestation_only"

    with pytest.raises(ReleasePostureError, match="stored chunk payload digest mismatch"):
        _validate(staged)


def test_an_index_of_another_corpus_with_a_forged_manifest_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    other = make_test_chunks(("03_order_changes_and_cancellations", "04_refund_policy"), 2)
    _stale_index_with_forged_manifest(staged, other)
    with pytest.raises(ReleasePostureError, match="supported document IDs mismatch"):
        _validate(staged)


def test_an_honest_index_of_another_corpus_is_detected_by_its_own_manifest(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    _swap_in_index(staged, make_test_chunks(("03_order_changes_and_cancellations",), 4), "other-index")
    with pytest.raises(ReleasePostureError, match="corpus fingerprint mismatch"):
        _validate(staged, with_store=False)


@pytest.mark.parametrize("digest", ["chunk_payload_digest", "embedding_digest", "collection_content_digest"])
def test_a_manifest_without_the_content_digests_is_refused(tmp_path: Path, digest: str) -> None:
    staged = _release(tmp_path)
    _rewrite_manifest(staged.index_dir, **{digest: None})
    with pytest.raises(ReleasePostureError, match=f"index manifest lacks {digest}"):
        _validate(staged, with_store=False)


def test_a_manifest_edited_to_fit_a_modified_store_is_still_detected(tmp_path: Path) -> None:
    """Hand-editing collection_content_digest hides a store edit from that check, not from the pins."""
    staged = _release(tmp_path)
    _edit_stored_text(staged.index_dir, staged.chunks[0].chunk_id, "edited")
    store = ChromaVectorStore(index_dir=staged.index_dir, collection_name="customer_claims", open_existing=True)
    try:
        edited = recompute_store_identity(store, embedding_model=FAKE_MODEL, rebuild_hint="")
    finally:
        store.close()
    _rewrite_manifest(staged.index_dir, collection_content_digest=edited.collection_content_digest)
    with pytest.raises(ReleasePostureError, match="stored chunk payload digest mismatch"):
        _validate(staged)


def test_stored_records_from_an_older_metadata_schema_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index written before ``version`` was stored cannot be verified and must be rebuilt."""
    from customer_claims_rag.retrieval import metadata_mapper

    real = metadata_mapper.chunk_to_vector_metadata

    def without_version(chunk):
        return {key: value for key, value in real(chunk).items() if key != "version"}

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.adapters.chroma_store.chunk_to_vector_metadata",
        without_version,
    )
    staged = _release(tmp_path)
    with pytest.raises(ReleasePostureError, match="incomplete or malformed.*version"):
        _validate(staged)


# --- committed pins ---------------------------------------------------------------------------


def test_tampering_with_the_canonical_corpus_manifest_is_detected(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    payload = json.loads(staged.corpus_manifest_path.read_text(encoding="utf-8"))
    payload["expected"]["chunk_payload_digest"] = "1" * 64
    staged.corpus_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleasePostureError, match="chunk payload digest mismatch"):
        _validate(staged, with_store=False)


def test_the_embedding_model_name_is_bound_into_the_fingerprint(tmp_path: Path) -> None:
    """An index claiming another model fails on the pinned fingerprint, not only on a string."""
    staged = _release(tmp_path)
    other_model_fingerprint = compute_corpus_fingerprint(staged.chunks, embedding_model="another-model")
    assert other_model_fingerprint != staged.corpus_fingerprint
    _rewrite_manifest(staged.index_dir, corpus_fingerprint=other_model_fingerprint)
    with pytest.raises(ReleasePostureError, match="corpus fingerprint mismatch"):
        _validate(staged, with_store=False)


def test_resolved_target_carries_the_pins_from_the_corpus_manifest_only(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    descriptor = load_release_posture_descriptor(staged.descriptor_path)
    resolved: ResolvedReleaseTarget = resolve_release_target(descriptor, "active", project_root=staged.root)
    assert resolved.expected_corpus_fingerprint == staged.corpus_fingerprint
    assert resolved.expected_chunk_payload_digest == staged.chunk_payload_digest
    assert resolved.supported_document_ids == tuple(sorted(DOCS))
    assert resolved.expected_chunk_count == 4


def test_a_fingerprint_claim_that_agrees_with_a_wrong_pin_is_caught_by_the_recompute(tmp_path: Path) -> None:
    """Same chunks (payload digest agrees), but the pin and the manifest both name a fingerprint
    the stored records do not hash to, e.g. an index fingerprinted for another model name."""
    staged = _release(tmp_path)
    wrong = "e" * 64
    payload = json.loads(staged.corpus_manifest_path.read_text(encoding="utf-8"))
    payload["expected"]["corpus_fingerprint"] = wrong
    staged.corpus_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _rewrite_manifest(staged.index_dir, corpus_fingerprint=wrong)

    assert _validate(staged, with_store=False).integrity == "manifest_self_attestation_only"
    with pytest.raises(ReleasePostureError, match="stored corpus fingerprint mismatch"):
        _validate(staged)
