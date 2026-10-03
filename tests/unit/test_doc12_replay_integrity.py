"""Tests for doc12 experiment replay integrity and index identity."""

from __future__ import annotations

import json
import re
from pathlib import Path

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.experiment_embedding_cache import (
    experiment_cache_path,
    resolve_experiment_embeddings,
)
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest,
    compute_embedding_digest,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.token_counter import TiktokenCounter
from tests.retrieval_helpers import make_chunk_record


ROOT = project_root()
CONFIG = ROOT / "configs/experiments/doc12_threat_atomic_units_v1.json"


def _build_fake_index(tmp_path: Path) -> Path:
    index_dir = tmp_path / "index"
    chunks = [
        make_chunk_record(chunk_id="01_service_overview::chunk-001", content="alpha policy"),
        make_chunk_record(chunk_id="12_staff_safety_and_threat_handling::chunk-001", content="threat policy"),
    ]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    builder = IndexBuilder(
        corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        batch_size=2,
    )
    builder.build_from_chunks(documents=[], chunks=chunks, rebuild=True)
    store.close()
    return index_dir


def test_rebuild_starts_from_empty_collection(tmp_path: Path) -> None:
    index_dir = _build_fake_index(tmp_path)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    assert store.count() == 2
    store.recreate_collection(embedding_dimension=8)
    assert store.count() == 0
    assert store.list_chunk_ids() == []
    store.close()


def test_expected_ids_equal_collection_ids(tmp_path: Path) -> None:
    index_dir = _build_fake_index(tmp_path)
    manifest = load_manifest(index_dir)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    assert set(store.list_chunk_ids()) == {
        "01_service_overview::chunk-001",
        "12_staff_safety_and_threat_handling::chunk-001",
    }
    assert manifest.chunk_count == store.count()
    store.close()


def test_chunk_and_embedding_digests_change_with_overlay(tmp_path: Path) -> None:
    chunks_a = [make_chunk_record(chunk_id="doc::1", content="threat alpha")]
    chunks_b = [make_chunk_record(chunk_id="doc::1", content="threat beta")]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    assert compute_chunk_payload_digest(chunks_a) != compute_chunk_payload_digest(chunks_b)
    vectors_a = provider.embed_documents([chunk.content for chunk in chunks_a])
    vectors_b = provider.embed_documents([chunk.content for chunk in chunks_b])
    assert compute_embedding_digest(chunks_a, vectors_a) != compute_embedding_digest(
        chunks_b,
        vectors_b,
    )


def test_experiment_embedding_cache_is_stable_across_rebuilds(tmp_path: Path) -> None:
    chunks = [
        make_chunk_record(chunk_id="doc::1", content="threat alpha"),
        make_chunk_record(chunk_id="doc::2", content="support operator"),
    ]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    cache_path = experiment_cache_path(tmp_path / "candidate")
    first = resolve_experiment_embeddings(chunks, embedding_provider=provider, cache_path=cache_path)
    second = resolve_experiment_embeddings(chunks, embedding_provider=provider, cache_path=cache_path)
    assert first == second
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["chunk_payload_digest"] == compute_chunk_payload_digest(chunks)


def test_three_temp_roots_same_collection_digest(tmp_path: Path) -> None:
    chunks = [
        make_chunk_record(chunk_id="doc::1", content="threat alpha"),
        make_chunk_record(chunk_id="doc::2", content="support operator"),
    ]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    cache_path = experiment_cache_path(tmp_path / "shared_cache")
    embeddings = resolve_experiment_embeddings(
        chunks,
        embedding_provider=provider,
        cache_path=cache_path,
    )
    digests: list[str] = []
    for label in ("A", "B", "C"):
        index_dir = tmp_path / f"BUILD_{label}"
        store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
        builder = IndexBuilder(
            corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
            embedding_provider=provider,
            vector_store=store,
            index_dir=index_dir,
            batch_size=2,
        )
        builder.build_from_chunks(
            documents=[],
            chunks=chunks,
            rebuild=True,
            embedding_cache_path=cache_path,
        )
        manifest = load_manifest(index_dir)
        digests.append(manifest.collection_content_digest or "")
        store.close()
    assert len({digest for digest in digests if digest}) == 1
    assert manifest.embedding_digest == compute_embedding_digest(chunks, embeddings)


def test_repeated_query_trace_stable_on_fixed_index(tmp_path: Path) -> None:
    chunks = [make_chunk_record(chunk_id="doc::1", content="threat alpha")]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    index_dir = tmp_path / "index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    IndexBuilder(
        corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
    ).build_from_chunks(documents=[], chunks=chunks, rebuild=True)
    retriever = BaselineRetriever(
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=4,
        fetch_k=4,
        similarity_threshold=0.0,
    )
    traces = [
        tuple(item.chunk_id for item in retriever.search("threat alpha").results)
        for _ in range(5)
    ]
    store.close()
    assert len({trace for trace in traces}) == 1


def test_retriever_backend_rank_tie_break_is_stable(tmp_path: Path) -> None:
    chunks = [
        make_chunk_record(chunk_id="a::1", content="same topic"),
        make_chunk_record(chunk_id="b::1", content="same topic"),
    ]
    provider = FakeEmbeddingProvider(vector_dimension=8)
    index_dir = tmp_path / "index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    builder = IndexBuilder(
        corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
    )
    builder.build_from_chunks(documents=[], chunks=chunks, rebuild=True)
    retriever = BaselineRetriever(
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=2,
        fetch_k=2,
        similarity_threshold=0.0,
    )
    first = [item.chunk_id for item in retriever.search("same topic").results]
    second = [item.chunk_id for item in retriever.search("same topic").results]
    assert first == second
    store.close()


def test_doc12_artifact_paths_are_repo_relative() -> None:
    # Tracked artifact: a missing or outdated file is a failure, not a skip.
    artifact = ROOT / "data/05_evaluation/doc12_threat_atomic_units_v1.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload.get("replay_integrity") is not None, "artifact predates replay integrity repair"
    for arm_key in ("baseline_arm", "candidate_arm"):
        index_dir = payload[arm_key]["index_dir"]
        assert not re.match(r"^[A-Za-z]:\\", index_dir)
        assert "Cursor_Projects" not in index_dir
    assert not re.match(r"^[A-Za-z]:\\", payload["reference_artifact_path"])
    replay = payload["replay_integrity"]
    frozen = replay.get("frozen_snapshot_replay") or {}
    assert frozen.get("authoritative") is False
    exact = replay.get("exact_replay") or {}
    assert exact, "artifact predates R3 exact evaluation oracle"
    assert exact.get("authoritative") is True
    ann = replay.get("ann_robustness") or {}
    if ann:
        assert ann.get("authoritative") is False
    live = replay.get("live_provider_robustness")
    if live is not None:
        assert live.get("authoritative") is False


def test_config_declares_embedding_snapshot_paths() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["candidate_index_dir"] == "data/04_index_experiments/doc12_threat_atomic_units_v1"
    snapshot = config.get("embedding_snapshot") or {}
    assert snapshot["candidate_snapshot_path"].endswith("doc12_threat_atomic_units_v1.npz")
    assert snapshot["candidate_manifest_path"].endswith("doc12_threat_atomic_units_v1.manifest.json")
    assert snapshot["baseline_snapshot_path"].endswith("doc08_atomic_risk_units_v1.npz")
    assert snapshot["baseline_manifest_path"].endswith("doc08_atomic_risk_units_v1.manifest.json")
