"""Shared helpers for release posture tests."""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.release.posture import ResolvedReleaseTarget

PRODUCTION_FROZEN_CONFIG_HASH = (
    "ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048"
)


def write_test_release_descriptor(
    path: Path,
    *,
    index_path_relative: str,
    corpus_fingerprint: str,
    chunk_count: int,
    document_count: int,
    supported_document_ids: list[str],
    target_name: str = "active",
    status: str = "selected_production_release",
    expected_frozen_config_hash: str = PRODUCTION_FROZEN_CONFIG_HASH,
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "release_posture_id": "test-release-posture",
        "default_target": target_name,
        "limitations_doc": "docs/06_release_posture.md",
        "frozen_retrieval_config_path": "configs/retrieval/vector_pool_expansion_v1.json",
        "expected_frozen_config_hash": expected_frozen_config_hash,
        "targets": {
            target_name: {
                "status": status,
                "index_path": index_path_relative,
                "expected_corpus_fingerprint": corpus_fingerprint,
                "expected_chunk_count": chunk_count,
                "expected_document_count": document_count,
                "supported_document_ids": supported_document_ids,
                "collection_name": "customer_claims",
                "embedding_model": "fake-embedding-model",
                "vector_dimension": 8,
            },
            "rollback": {
                "status": "emergency_rollback_archive_only",
                "index_path": "data/04_index",
                "expected_corpus_fingerprint": "rollback-fingerprint",
                "expected_chunk_count": 1,
                "expected_document_count": 1,
                "supported_document_ids": ["01_service_overview"],
                "collection_name": "customer_claims",
                "embedding_model": "fake-embedding-model",
                "vector_dimension": 8,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolved_release_target_for_index(
    index_dir: Path,
    *,
    project_root: Path,
    corpus_fingerprint: str,
    chunk_count: int,
    document_count: int,
    supported_document_ids: tuple[str, ...],
    target_name: str = "active",
    status: str = "selected_production_release",
    embedding_model: str = "fake-embedding-model",
    vector_dimension: int = 8,
) -> ResolvedReleaseTarget:
    relative = index_dir.resolve().relative_to(project_root.resolve()).as_posix()
    return ResolvedReleaseTarget(
        target_name=target_name,
        status=status,
        index_dir=index_dir.resolve(),
        index_path_relative=relative,
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_chunk_count=chunk_count,
        expected_document_count=document_count,
        supported_document_ids=supported_document_ids,
        collection_name="customer_claims",
        embedding_model=embedding_model,
        vector_dimension=vector_dimension,
    )
