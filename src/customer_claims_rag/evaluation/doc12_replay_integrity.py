"""Deterministic replay and stability checks for doc12 threat experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.doc08_atomic_contract import DOC12_DOCUMENT_ID
from customer_claims_rag.evaluation.doc12_threat_atomic_models import ReplayStabilityResult
from customer_claims_rag.evaluation.extension_parser import load_extension_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import rerank_to_final_top_k
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import (
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
)
from customer_claims_rag.retrieval.experiment_embedding_cache import experiment_cache_path
from customer_claims_rag.retrieval.experiment_index_publish import (
    experiment_staging_dir,
    publish_staged_index,
)
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, load_reranker_config
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings
from customer_claims_rag.token_counter import TiktokenCounter


@dataclass(frozen=True)
class E008Trace:
    vector_top_ids: tuple[str, ...]
    vector_doc12_ranks: tuple[int, ...]
    pool_doc12_ranks: tuple[int, ...]
    final_doc12_rank: int | None
    primary_hit_at_4: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector_top_ids": list(self.vector_top_ids),
            "vector_doc12_ranks": list(self.vector_doc12_ranks),
            "pool_doc12_ranks": list(self.pool_doc12_ranks),
            "final_doc12_rank": self.final_doc12_rank,
            "primary_hit_at_4": self.primary_hit_at_4,
        }


def _trace_digest(traces: list[E008Trace]) -> str:
    payload = [trace.as_dict() for trace in traces]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def trace_e008_case(
    *,
    index_dir: Path,
    project_root: Path,
    settings: RetrievalSettings | None = None,
) -> E008Trace:
    resolved_settings = settings or RetrievalSettings.from_env()
    cases = load_extension_corpus(
        questions_path=project_root / "tests/extension/doc08_atomic_extension_v1_questions.md",
        expected_path=project_root / "tests/extension/doc08_atomic_extension_v1_expected.md",
        expected_ids=["E008"],
    )
    query = cases[0].query
    embedding = create_embedding_provider(
        model_name=resolved_settings.embedding_model,
        api_key=resolved_settings.openai_api_key,
    )
    store = create_vector_store(
        index_dir=index_dir,
        collection_name=resolved_settings.collection_name,
    )
    retriever = BaselineRetriever(
        embedding_provider=embedding,
        vector_store=store,
        index_dir=index_dir,
        top_k=48,
        fetch_k=48,
        similarity_threshold=0.0,
    )
    reranker = SourceAuthorityV1Reranker(
        load_reranker_config(project_root / "configs/reranking/source_authority_v1.json")
    )
    response = retriever.search(query)
    vector_hits = list(response.results)
    pool, _ = apply_per_document_cap(vector_hits, fetch_k=48, pool_k=36, per_document_cap=4)
    final, _ = rerank_to_final_top_k(reranker, query, pool, final_top_k=12)
    store.close()
    vector_doc12 = tuple(hit.rank for hit in vector_hits if hit.document_id == DOC12_DOCUMENT_ID)
    pool_doc12 = tuple(item.rank for item in pool if item.document_id == DOC12_DOCUMENT_ID)
    final_doc12_rank = next(
        (index + 1 for index, item in enumerate(final) if item.document_id == DOC12_DOCUMENT_ID),
        None,
    )
    return E008Trace(
        vector_top_ids=tuple(hit.chunk_id for hit in vector_hits[:48]),
        vector_doc12_ranks=vector_doc12,
        pool_doc12_ranks=pool_doc12,
        final_doc12_rank=final_doc12_rank,
        primary_hit_at_4=final_doc12_rank is not None and final_doc12_rank <= 4,
    )


def run_repeated_query_stability(
    *,
    index_dir: Path,
    project_root: Path,
    runs: int = 10,
) -> dict[str, Any]:
    traces = [trace_e008_case(index_dir=index_dir, project_root=project_root) for _ in range(runs)]
    digest = _trace_digest(traces)
    unique = {json.dumps(trace.as_dict(), sort_keys=True) for trace in traces}
    return {
        "runs": runs,
        "all_identical": len(unique) == 1,
        "trace_digest": digest,
        "sample": traces[0].as_dict(),
    }


def build_candidate_index_to_dir(
    *,
    config: dict,
    canonical_dir: Path,
    index_dir: Path,
    project_root: Path,
    settings: RetrievalSettings | None = None,
    run_id: str | None = None,
    shared_cache_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or RetrievalSettings.from_env()
    cache_home = shared_cache_dir or index_dir
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)

    staging_dir, build_run_id = experiment_staging_dir(index_dir, run_id=run_id)
    doc08_overlay = project_root / config["baseline_corpus"]["doc08_overlay_path"]
    doc12_overlay = project_root / config["candidate_overlay"]["overlay_path"]
    overlay = build_doc12_experimental_corpora(
        canonical_dir=canonical_dir,
        doc08_overlay_path=doc08_overlay,
        doc12_overlay_path=doc12_overlay,
        permitted_root=project_root,
        staging_parent=staging_dir / "corpus_stage",
    )
    try:
        builder = CorpusBuilder(permitted_root=project_root, token_counter=TiktokenCounter())
        embedding = create_embedding_provider(
            model_name=resolved_settings.embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        store = create_vector_store(
            index_dir=staging_dir,
            collection_name=resolved_settings.collection_name,
        )
        index_builder = IndexBuilder(
            corpus_builder=builder,
            embedding_provider=embedding,
            vector_store=store,
            index_dir=staging_dir,
            batch_size=resolved_settings.embedding_batch_size,
        )
        report = index_builder.build_from_chunks(
            documents=[],
            chunks=overlay.candidate_chunks,
            rebuild=True,
            embedding_cache_path=experiment_cache_path(cache_home),
            build_run_id=build_run_id,
        )
        store.close()
        publish_staged_index(staging_dir, index_dir)
        manifest = load_manifest(index_dir)
        trace = trace_e008_case(index_dir=index_dir, project_root=project_root, settings=resolved_settings)
        return {
            "build_run_id": build_run_id,
            "corpus_fingerprint": report.fingerprint,
            "chunk_payload_digest": manifest.chunk_payload_digest,
            "embedding_digest": manifest.embedding_digest,
            "collection_content_digest": manifest.collection_content_digest,
            "index_fingerprint": manifest.corpus_fingerprint,
            "e008_trace": trace.as_dict(),
        }
    finally:
        cleanup_overlay_temp_dir(overlay.temp_input_dir)


def run_independent_rebuild_stability(
    *,
    config: dict,
    canonical_dir: Path,
    parent_dir: Path,
    project_root: Path,
    cache_home: Path,
    builds: int = 3,
) -> dict[str, Any]:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label in ("A", "B", "C")[:builds]:
        index_dir = parent_dir / f"BUILD_{label}"
        results.append(
            build_candidate_index_to_dir(
                config=config,
                canonical_dir=canonical_dir,
                index_dir=index_dir,
                project_root=project_root,
                run_id=f"{label.lower()}-{uuid.uuid4().hex[:8]}",
                shared_cache_dir=cache_home,
            )
        )
    chunk_digests = {item["chunk_payload_digest"] for item in results}
    embedding_digests = {item["embedding_digest"] for item in results}
    collection_digests = {item["collection_content_digest"] for item in results}
    e008_traces = {json.dumps(item["e008_trace"], sort_keys=True) for item in results}
    return {
        "builds": results,
        "chunk_payload_identical": len(chunk_digests) == 1,
        "embedding_identical": len(embedding_digests) == 1,
        "collection_identical": len(collection_digests) == 1,
        "e008_identical": len(e008_traces) == 1,
        "all_identical": (
            len(chunk_digests) == 1
            and len(embedding_digests) == 1
            and len(collection_digests) == 1
            and len(e008_traces) == 1
        ),
    }


def run_repeated_full_evaluation_stability(
    *,
    evaluator,
    runs: int = 3,
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    for _ in range(runs):
        run = evaluator.evaluate()
        e008 = next(item for item in run.extension.case_diagnostics if item.case_id == "E008")
        snapshots.append(
            {
                "threat_doc12_hit_at_4": run.extension.candidate.threat_doc12_hit_at_4,
                "e008_final_rank_doc12": e008.candidate_final_rank_doc12,
                "e008_primary_hit_at_4": e008.candidate_primary_hit_at_4,
                "collection_content_digest": run.candidate_arm.collection_content_digest,
            }
        )
    unique = {json.dumps(item, sort_keys=True) for item in snapshots}
    return {
        "runs": runs,
        "all_identical": len(unique) == 1,
        "snapshots": snapshots,
    }


def build_replay_stability_result(
    *,
    index_dir: Path,
    project_root: Path,
    config: dict,
    canonical_dir: Path,
    evaluator,
    rebuild_parent: Path,
) -> ReplayStabilityResult:
    repeated_query = run_repeated_query_stability(
        index_dir=index_dir,
        project_root=project_root,
        runs=10,
    )
    repeated_full = run_repeated_full_evaluation_stability(evaluator=evaluator, runs=3)
    independent = run_independent_rebuild_stability(
        config=config,
        canonical_dir=canonical_dir,
        parent_dir=rebuild_parent,
        project_root=project_root,
        cache_home=index_dir,
        builds=3,
    )
    try:
        all_identical = (
            repeated_query["all_identical"]
            and repeated_full["all_identical"]
            and independent["all_identical"]
        )
        integrity_verdict = (
            "PASS — EXPERIMENT INTEGRITY REPAIRED"
            if all_identical
            else "FAIL — NONDETERMINISTIC EXPERIMENT"
        )
        return ReplayStabilityResult(
            repeated_query_runs=repeated_query,
            repeated_full_runs=repeated_full,
            independent_rebuilds=independent,
            all_identical=all_identical,
            integrity_verdict=integrity_verdict,
        )
    finally:
        shutil.rmtree(rebuild_parent, ignore_errors=True)
