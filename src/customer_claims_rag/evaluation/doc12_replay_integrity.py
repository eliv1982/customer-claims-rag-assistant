"""Frozen-snapshot replay and live-provider robustness checks for doc12 threat experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.doc08_atomic_contract import DOC12_DOCUMENT_ID
from customer_claims_rag.evaluation.doc12_threat_atomic_contract import repo_relative_path
from customer_claims_rag.evaluation.doc12_ann_robustness import digest_exact_run, run_ann_robustness_matrix
from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    AnnRebuildStability,
    EnvironmentProvenance,
    ExactReplayResult,
    FrozenSnapshotReplayResult,
    LiveProviderRobustnessResult,
    ReplayIntegrityResult,
    ReplayIntegrityVerdict,
)
from customer_claims_rag.evaluation.environment_provenance import capture_environment_provenance
from customer_claims_rag.evaluation.exact_frozen_arm import (
    ExactFrozenArm,
    copy_snapshot_to_temp_root,
    load_exact_frozen_arm,
)
from customer_claims_rag.evaluation.extension_parser import load_extension_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import rerank_to_final_top_k
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import (
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
)
from customer_claims_rag.retrieval.deterministic_search import deterministic_similarity_hits
from customer_claims_rag.retrieval.embedding_snapshot import load_snapshot_manifest
from customer_claims_rag.retrieval.experiment_index_publish import (
    experiment_staging_dir,
    publish_staged_index,
)
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, load_reranker_config
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval.snapshot_embedding_provider import SnapshotEmbeddingProvider
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
    deterministic: bool = False,
    query_vector: list[float] | None = None,
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
    reranker = SourceAuthorityV1Reranker(
        load_reranker_config(project_root / "configs/reranking/source_authority_v1.json")
    )
    resolved_query_vector = query_vector or embedding.embed_query(query)

    if deterministic:
        records = store.export_collection_records()
        ranked_hits = deterministic_similarity_hits(
            records,
            query_vector=resolved_query_vector,
            k=48,
        )
        from customer_claims_rag.retrieval.models import SearchResult

        search_results = [
            SearchResult(
                rank=index + 1,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                content=hit.content,
                source_path=hit.source_path,
                chunk_type=hit.chunk_type,
                topic=hit.topic,
                risk_level=hit.risk_level,
                heading=hit.heading,
                heading_path=hit.heading_path,
                section=hit.section,
                subsection=hit.subsection,
                similarity=hit.similarity,
                distance=hit.distance,
            )
            for index, hit in enumerate(ranked_hits)
        ]
    else:
        retriever = BaselineRetriever(
            embedding_provider=embedding,
            vector_store=store,
            index_dir=index_dir,
            top_k=48,
            fetch_k=48,
            similarity_threshold=0.0,
        )
        search_results = list(retriever.search(query).results)

    store.close()
    pool, _ = apply_per_document_cap(search_results, fetch_k=48, pool_k=36, per_document_cap=4)
    final, _ = rerank_to_final_top_k(reranker, query, pool, final_top_k=12)
    vector_doc12 = tuple(
        hit.rank for hit in search_results if hit.document_id == DOC12_DOCUMENT_ID
    )
    pool_doc12 = tuple(item.rank for item in pool if item.document_id == DOC12_DOCUMENT_ID)
    final_doc12_rank = next(
        (index + 1 for index, item in enumerate(final) if item.document_id == DOC12_DOCUMENT_ID),
        None,
    )
    return E008Trace(
        vector_top_ids=tuple(hit.chunk_id for hit in search_results[:48]),
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
    embedding_snapshot: Path | None = None,
    embedding_snapshot_manifest: Path | None = None,
    live_provider: bool = False,
    e008_query_vector: list[float] | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or RetrievalSettings.from_env()
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

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
        query_provider = create_embedding_provider(
            model_name=resolved_settings.embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        if embedding_snapshot is not None:
            embedding = SnapshotEmbeddingProvider(query_provider=query_provider)
            resolved_snapshot = (
                embedding_snapshot
                if embedding_snapshot.is_absolute()
                else project_root / embedding_snapshot
            )
            if embedding_snapshot_manifest is None:
                resolved_manifest = resolved_snapshot.with_suffix(".manifest.json")
            else:
                resolved_manifest = (
                    embedding_snapshot_manifest
                    if embedding_snapshot_manifest.is_absolute()
                    else project_root / embedding_snapshot_manifest
                )
        else:
            embedding = query_provider
            resolved_snapshot = None
            resolved_manifest = None

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
            embedding_cache_path=None,
            embedding_snapshot_path=resolved_snapshot,
            embedding_snapshot_manifest=resolved_manifest,
            build_run_id=build_run_id,
        )
        store.close()
        publish_staged_index(staging_dir, index_dir)
        manifest = load_manifest(index_dir)
        if manifest.chunk_count is None:
            raise RuntimeError(f"index manifest incomplete after publish: {index_dir}")
        trace = trace_e008_case(
            index_dir=index_dir,
            project_root=project_root,
            settings=resolved_settings,
            deterministic=resolved_snapshot is not None,
            query_vector=e008_query_vector,
        )
        return {
            "build_run_id": build_run_id,
            "corpus_fingerprint": report.fingerprint,
            "chunk_payload_digest": manifest.chunk_payload_digest,
            "embedding_digest": manifest.embedding_digest,
            "collection_content_digest": manifest.collection_content_digest,
            "index_fingerprint": manifest.corpus_fingerprint,
            "e008_trace": trace.as_dict(),
            "live_provider": live_provider,
        }
    finally:
        cleanup_overlay_temp_dir(overlay.temp_input_dir)


def run_frozen_snapshot_replay(
    *,
    config: dict,
    canonical_dir: Path,
    parent_dir: Path,
    project_root: Path,
    embedding_snapshot: Path,
    embedding_snapshot_manifest: Path,
    evaluator,
    builds: int = 3,
) -> FrozenSnapshotReplayResult:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    index_dir_for_queries = parent_dir / "AUTHORITATIVE"
    cases = load_extension_corpus(
        questions_path=project_root / "tests/extension/doc08_atomic_extension_v1_questions.md",
        expected_path=project_root / "tests/extension/doc08_atomic_extension_v1_expected.md",
        expected_ids=["E008"],
    )
    settings = RetrievalSettings.from_env()
    e008_query_vector = create_embedding_provider(
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
    ).embed_query(cases[0].query)

    results = []
    for label in ("A", "B", "C")[:builds]:
        index_dir = parent_dir / f"BUILD_{label}"
        results.append(
            build_candidate_index_to_dir(
                config=config,
                canonical_dir=canonical_dir,
                index_dir=index_dir,
                project_root=project_root,
                run_id=f"frozen-{label.lower()}-{uuid.uuid4().hex[:8]}",
                embedding_snapshot=embedding_snapshot,
                embedding_snapshot_manifest=embedding_snapshot_manifest,
                live_provider=False,
                e008_query_vector=e008_query_vector,
            )
        )
    shutil.copytree(parent_dir / "BUILD_A", index_dir_for_queries, dirs_exist_ok=True)

    repeated_query = run_repeated_query_stability(
        index_dir=index_dir_for_queries,
        project_root=project_root,
        runs=10,
    )
    evaluator.replace_candidate_index(parent_dir / "BUILD_A")
    repeated_full = run_repeated_full_evaluation_stability(evaluator=evaluator, runs=3)

    chunk_digests = {item["chunk_payload_digest"] for item in results}
    embedding_digests = {item["embedding_digest"] for item in results}
    collection_digests = {item["collection_content_digest"] for item in results}
    e008_traces = {json.dumps(item["e008_trace"], sort_keys=True) for item in results}

    return FrozenSnapshotReplayResult(
        runs=builds,
        builds=results,
        chunk_payload_identical=len(chunk_digests) == 1,
        embedding_identical=len(embedding_digests) == 1,
        collection_identical=len(collection_digests) == 1,
        e008_identical=len(e008_traces) == 1,
        all_identical=(
            len(chunk_digests) == 1
            and len(embedding_digests) == 1
            and len(collection_digests) == 1
            and len(e008_traces) == 1
            and repeated_query["all_identical"]
            and repeated_full["all_identical"]
        ),
        repeated_query_runs=repeated_query,
        repeated_full_runs=repeated_full,
        authoritative=False,
    )


def run_live_provider_robustness(
    *,
    config: dict,
    canonical_dir: Path,
    parent_dir: Path,
    project_root: Path,
    evaluator,
    runs: int = 9,
    original_candidate_index: Path,
) -> LiveProviderRobustnessResult:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for index in range(runs):
        index_dir = parent_dir / f"LIVE_{index + 1:02d}"
        build = build_candidate_index_to_dir(
            config=config,
            canonical_dir=canonical_dir,
            index_dir=index_dir,
            project_root=project_root,
            run_id=f"live-{index + 1}-{uuid.uuid4().hex[:8]}",
            live_provider=True,
        )
        evaluator.replace_candidate_index(index_dir)
        run = evaluator.evaluate()
        e008 = next(item for item in run.extension.case_diagnostics if item.case_id == "E008")
        h005 = next(item for item in run.holdout.case_diagnostics if item.case_id == "H005")
        summaries.append(
            {
                "run": index + 1,
                "embedding_digest": build["embedding_digest"],
                "e008_final_rank_doc12": e008.candidate_final_rank_doc12,
                "e008_primary_hit_at_4": e008.candidate_primary_hit_at_4,
                "h005_final_rank_doc12": h005.candidate_final_rank_doc12,
                "h005_primary_hit_at_4": h005.candidate_primary_hit_at_4,
                "experiment_verdict": run.verdict,
                "frozen_aggregate_primary_hit_at_4": run.frozen_candidate_metrics.primary_source_hit_rate_at_4,
                "extension_threat_doc12_hit_at_4": run.extension.candidate.threat_doc12_hit_at_4,
                "holdout_negative_doc12_top4": run.holdout.candidate.negative_doc12_top4,
            }
        )

    evaluator.replace_candidate_index(original_candidate_index)

    unique_digests = {item["embedding_digest"] for item in summaries}
    e008_pass = sum(1 for item in summaries if item["e008_primary_hit_at_4"])
    e008_fail = len(summaries) - e008_pass
    rejected_count = sum(1 for item in summaries if item["experiment_verdict"] == "REJECTED")

    e008_ranks = sorted({item["e008_final_rank_doc12"] for item in summaries})
    threat_totals = sorted({item["extension_threat_doc12_hit_at_4"] for item in summaries})

    return LiveProviderRobustnessResult(
        runs=runs,
        unique_embedding_digests=len(unique_digests),
        e008_hit4_pass_count=e008_pass,
        e008_hit4_fail_count=e008_fail,
        experiment_verdict_rejected_count=rejected_count,
        observed_metric_variability={
            "sample_size": runs,
            "wording": "In this nine-run diagnostic sample...",
            "unique_e008_final_ranks": e008_ranks,
            "unique_extension_threat_hit_at_4": threat_totals,
            "unique_embedding_digests": len(unique_digests),
        },
        run_summaries=summaries,
        authoritative=False,
    )


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


def capture_pip_environment(*, project_root: Path) -> dict[str, Any]:
    """Backward-compatible wrapper; prefer capture_environment_provenance."""
    provenance = capture_environment_provenance(project_root)
    return provenance


def run_exact_replay_matrix(
    *,
    evaluator,
    exact_baseline_arm: ExactFrozenArm,
    exact_candidate_arm: ExactFrozenArm,
    baseline_npz: Path,
    baseline_manifest: Path,
    candidate_npz: Path,
    candidate_manifest: Path,
    project_root: Path,
    canonical_dir: Path,
    parent_dir: Path,
    runs: int = 3,
) -> ExactReplayResult:
    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    full_digests: list[str] = []
    for _ in range(runs):
        run = evaluator.evaluate()
        full_digests.append(digest_exact_run(run))

    loader_digests: list[str] = []
    for label in ("ROOT_A", "ROOT_B", "ROOT_C"):
        temp_root = parent_dir / label
        copy_snapshot_to_temp_root(
            npz_path=baseline_npz,
            manifest_path=baseline_manifest,
            temp_root=temp_root / "baseline",
        )
        copy_snapshot_to_temp_root(
            npz_path=candidate_npz,
            manifest_path=candidate_manifest,
            temp_root=temp_root / "candidate",
        )
        baseline_copy_npz, baseline_copy_manifest = copy_snapshot_to_temp_root(
            npz_path=baseline_npz,
            manifest_path=baseline_manifest,
            temp_root=temp_root / "reload_baseline",
        )
        candidate_copy_npz, candidate_copy_manifest = copy_snapshot_to_temp_root(
            npz_path=candidate_npz,
            manifest_path=candidate_manifest,
            temp_root=temp_root / "reload_candidate",
        )
        from customer_claims_rag.ingestion.corpus_overlay import (
            build_doc12_experimental_corpora,
            cleanup_overlay_temp_dir,
        )

        overlay = build_doc12_experimental_corpora(
            canonical_dir=canonical_dir,
            doc08_overlay_path=project_root
            / evaluator.config["baseline_corpus"]["doc08_overlay_path"],
            doc12_overlay_path=project_root
            / evaluator.config["candidate_overlay"]["overlay_path"],
            permitted_root=project_root,
        )
        try:
            reloaded_baseline = load_exact_frozen_arm(
                arm="baseline",
                chunks=overlay.baseline_chunks,
                npz_path=baseline_copy_npz,
                manifest_path=baseline_copy_manifest,
                embedding_model=evaluator.embedding_model,
                project_root=project_root,
            )
            reloaded_candidate = load_exact_frozen_arm(
                arm="candidate",
                chunks=overlay.candidate_chunks,
                npz_path=candidate_copy_npz,
                manifest_path=candidate_copy_manifest,
                embedding_model=evaluator.embedding_model,
                project_root=project_root,
            )
        finally:
            cleanup_overlay_temp_dir(overlay.temp_input_dir)

        assert reloaded_baseline.chunk_ids == exact_baseline_arm.chunk_ids
        assert reloaded_candidate.chunk_ids == exact_candidate_arm.chunk_ids
        assert reloaded_baseline.embedding_digest == exact_baseline_arm.embedding_digest
        assert reloaded_candidate.embedding_digest == exact_candidate_arm.embedding_digest
        loader_digests.append(
            hashlib.sha256(
                json.dumps(
                    {
                        "baseline": reloaded_baseline.embedding_digest,
                        "candidate": reloaded_candidate.embedding_digest,
                        "baseline_ids": list(reloaded_baseline.chunk_ids[:3]),
                        "candidate_ids": list(reloaded_candidate.chunk_ids[:3]),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )

    reconstruction_root = parent_dir / "SOURCE_COMMIT"
    copy_snapshot_to_temp_root(
        npz_path=candidate_npz,
        manifest_path=candidate_manifest,
        temp_root=reconstruction_root,
    )
    reconstruction_identical = (
        hashlib.sha256(candidate_npz.read_bytes()).hexdigest()
        == hashlib.sha256((reconstruction_root / candidate_npz.name).read_bytes()).hexdigest()
    )

    return ExactReplayResult(
        authoritative=True,
        repeated_full_runs=runs,
        repeated_full_identical=len(set(full_digests)) == 1,
        repeated_full_digest=full_digests[0] if full_digests else None,
        independent_loader_roots=3,
        independent_loader_identical=len(set(loader_digests)) == 1,
        loader_digests=loader_digests,
        source_commit_reconstruction_identical=reconstruction_identical,
        baseline_snapshot_digest=exact_baseline_arm.snapshot_digest,
        candidate_snapshot_digest=exact_candidate_arm.snapshot_digest,
    )


def build_replay_integrity_result(
    *,
    index_dir: Path,
    project_root: Path,
    config: dict,
    canonical_dir: Path,
    evaluator,
    ann_evaluator,
    exact_baseline_arm: ExactFrozenArm,
    exact_candidate_arm: ExactFrozenArm,
    rebuild_parent: Path,
    embedding_snapshot: Path,
    embedding_snapshot_manifest: Path,
    baseline_snapshot: Path,
    baseline_snapshot_manifest: Path,
    run_live_probe: bool = True,
    exact_run=None,
) -> ReplayIntegrityResult:
    resolved_snapshot = (
        embedding_snapshot if embedding_snapshot.is_absolute() else project_root / embedding_snapshot
    )
    resolved_manifest = (
        embedding_snapshot_manifest
        if embedding_snapshot_manifest.is_absolute()
        else project_root / embedding_snapshot_manifest
    )
    resolved_baseline_snapshot = (
        baseline_snapshot
        if baseline_snapshot.is_absolute()
        else project_root / baseline_snapshot
    )
    resolved_baseline_manifest = (
        baseline_snapshot_manifest
        if baseline_snapshot_manifest.is_absolute()
        else project_root / baseline_snapshot_manifest
    )
    manifest = load_snapshot_manifest(resolved_manifest)
    original_candidate_index = evaluator.candidate_index_dir
    environment_payload = capture_environment_provenance(project_root)
    environment = EnvironmentProvenance(
        kind="project_venv",
        python_version=str(environment_payload["python_version"]),
        pip_version=str(environment_payload["pip_version"]),
        pip_check_exit_code=int(environment_payload["pip_check_exit_code"]),
        dependency_check_summary=str(environment_payload["dependency_check_summary"]),
    )

    frozen_parent = rebuild_parent / "frozen_ann"
    live_parent = rebuild_parent / "live"
    exact_parent = rebuild_parent / "exact"
    ann_parent = rebuild_parent / "ann_robustness"
    try:
        frozen = run_frozen_snapshot_replay(
            config=config,
            canonical_dir=canonical_dir,
            parent_dir=frozen_parent,
            project_root=project_root,
            embedding_snapshot=resolved_snapshot,
            embedding_snapshot_manifest=resolved_manifest,
            evaluator=ann_evaluator,
            builds=3,
        )
        exact_replay = run_exact_replay_matrix(
            evaluator=evaluator,
            exact_baseline_arm=exact_baseline_arm,
            exact_candidate_arm=exact_candidate_arm,
            baseline_npz=resolved_baseline_snapshot,
            baseline_manifest=resolved_baseline_manifest,
            candidate_npz=resolved_snapshot,
            candidate_manifest=resolved_manifest,
            project_root=project_root,
            canonical_dir=canonical_dir,
            parent_dir=exact_parent,
        )
        authoritative_run = exact_run or evaluator.evaluate()
        ann_robustness = run_ann_robustness_matrix(
            config=config,
            canonical_dir=canonical_dir,
            parent_dir=ann_parent,
            project_root=project_root,
            ann_evaluator=ann_evaluator,
            exact_run=authoritative_run,
            embedding_snapshot=resolved_snapshot,
            embedding_snapshot_manifest=resolved_manifest,
            builds=3,
        )
        live = None
        if run_live_probe:
            live = run_live_provider_robustness(
                config=config,
                canonical_dir=canonical_dir,
                parent_dir=live_parent,
                project_root=project_root,
                evaluator=ann_evaluator,
                runs=9,
                original_candidate_index=original_candidate_index,
            )
        else:
            ann_evaluator.replace_candidate_index(original_candidate_index)

        integrity_verdict: ReplayIntegrityVerdict = (
            "PASS — EXACT FIXED-SNAPSHOT EVALUATION REPRODUCIBLE"
            if exact_replay.repeated_full_identical
            and exact_replay.independent_loader_identical
            and exact_replay.source_commit_reconstruction_identical
            else "FAIL — EXACT FIXED-SNAPSHOT EVALUATION NOT REPRODUCIBLE"
        )

        candidate_manifest = load_manifest(index_dir)
        return ReplayIntegrityResult(
            frozen_snapshot_replay=frozen,
            exact_replay=exact_replay,
            ann_robustness=ann_robustness,
            ann_rebuild_stability=AnnRebuildStability(status="VARIABLE", authoritative=False),
            live_provider_robustness=live,
            integrity_verdict=integrity_verdict,
            corpus_fingerprint=manifest.get("corpus_fingerprint"),
            chunk_payload_digest=manifest.get("chunk_payload_digest"),
            embedding_snapshot_path=repo_relative_path(resolved_snapshot, project_root),
            embedding_snapshot_manifest_path=repo_relative_path(resolved_manifest, project_root),
            baseline_snapshot_path=repo_relative_path(resolved_baseline_snapshot, project_root),
            baseline_snapshot_manifest_path=repo_relative_path(
                resolved_baseline_manifest,
                project_root,
            ),
            embedding_snapshot_digest=manifest.get("snapshot_digest"),
            embedding_digest=manifest.get("embedding_digest"),
            collection_content_digest=candidate_manifest.collection_content_digest,
            index_fingerprint=candidate_manifest.corpus_fingerprint,
            lineage=dict(manifest.get("lineage") or {}),
            environment=environment,
            pip_environment=environment_payload,
        )
    finally:
        shutil.rmtree(rebuild_parent, ignore_errors=True)


def build_replay_stability_result(
    *,
    index_dir: Path,
    project_root: Path,
    config: dict,
    canonical_dir: Path,
    evaluator,
    ann_evaluator,
    exact_baseline_arm: ExactFrozenArm,
    exact_candidate_arm: ExactFrozenArm,
    rebuild_parent: Path,
    embedding_snapshot: Path | None = None,
    embedding_snapshot_manifest: Path | None = None,
    baseline_snapshot: Path | None = None,
    baseline_snapshot_manifest: Path | None = None,
    exact_run=None,
) -> ReplayIntegrityResult:
    """Backward-compatible entry point used by evaluation CLI."""
    snapshot_cfg = config.get("embedding_snapshot") or {}
    snapshot_path = embedding_snapshot or project_root / snapshot_cfg.get(
        "candidate_snapshot_path",
        "data/05_evaluation/embedding_snapshots/doc12_threat_atomic_units_v1.npz",
    )
    manifest_path = embedding_snapshot_manifest or project_root / snapshot_cfg.get(
        "candidate_manifest_path",
        "data/05_evaluation/embedding_snapshots/doc12_threat_atomic_units_v1.manifest.json",
    )
    baseline_path = baseline_snapshot or project_root / snapshot_cfg.get(
        "baseline_snapshot_path",
        "data/05_evaluation/embedding_snapshots/doc08_atomic_risk_units_v1.npz",
    )
    baseline_manifest_path = baseline_snapshot_manifest or project_root / snapshot_cfg.get(
        "baseline_manifest_path",
        "data/05_evaluation/embedding_snapshots/doc08_atomic_risk_units_v1.manifest.json",
    )
    return build_replay_integrity_result(
        index_dir=index_dir,
        project_root=project_root,
        config=config,
        canonical_dir=canonical_dir,
        evaluator=evaluator,
        ann_evaluator=ann_evaluator,
        exact_baseline_arm=exact_baseline_arm,
        exact_candidate_arm=exact_candidate_arm,
        rebuild_parent=rebuild_parent,
        embedding_snapshot=snapshot_path,
        embedding_snapshot_manifest=manifest_path,
        baseline_snapshot=baseline_path,
        baseline_snapshot_manifest=baseline_manifest_path,
        exact_run=exact_run,
    )
