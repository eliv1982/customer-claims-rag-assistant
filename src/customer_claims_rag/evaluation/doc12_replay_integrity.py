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
from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    FrozenSnapshotReplayResult,
    LiveProviderRobustnessResult,
    ReplayIntegrityResult,
    ReplayIntegrityVerdict,
)
from customer_claims_rag.evaluation.extension_parser import load_extension_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import rerank_to_final_top_k
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import (
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
)
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
    embedding_snapshot: Path | None = None,
    embedding_snapshot_manifest: Path | None = None,
    live_provider: bool = False,
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
        trace = trace_e008_case(index_dir=index_dir, project_root=project_root, settings=resolved_settings)
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
        authoritative=True,
    )


def run_live_provider_robustness(
    *,
    config: dict,
    canonical_dir: Path,
    parent_dir: Path,
    project_root: Path,
    evaluator,
    runs: int = 9,
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
    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return {
        "sys_executable": sys.executable.replace("\\", "/"),
        "working_directory": str(project_root).replace("\\", "/"),
        "pip_version_stdout": pip_version.stdout.strip(),
        "pip_version_stderr": pip_version.stderr.strip(),
        "pip_check_stdout": pip_check.stdout.strip(),
        "pip_check_stderr": pip_check.stderr.strip(),
        "pip_check_exit_code": pip_check.returncode,
    }


def build_replay_integrity_result(
    *,
    index_dir: Path,
    project_root: Path,
    config: dict,
    canonical_dir: Path,
    evaluator,
    rebuild_parent: Path,
    embedding_snapshot: Path,
    embedding_snapshot_manifest: Path,
    run_live_probe: bool = True,
) -> ReplayIntegrityResult:
    resolved_snapshot = (
        embedding_snapshot if embedding_snapshot.is_absolute() else project_root / embedding_snapshot
    )
    resolved_manifest = (
        embedding_snapshot_manifest
        if embedding_snapshot_manifest.is_absolute()
        else project_root / embedding_snapshot_manifest
    )
    manifest = load_snapshot_manifest(resolved_manifest)

    frozen_parent = rebuild_parent / "frozen"
    live_parent = rebuild_parent / "live"
    try:
        frozen = run_frozen_snapshot_replay(
            config=config,
            canonical_dir=canonical_dir,
            parent_dir=frozen_parent,
            project_root=project_root,
            embedding_snapshot=resolved_snapshot,
            embedding_snapshot_manifest=resolved_manifest,
            evaluator=evaluator,
            builds=3,
        )
        live = None
        if run_live_probe:
            live = run_live_provider_robustness(
                config=config,
                canonical_dir=canonical_dir,
                parent_dir=live_parent,
                project_root=project_root,
                evaluator=evaluator,
                runs=9,
            )

        integrity_verdict: ReplayIntegrityVerdict = (
            "PASS — FIXED-SNAPSHOT REPLAY REPRODUCIBLE"
            if frozen.all_identical
            else "FAIL — FIXED-SNAPSHOT REPLAY NOT REPRODUCIBLE"
        )

        candidate_manifest = load_manifest(index_dir)
        return ReplayIntegrityResult(
            frozen_snapshot_replay=frozen,
            live_provider_robustness=live,
            integrity_verdict=integrity_verdict,
            corpus_fingerprint=manifest.get("corpus_fingerprint"),
            chunk_payload_digest=manifest.get("chunk_payload_digest"),
            embedding_snapshot_path=repo_relative_path(resolved_snapshot, project_root),
            embedding_snapshot_manifest_path=repo_relative_path(resolved_manifest, project_root),
            embedding_snapshot_digest=manifest.get("snapshot_digest"),
            embedding_digest=manifest.get("embedding_digest"),
            collection_content_digest=candidate_manifest.collection_content_digest,
            index_fingerprint=candidate_manifest.corpus_fingerprint,
            lineage=dict(manifest.get("lineage") or {}),
            pip_environment=capture_pip_environment(project_root=project_root),
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
    rebuild_parent: Path,
    embedding_snapshot: Path | None = None,
    embedding_snapshot_manifest: Path | None = None,
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
    return build_replay_integrity_result(
        index_dir=index_dir,
        project_root=project_root,
        config=config,
        canonical_dir=canonical_dir,
        evaluator=evaluator,
        rebuild_parent=rebuild_parent,
        embedding_snapshot=snapshot_path,
        embedding_snapshot_manifest=manifest_path,
        run_live_probe=True,
    )
