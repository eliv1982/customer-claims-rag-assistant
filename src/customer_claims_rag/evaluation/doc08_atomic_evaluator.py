"""Evaluator for doc08 atomic corpus A/B experiment."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.doc08_atomic_metrics import (
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    REQUIRED_CASE_IDS,
    THREAT_EXTENSION_IDS,
    build_case_diagnostic,
    build_doc08_footprint,
    compute_extension_metrics,
    evaluate_extension_acceptance,
    evaluate_frozen_acceptance,
    list_unreachable,
)
from customer_claims_rag.evaluation.doc08_atomic_models import (
    Doc08AtomicCaseResult,
    Doc08AtomicEvaluationRun,
    Doc08CaseDiagnostic,
    Doc08ChunkDiffModel,
    Doc08Verdict,
    ExtensionEvaluationResult,
    RetrievalArmMetadata,
)
from customer_claims_rag.evaluation.extension_parser import (
    compute_extension_dataset_fingerprint,
    load_extension_corpus,
)
from customer_claims_rag.evaluation.git_state import read_git_state
from customer_claims_rag.evaluation.metrics import aggregate_case_metrics, compute_case_metrics
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    build_ranking_arm_result,
    chunks_from_search_results,
    compute_pool_reachability,
    compute_evaluation_dataset_fingerprint,
    rerank_to_final_top_k,
)
from customer_claims_rag.evaluation.diversity_metrics import load_vector_pool_cap_config, compute_vector_pool_cap_config_hash
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.ingestion.corpus_overlay import (
    DOC08_DOCUMENT_ID,
    Doc08ChunkDiff,
    build_baseline_and_overlay_chunks,
    cleanup_overlay_temp_dir,
    compute_doc08_chunk_diff,
    compute_doc08_fingerprint,
)
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, compute_config_hash, load_reranker_config
from customer_claims_rag.retrieval.retriever import BaselineRetriever


def _vector_ranks_by_document(results) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for index, result in enumerate(results, start=1):
        ranks.setdefault(result.document_id, []).append(index)
    return ranks


class Doc08AtomicAbEvaluator:
    """Compare canonical vs overlay index with identical retrieval chain."""

    def __init__(
        self,
        *,
        baseline_retriever: BaselineRetriever,
        candidate_retriever: BaselineRetriever,
        reranker: SourceAuthorityV1Reranker,
        fetch_k: int,
        pool_k: int,
        per_document_cap: int,
        final_top_k: int,
        threshold: float,
        frozen_cases: list,
        extension_cases: list,
        baseline_index_dir: Path,
        candidate_index_dir: Path,
        collection_name: str,
        embedding_model: str,
        experiment_id: str,
        config: dict,
        chunk_diff: Doc08ChunkDiff,
        baseline_doc08_fingerprint: str,
        candidate_doc08_fingerprint: str,
        frozen_benchmark_fingerprint: str,
        extension_benchmark_fingerprint: str,
        production_retrieval_config_hash: str,
        project_root: Path | None = None,
    ) -> None:
        self.baseline_retriever = baseline_retriever
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.pool_k = pool_k
        self.per_document_cap = per_document_cap
        self.final_top_k = final_top_k
        self.threshold = threshold
        self.frozen_cases = frozen_cases
        self.extension_cases = extension_cases
        self.baseline_index_dir = baseline_index_dir
        self.candidate_index_dir = candidate_index_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.experiment_id = experiment_id
        self.config = config
        self.chunk_diff = chunk_diff
        self.baseline_doc08_fingerprint = baseline_doc08_fingerprint
        self.candidate_doc08_fingerprint = candidate_doc08_fingerprint
        self.frozen_benchmark_fingerprint = frozen_benchmark_fingerprint
        self.extension_benchmark_fingerprint = extension_benchmark_fingerprint
        self.production_retrieval_config_hash = production_retrieval_config_hash
        self.project_root = project_root

    def evaluate(self) -> Doc08AtomicEvaluationRun:
        started = time.perf_counter()
        self.baseline_retriever.validate_index()
        self.candidate_retriever.validate_index()

        baseline_manifest = load_manifest(self.baseline_index_dir)
        candidate_manifest = load_manifest(self.candidate_index_dir)
        git_state = read_git_state(self.project_root)

        frozen_results, frozen_diagnostics = self._evaluate_cases(self.frozen_cases)
        ext_baseline_cases, ext_candidate_cases = self._evaluate_extension_cases(self.extension_cases)

        baseline_case_results = [r for r in frozen_results]
        baseline_metrics_list = [item.baseline_case for item in frozen_results]
        candidate_metrics_list = [item.candidate_case for item in frozen_results]

        frozen_baseline_metrics = aggregate_case_metrics(baseline_metrics_list)
        frozen_candidate_metrics = aggregate_case_metrics(candidate_metrics_list)

        faq_top4 = sum(
            1
            for item in frozen_results
            if any(
                ch.document_id == "10_customer_faq"
                for ch in item.candidate_case.retrieved_chunks[:4]
            )
        )

        primary_unreach_b, fully_unreach_b = list_unreachable(frozen_results, arm="baseline")
        primary_unreach_c, fully_unreach_c = list_unreachable(frozen_results, arm="candidate")

        t044_diag = next((d for d in frozen_diagnostics if d.case_id == "T044"), None)
        t047_diag = next((d for d in frozen_diagnostics if d.case_id == "T047"), None)

        frozen_checks, frozen_verdict = evaluate_frozen_acceptance(
            baseline_metrics=frozen_baseline_metrics,
            candidate_metrics=frozen_candidate_metrics,
            faq_top4_candidate=faq_top4,
            primary_unreachable_baseline=primary_unreach_b,
            primary_unreachable_candidate=primary_unreach_c,
            t044_diag=t044_diag,
            t047_diag=t047_diag,
        )

        ext_b_metrics = compute_extension_metrics(
            ext_baseline_cases,
            privacy_ids=PRIVACY_EXTENSION_IDS,
            threat_ids=THREAT_EXTENSION_IDS,
            negative_ids=NEGATIVE_EXTENSION_IDS,
        )
        ext_c_metrics = compute_extension_metrics(
            ext_candidate_cases,
            privacy_ids=PRIVACY_EXTENSION_IDS,
            threat_ids=THREAT_EXTENSION_IDS,
            negative_ids=NEGATIVE_EXTENSION_IDS,
        )
        ext_checks, ext_verdict = evaluate_extension_acceptance(ext_c_metrics)

        all_checks = frozen_checks + ext_checks
        verdict: Doc08Verdict = (
            "ACCEPTED AS TARGETED CORPUS REPAIR"
            if frozen_verdict == "ACCEPTED AS TARGETED CORPUS REPAIR"
            and ext_verdict == "ACCEPTED AS TARGETED CORPUS REPAIR"
            and all(c.passed for c in all_checks)
            else "REJECTED"
        )

        promoted = [
            item.case_id
            for item in frozen_results
            if not item.baseline_case.primary_hit_at_4 and item.candidate_case.primary_hit_at_4
        ]
        regressed = [
            item.case_id
            for item in frozen_results
            if item.baseline_case.primary_hit_at_4 and not item.candidate_case.primary_hit_at_4
        ]

        reranker_hash = compute_config_hash(self.reranker.config)
        baseline_arm = RetrievalArmMetadata(
            arm="baseline",
            index_dir=str(self.baseline_index_dir),
            index_fingerprint=baseline_manifest.corpus_fingerprint,
            corpus_fingerprint=baseline_manifest.corpus_fingerprint,
            doc08_fingerprint=self.baseline_doc08_fingerprint,
            chunk_count=baseline_manifest.chunk_count,
            document_count=baseline_manifest.document_count,
            fetch_k=self.fetch_k,
            candidate_pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
            final_top_k=self.final_top_k,
            threshold=self.threshold,
            reranker_id=self.reranker.config.reranker_id,
            reranker_config_hash=reranker_hash,
            collection=self.collection_name,
            embedding_model=self.embedding_model,
        )
        candidate_arm = RetrievalArmMetadata(
            arm="candidate",
            index_dir=str(self.candidate_index_dir),
            index_fingerprint=candidate_manifest.corpus_fingerprint,
            corpus_fingerprint=candidate_manifest.corpus_fingerprint,
            doc08_fingerprint=self.candidate_doc08_fingerprint,
            chunk_count=candidate_manifest.chunk_count,
            document_count=candidate_manifest.document_count,
            fetch_k=self.fetch_k,
            candidate_pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
            final_top_k=self.final_top_k,
            threshold=self.threshold,
            reranker_id=self.reranker.config.reranker_id,
            reranker_config_hash=reranker_hash,
            collection=self.collection_name,
            embedding_model=self.embedding_model,
        )

        return Doc08AtomicEvaluationRun(
            timestamp=datetime.now(timezone.utc),
            experiment_id=self.experiment_id,
            source_commit=git_state.commit,
            artifact_commit=None,
            baseline_arm=baseline_arm,
            candidate_arm=candidate_arm,
            frozen_benchmark_fingerprint=self.frozen_benchmark_fingerprint,
            extension_benchmark_fingerprint=self.extension_benchmark_fingerprint,
            production_retrieval_config_hash=self.production_retrieval_config_hash,
            chunk_diff=Doc08ChunkDiffModel.from_dataclass(self.chunk_diff),
            frozen_baseline_metrics=frozen_baseline_metrics,
            frozen_candidate_metrics=frozen_candidate_metrics,
            case_results=frozen_results,
            required_diagnostics=[d for d in frozen_diagnostics if d.case_id in REQUIRED_CASE_IDS],
            doc08_footprint_baseline=build_doc08_footprint(frozen_results, arm="baseline"),
            doc08_footprint_candidate=build_doc08_footprint(frozen_results, arm="candidate"),
            promoted_cases=sorted(promoted),
            regressed_cases=sorted(regressed),
            primary_unreachable_baseline=primary_unreach_b,
            primary_unreachable_candidate=primary_unreach_c,
            fully_unreachable_baseline=fully_unreach_b,
            fully_unreachable_candidate=fully_unreach_c,
            extension=ExtensionEvaluationResult(
                benchmark_id=self.config["extension_benchmark"]["benchmark_id"],
                fingerprint=self.extension_benchmark_fingerprint,
                baseline=ext_b_metrics,
                candidate=ext_c_metrics,
                verdict=ext_verdict,
                acceptance_checks=ext_checks,
            ),
            acceptance_checks=all_checks,
            interpretation_boundary=[
                "T047: doc12 remains product-correct primary for direct courier threats; "
                "doc08 acceptance as parent escalation source does not require outranking doc12.",
                "Frozen expected primary for T047 remains 08 per frozen benchmark; "
                "extension set validates doc12 specificity separately.",
            ],
            verdict=verdict,
            latency_seconds_total=time.perf_counter() - started,
            config=self.config,
        )

    def _evaluate_cases(
        self,
        cases: list,
    ) -> tuple[list[Doc08AtomicCaseResult], list[Doc08CaseDiagnostic]]:
        results: list[Doc08AtomicCaseResult] = []
        diagnostics: list[Doc08CaseDiagnostic] = []
        for case in cases:
            outcome = self._evaluate_single_case(case)
            results.append(outcome["result"])
            diagnostics.append(outcome["diagnostic"])
        return results, diagnostics

    def _evaluate_extension_cases(self, cases: list) -> tuple[list, list]:
        baseline_cases: list = []
        candidate_cases: list = []
        for case in cases:
            b_resp = self.baseline_retriever.search(case.query)
            c_resp = self.candidate_retriever.search(case.query)
            b_pool, _ = apply_per_document_cap(
                list(b_resp.results),
                fetch_k=self.fetch_k,
                pool_k=self.pool_k,
                per_document_cap=self.per_document_cap,
            )
            c_pool, _ = apply_per_document_cap(
                list(c_resp.results),
                fetch_k=self.fetch_k,
                pool_k=self.pool_k,
                per_document_cap=self.per_document_cap,
            )
            b_final, _ = rerank_to_final_top_k(
                self.reranker, case.query, b_pool, final_top_k=self.final_top_k
            )
            c_final, _ = rerank_to_final_top_k(
                self.reranker, case.query, c_pool, final_top_k=self.final_top_k
            )
            baseline_cases.append(
                compute_case_metrics(
                    test_id=case.test_id,
                    query=case.query,
                    expected_risk=case.expected_risk,
                    expected_primary_documents=case.expected_primary_documents,
                    expected_supporting_documents=case.expected_supporting_documents,
                    fallback_expected=case.fallback_expected,
                    category=case.category,
                    retrieved_chunks=chunks_from_search_results(b_final),
                )
            )
            candidate_cases.append(
                compute_case_metrics(
                    test_id=case.test_id,
                    query=case.query,
                    expected_risk=case.expected_risk,
                    expected_primary_documents=case.expected_primary_documents,
                    expected_supporting_documents=case.expected_supporting_documents,
                    fallback_expected=case.fallback_expected,
                    category=case.category,
                    retrieved_chunks=chunks_from_search_results(c_final),
                )
            )
        return baseline_cases, candidate_cases

    def _evaluate_single_case(self, case) -> dict:
        b_resp = self.baseline_retriever.search(case.query)
        c_resp = self.candidate_retriever.search(case.query)
        b_vector = list(b_resp.results)
        c_vector = list(c_resp.results)

        b_pool, _ = apply_per_document_cap(
            b_vector, fetch_k=self.fetch_k, pool_k=self.pool_k, per_document_cap=self.per_document_cap
        )
        c_pool, _ = apply_per_document_cap(
            c_vector, fetch_k=self.fetch_k, pool_k=self.pool_k, per_document_cap=self.per_document_cap
        )

        b_reach = compute_pool_reachability(
            b_pool,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )
        c_reach = compute_pool_reachability(
            c_pool,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )

        b_final, b_reranked = rerank_to_final_top_k(
            self.reranker, case.query, b_pool, final_top_k=self.final_top_k
        )
        c_final, c_reranked = rerank_to_final_top_k(
            self.reranker, case.query, c_pool, final_top_k=self.final_top_k
        )

        b_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(b_final),
        )
        c_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(c_final),
        )

        b_ranking = build_ranking_arm_result(
            pool=b_pool,
            final_results=b_final,
            reranked=b_reranked,
            case=b_case,
            pool_reachability=b_reach,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )
        c_ranking = build_ranking_arm_result(
            pool=c_pool,
            final_results=c_final,
            reranked=c_reranked,
            case=c_case,
            pool_reachability=c_reach,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )

        result = Doc08AtomicCaseResult(
            case_id=case.test_id,
            baseline_case=b_case,
            candidate_case=c_case,
            baseline_ranking=b_ranking,
            candidate_ranking=c_ranking,
            baseline_pool=b_reach,
            candidate_pool=c_reach,
            baseline_pool_doc_ids=[r.document_id for r in b_pool],
            candidate_pool_doc_ids=[r.document_id for r in c_pool],
            baseline_final_doc_ids=[r.document_id for r in b_final],
            candidate_final_doc_ids=[r.document_id for r in c_final],
        )
        diagnostic = build_case_diagnostic(
            result,
            baseline_vector=b_vector,
            candidate_vector=c_vector,
            query=case.query,
            expected_primary=case.expected_primary_documents,
        )
        return {"result": result, "diagnostic": diagnostic}


def prepare_chunk_diff_and_fingerprints(
    *,
    canonical_dir: Path,
    overlay_path: Path,
    permitted_root: Path,
    embedding_model: str,
) -> tuple[Doc08ChunkDiff, str, str]:
    overlay_result = build_baseline_and_overlay_chunks(
        canonical_dir=canonical_dir,
        overlay_document_path=overlay_path,
        permitted_root=permitted_root,
    )
    try:
        diff = compute_doc08_chunk_diff(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
        )
        if not diff.non_doc08_byte_identical:
            raise ValueError("non-doc08 chunks are not byte-identical")
        baseline_fp = compute_doc08_fingerprint(
            overlay_result.baseline_chunks, embedding_model=embedding_model
        )
        candidate_fp = compute_doc08_fingerprint(
            overlay_result.candidate_chunks, embedding_model=embedding_model
        )
        return diff, baseline_fp, candidate_fp
    finally:
        cleanup_overlay_temp_dir(overlay_result.temp_input_dir)
