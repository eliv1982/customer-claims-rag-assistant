"""Evaluator for doc08 atomic corpus A/B experiment."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.doc08_atomic_contract import (
    DirtySourceTreeError,
    _has_disallowed_worktree_changes,
    ensure_clean_source_tree,
    load_baseline_reference_oracle,
)
from customer_claims_rag.evaluation.doc08_atomic_metrics import (
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    REQUIRED_CASE_IDS,
    THREAT_EXTENSION_IDS,
    assert_baseline_reproduction,
    build_case_diagnostic,
    build_doc08_footprint,
    build_doc08_reachability_comparison,
    build_extension_case_diagnostic,
    build_frozen_reachability_snapshot,
    compute_extension_metrics,
    compute_primary_hit_at_12,
    count_faq_in_top4,
    evaluate_extension_acceptance,
    evaluate_frozen_acceptance,
    fully_unreachable_case_ids,
    primary_unreachable_case_ids,
    validate_baseline_reproduction,
)
from customer_claims_rag.evaluation.doc08_atomic_models import (
    BaselineReproductionResult,
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
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    build_ranking_arm_result,
    chunks_from_search_results,
    compute_pool_reachability,
    rerank_to_final_top_k,
)
from customer_claims_rag.evaluation.diversity_metrics import compute_vector_pool_cap_config_hash
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.ingestion.corpus_overlay import (
    Doc08ChunkDiff,
    build_baseline_and_overlay_chunks,
    cleanup_overlay_temp_dir,
    compute_doc08_chunk_diff,
    compute_doc08_fingerprint,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, compute_config_hash
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
        reference_artifact_path: Path,
        project_root: Path | None = None,
        allow_dirty_source: bool = False,
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
        self.reference_artifact_path = reference_artifact_path
        self.project_root = project_root
        self.allow_dirty_source = allow_dirty_source
        self.reference_oracle = load_baseline_reference_oracle(reference_artifact_path)

    def evaluate(self) -> Doc08AtomicEvaluationRun:
        started = time.perf_counter()
        execution_ts = datetime.now(timezone.utc)
        self.baseline_retriever.validate_index()
        self.candidate_retriever.validate_index()

        baseline_manifest = load_manifest(self.baseline_index_dir)
        candidate_manifest = load_manifest(self.candidate_index_dir)
        git_state = read_git_state(self.project_root)
        ensure_clean_source_tree(
            git_state,
            allow_dirty=self.allow_dirty_source,
            project_root=self.project_root,
        )
        source_dirty = (
            _has_disallowed_worktree_changes(self.project_root)
            if self.project_root is not None
            else bool(git_state.dirty)
        )

        frozen_results, frozen_diagnostics = self._evaluate_cases(self.frozen_cases)
        extension_outcome = self._evaluate_extension_cases(self.extension_cases)

        baseline_metrics_list = [item.baseline_case for item in frozen_results]
        candidate_metrics_list = [item.candidate_case for item in frozen_results]
        frozen_baseline_metrics = aggregate_case_metrics(baseline_metrics_list)
        frozen_candidate_metrics = aggregate_case_metrics(candidate_metrics_list)

        baseline_primary_hit_at_12 = compute_primary_hit_at_12(frozen_results, arm="baseline")
        candidate_primary_hit_at_12 = compute_primary_hit_at_12(frozen_results, arm="candidate")
        reachability = build_doc08_reachability_comparison(frozen_results)
        faq_top4_baseline = count_faq_in_top4(frozen_results, arm="baseline")
        faq_top4_candidate = count_faq_in_top4(frozen_results, arm="candidate")
        primary_unreach_b = primary_unreachable_case_ids(frozen_results, arm="baseline")
        primary_unreach_c = primary_unreachable_case_ids(frozen_results, arm="candidate")
        fully_unreach_b = fully_unreachable_case_ids(frozen_results, arm="baseline")
        fully_unreach_c = fully_unreachable_case_ids(frozen_results, arm="candidate")

        reproduction_checks = validate_baseline_reproduction(
            oracle=self.reference_oracle,
            baseline_metrics=frozen_baseline_metrics,
            reachability=reachability,
            primary_unreachable=primary_unreach_b,
            fully_unreachable=fully_unreach_b,
            faq_top4=faq_top4_baseline,
            primary_hit_at_12=baseline_primary_hit_at_12,
        )
        baseline_reproduction = BaselineReproductionResult(
            reference_experiment_id=self.reference_oracle.experiment_id,
            reference_arm=self.reference_oracle.reference_arm,
            reference_artifact_path=self.reference_oracle.artifact_path,
            passed=all(check.passed for check in reproduction_checks),
            checks=reproduction_checks,
        )
        assert_baseline_reproduction(reproduction_checks)

        t044_diag = next((diag for diag in frozen_diagnostics if diag.case_id == "T044"), None)
        t047_diag = next((diag for diag in frozen_diagnostics if diag.case_id == "T047"), None)

        frozen_checks, frozen_verdict = evaluate_frozen_acceptance(
            baseline_metrics=frozen_baseline_metrics,
            candidate_metrics=frozen_candidate_metrics,
            baseline_primary_hit_at_12=baseline_primary_hit_at_12,
            candidate_primary_hit_at_12=candidate_primary_hit_at_12,
            reachability=reachability,
            faq_top4_baseline=faq_top4_baseline,
            faq_top4_candidate=faq_top4_candidate,
            primary_unreachable_baseline=primary_unreach_b,
            primary_unreachable_candidate=primary_unreach_c,
            t044_reachable=t044_diag.candidate_primary_reachable if t044_diag else False,
            t044_hit12=t044_diag.candidate_primary_hit_at_12 if t044_diag else False,
            t047_doc12_final_rank=t047_diag.candidate_final_rank_doc12 if t047_diag else None,
        )

        ext_checks, ext_verdict = evaluate_extension_acceptance(extension_outcome["candidate_metrics"])
        all_checks = reproduction_checks + frozen_checks + ext_checks
        verdict: Doc08Verdict = (
            "ACCEPTED AS TARGETED CORPUS REPAIR"
            if frozen_verdict == "ACCEPTED AS TARGETED CORPUS REPAIR"
            and ext_verdict == "ACCEPTED AS TARGETED CORPUS REPAIR"
            and all(check.passed for check in frozen_checks + ext_checks)
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
        baseline_arm = self._build_arm_metadata(
            arm="baseline",
            manifest=baseline_manifest,
            doc08_fingerprint=self.baseline_doc08_fingerprint,
            reranker_hash=reranker_hash,
            execution_timestamp=execution_ts,
        )
        candidate_arm = self._build_arm_metadata(
            arm="candidate",
            manifest=candidate_manifest,
            doc08_fingerprint=self.candidate_doc08_fingerprint,
            reranker_hash=reranker_hash,
            execution_timestamp=execution_ts,
        )

        return Doc08AtomicEvaluationRun(
            timestamp=execution_ts,
            experiment_id=self.experiment_id,
            source_commit=git_state.commit,
            source_dirty=source_dirty,
            artifact_commit=None,
            reference_experiment_id=self.reference_oracle.experiment_id,
            reference_arm=self.reference_oracle.reference_arm,
            reference_artifact_path=self.reference_oracle.artifact_path,
            baseline_reproduction=baseline_reproduction,
            baseline_arm=baseline_arm,
            candidate_arm=candidate_arm,
            frozen_benchmark_fingerprint=self.frozen_benchmark_fingerprint,
            extension_benchmark_fingerprint=self.extension_benchmark_fingerprint,
            production_retrieval_config_hash=self.production_retrieval_config_hash,
            chunk_diff=Doc08ChunkDiffModel.from_dataclass(self.chunk_diff),
            frozen_baseline_metrics=frozen_baseline_metrics,
            frozen_candidate_metrics=frozen_candidate_metrics,
            frozen_baseline_primary_hit_at_12=baseline_primary_hit_at_12,
            frozen_candidate_primary_hit_at_12=candidate_primary_hit_at_12,
            frozen_reachability=reachability,
            frozen_reachability_baseline=build_frozen_reachability_snapshot(
                frozen_results,
                arm="baseline",
            ),
            frozen_reachability_candidate=build_frozen_reachability_snapshot(
                frozen_results,
                arm="candidate",
            ),
            faq_top4_baseline=faq_top4_baseline,
            faq_top4_candidate=faq_top4_candidate,
            case_results=frozen_results,
            required_diagnostics=[diag for diag in frozen_diagnostics if diag.case_id in REQUIRED_CASE_IDS],
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
                baseline=extension_outcome["baseline_metrics"],
                candidate=extension_outcome["candidate_metrics"],
                case_diagnostics=extension_outcome["diagnostics"],
                verdict=ext_verdict,
                acceptance_checks=ext_checks,
            ),
            acceptance_checks=all_checks,
            interpretation_boundary=[
                "T047: doc12 remains product-correct primary for direct courier threats; "
                "doc08 acceptance as parent escalation source does not require outranking doc12.",
                "Frozen expected primary for T047 remains 08 per frozen benchmark; "
                "extension set validates doc12 specificity separately.",
                "Extension threat failures where baseline already fails are not classified as doc08 regression.",
            ],
            verdict=verdict,
            latency_seconds_total=time.perf_counter() - started,
            config=self.config,
        )

    def _build_arm_metadata(
        self,
        *,
        arm: str,
        manifest,
        doc08_fingerprint: str,
        reranker_hash: str,
        execution_timestamp: datetime,
    ) -> RetrievalArmMetadata:
        return RetrievalArmMetadata(
            arm=arm,
            index_dir=str(self.baseline_index_dir if arm == "baseline" else self.candidate_index_dir),
            index_fingerprint=manifest.corpus_fingerprint,
            corpus_fingerprint=manifest.corpus_fingerprint,
            doc08_fingerprint=doc08_fingerprint,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
            fetch_k=self.fetch_k,
            candidate_pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
            final_top_k=self.final_top_k,
            threshold=self.threshold,
            reranker_id=self.reranker.config.reranker_id,
            reranker_config_hash=reranker_hash,
            collection=self.collection_name,
            embedding_model=self.embedding_model,
            benchmark_fingerprint=self.frozen_benchmark_fingerprint,
            execution_timestamp=execution_timestamp,
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

    def _evaluate_extension_cases(self, cases: list) -> dict:
        baseline_cases = []
        candidate_cases = []
        diagnostics = []
        for case in cases:
            outcome = self._evaluate_single_case(case)
            baseline_cases.append(outcome["result"].baseline_case)
            candidate_cases.append(outcome["result"].candidate_case)
            diagnostics.append(
                build_extension_case_diagnostic(
                    case_id=case.test_id,
                    query=case.query,
                    expected_primary=case.expected_primary_documents,
                    expected_supporting=case.expected_supporting_documents,
                    baseline_vector_ranks=_vector_ranks_by_document(outcome["baseline_vector"]),
                    candidate_vector_ranks=_vector_ranks_by_document(outcome["candidate_vector"]),
                    baseline_pool_doc_ids=outcome["result"].baseline_pool_doc_ids,
                    candidate_pool_doc_ids=outcome["result"].candidate_pool_doc_ids,
                    baseline_final_doc_ids=outcome["result"].baseline_final_doc_ids,
                    candidate_final_doc_ids=outcome["result"].candidate_final_doc_ids,
                    baseline_case=outcome["result"].baseline_case,
                    candidate_case=outcome["result"].candidate_case,
                )
            )
        return {
            "baseline_metrics": compute_extension_metrics(
                baseline_cases,
                privacy_ids=PRIVACY_EXTENSION_IDS,
                threat_ids=THREAT_EXTENSION_IDS,
                negative_ids=NEGATIVE_EXTENSION_IDS,
            ),
            "candidate_metrics": compute_extension_metrics(
                candidate_cases,
                privacy_ids=PRIVACY_EXTENSION_IDS,
                threat_ids=THREAT_EXTENSION_IDS,
                negative_ids=NEGATIVE_EXTENSION_IDS,
            ),
            "diagnostics": diagnostics,
        }

    def _evaluate_single_case(self, case) -> dict:
        b_resp = self.baseline_retriever.search(case.query)
        c_resp = self.candidate_retriever.search(case.query)
        b_vector = list(b_resp.results)
        c_vector = list(c_resp.results)

        b_pool, _ = apply_per_document_cap(
            b_vector,
            fetch_k=self.fetch_k,
            pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
        )
        c_pool, _ = apply_per_document_cap(
            c_vector,
            fetch_k=self.fetch_k,
            pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
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
            baseline_pool_doc_ids=[item.document_id for item in b_pool],
            candidate_pool_doc_ids=[item.document_id for item in c_pool],
            baseline_final_doc_ids=[item.document_id for item in b_final],
            candidate_final_doc_ids=[item.document_id for item in c_final],
        )
        diagnostic = build_case_diagnostic(
            result,
            baseline_vector=b_vector,
            candidate_vector=c_vector,
            query=case.query,
            expected_primary=case.expected_primary_documents,
        )
        return {
            "result": result,
            "diagnostic": diagnostic,
            "baseline_vector": b_vector,
            "candidate_vector": c_vector,
        }


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
            overlay_result.baseline_chunks,
            embedding_model=embedding_model,
        )
        candidate_fp = compute_doc08_fingerprint(
            overlay_result.candidate_chunks,
            embedding_model=embedding_model,
        )
        return diff, baseline_fp, candidate_fp
    finally:
        cleanup_overlay_temp_dir(overlay_result.temp_input_dir)


__all__ = [
    "Doc08AtomicAbEvaluator",
    "prepare_chunk_diff_and_fingerprints",
]
