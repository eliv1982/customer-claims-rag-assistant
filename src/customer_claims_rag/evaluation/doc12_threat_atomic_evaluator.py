"""Evaluator for doc12 threat atomic corpus A/B experiment."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.doc08_atomic_contract import (
    _has_disallowed_worktree_changes,
    ensure_clean_source_tree,
)
from customer_claims_rag.evaluation.doc08_atomic_models import (
    AcceptanceCheck,
    BaselineReproductionResult,
    ExtensionEvaluationResult,
    RetrievalArmMetadata,
)
from customer_claims_rag.evaluation.doc12_threat_atomic_contract import (
    EXPECTED_DOC08_FINGERPRINT,
    HOLDOUT_POSITIVE_IDS,
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    THREAT_EXTENSION_IDS,
    assert_baseline_reproduction,
    build_doc12_reachability_comparison,
    build_extension_case_diagnostic,
    build_frozen_reachability_snapshot,
    classify_holdout_positive_delta,
    compute_extension_metrics,
    compute_holdout_metrics,
    compute_primary_hit_at_12,
    count_faq_in_top4,
    evaluate_extension_acceptance,
    evaluate_frozen_acceptance,
    evaluate_holdout_acceptance,
    fully_unreachable_case_ids,
    load_doc08_experimental_oracle,
    primary_unreachable_case_ids,
    repo_relative_path,
    validate_baseline_reproduction,
)
from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    Doc12ChunkDiffModel,
    Doc12ThreatAtomicCaseResult,
    Doc12ThreatAtomicEvaluationRun,
    Doc12Verdict,
    ExperimentVerdictSource,
    HoldoutEvaluationResult,
    ReplayIntegrityResult,
    ReplayIntegrityVerdict,
    ReplayStabilityResult,
)
from customer_claims_rag.evaluation.exact_frozen_arm import ExactFrozenArm
from customer_claims_rag.evaluation.git_state import read_git_state
from customer_claims_rag.evaluation.metrics import aggregate_case_metrics, compute_case_metrics
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    build_ranking_arm_result,
    chunks_from_search_results,
    compute_pool_reachability,
    rerank_to_final_top_k,
)
from customer_claims_rag.ingestion.corpus_overlay import (
    DocumentChunkDiff,
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
    compute_doc12_chunk_diff,
    verify_doc08_overlay_unchanged,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.ports import EmbeddingProvider
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, compute_config_hash
from customer_claims_rag.retrieval.retriever import BaselineRetriever


def _vector_ranks_by_document(results) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for index, result in enumerate(results, start=1):
        ranks.setdefault(result.document_id, []).append(index)
    return ranks


class Doc12ThreatAtomicAbEvaluator:
    """Compare doc08 experimental baseline vs doc12 overlay candidate."""

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
        holdout_cases: list,
        baseline_index_dir: Path,
        candidate_index_dir: Path,
        collection_name: str,
        embedding_model: str,
        experiment_id: str,
        config: dict,
        chunk_diff: DocumentChunkDiff,
        baseline_doc08_fingerprint: str,
        candidate_doc08_fingerprint: str,
        doc08_fingerprint_unchanged: bool,
        frozen_benchmark_fingerprint: str,
        extension_benchmark_fingerprint: str,
        holdout_benchmark_fingerprint: str,
        production_retrieval_config_hash: str,
        reference_artifact_path: Path,
        project_root: Path | None = None,
        allow_dirty_source: bool = False,
        replay_stability: ReplayStabilityResult | None = None,
        replay_integrity: ReplayIntegrityResult | None = None,
        exact_baseline_arm: ExactFrozenArm | None = None,
        exact_candidate_arm: ExactFrozenArm | None = None,
        query_embedding_provider: EmbeddingProvider | None = None,
        evaluation_backend: ExperimentVerdictSource = "exact_fixed_snapshot_evaluation",
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
        self.holdout_cases = holdout_cases
        self.baseline_index_dir = baseline_index_dir
        self.candidate_index_dir = candidate_index_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.experiment_id = experiment_id
        self.config = config
        self.chunk_diff = chunk_diff
        self.baseline_doc08_fingerprint = baseline_doc08_fingerprint
        self.candidate_doc08_fingerprint = candidate_doc08_fingerprint
        self.doc08_fingerprint_unchanged = doc08_fingerprint_unchanged
        self.frozen_benchmark_fingerprint = frozen_benchmark_fingerprint
        self.extension_benchmark_fingerprint = extension_benchmark_fingerprint
        self.holdout_benchmark_fingerprint = holdout_benchmark_fingerprint
        self.production_retrieval_config_hash = production_retrieval_config_hash
        self.reference_artifact_path = reference_artifact_path
        self.project_root = project_root
        self.allow_dirty_source = allow_dirty_source
        self.replay_stability = replay_stability
        self.replay_integrity = replay_integrity
        self.exact_baseline_arm = exact_baseline_arm
        self.exact_candidate_arm = exact_candidate_arm
        self.query_embedding_provider = query_embedding_provider or baseline_retriever.embedding_provider
        self.evaluation_backend = evaluation_backend
        self._candidate_vector_store = candidate_retriever.vector_store
        self.reference_oracle = load_doc08_experimental_oracle(
            reference_artifact_path,
            project_root=project_root,
        )

    def replace_candidate_index(self, candidate_index_dir: Path) -> None:
        """Point candidate arm at another rebuilt index (live-provider diagnostics only)."""
        self._candidate_vector_store.close()
        from customer_claims_rag.retrieval_config import RetrievalSettings

        settings = RetrievalSettings.from_env()
        embedding = create_embedding_provider(
            model_name=self.embedding_model,
            api_key=settings.openai_api_key,
        )
        store = create_vector_store(
            index_dir=candidate_index_dir,
            collection_name=self.collection_name,
        )
        self.candidate_index_dir = candidate_index_dir
        self._candidate_vector_store = store
        self.candidate_retriever = BaselineRetriever(
            embedding_provider=embedding,
            vector_store=store,
            index_dir=candidate_index_dir,
            top_k=self.fetch_k,
            fetch_k=self.fetch_k,
            similarity_threshold=self.threshold,
        )

    def evaluate(self) -> Doc12ThreatAtomicEvaluationRun:
        started = time.perf_counter()
        execution_ts = datetime.now(timezone.utc)
        if not self._uses_exact_backend():
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

        frozen_results = self._evaluate_cases(self.frozen_cases)
        extension_outcome = self._evaluate_extension_cases(self.extension_cases)
        holdout_outcome = self._evaluate_holdout_cases(self.holdout_cases)

        baseline_metrics_list = [item.baseline_case for item in frozen_results]
        candidate_metrics_list = [item.candidate_case for item in frozen_results]
        frozen_baseline_metrics = aggregate_case_metrics(baseline_metrics_list)
        frozen_candidate_metrics = aggregate_case_metrics(candidate_metrics_list)

        baseline_primary_hit_at_12 = compute_primary_hit_at_12(frozen_results, arm="baseline")
        candidate_primary_hit_at_12 = compute_primary_hit_at_12(frozen_results, arm="candidate")
        reachability = build_doc12_reachability_comparison(frozen_results)
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
            baseline_index_fingerprint=baseline_manifest.corpus_fingerprint,
            doc08_fingerprint=self.baseline_doc08_fingerprint,
        )
        baseline_reproduction = BaselineReproductionResult(
            reference_experiment_id=self.reference_oracle.experiment_id,
            reference_arm=self.reference_oracle.reference_arm,
            reference_artifact_path=self.reference_oracle.artifact_path,
            passed=all(check.passed for check in reproduction_checks),
            checks=reproduction_checks,
        )
        assert_baseline_reproduction(reproduction_checks)

        frozen_checks, frozen_verdict, case_delta = evaluate_frozen_acceptance(
            case_results=frozen_results,
            baseline_metrics=frozen_baseline_metrics,
            candidate_metrics=frozen_candidate_metrics,
            baseline_primary_hit_at_12=baseline_primary_hit_at_12,
            candidate_primary_hit_at_12=candidate_primary_hit_at_12,
        )

        ext_checks, ext_verdict = evaluate_extension_acceptance(
            extension_outcome["candidate_metrics"],
        )
        holdout_checks, holdout_verdict = evaluate_holdout_acceptance(
            holdout_outcome["candidate_metrics"],
        )

        doc08_unchanged_check_passed = self.doc08_fingerprint_unchanged
        all_checks = (
            reproduction_checks
            + frozen_checks
            + ext_checks
            + holdout_checks
            + [
                AcceptanceCheck(
                    criterion_id="doc08_overlay_unchanged",
                    description="Doc08 overlay unchanged between baseline and candidate corpora",
                    passed=doc08_unchanged_check_passed,
                    candidate_value=str(self.doc08_fingerprint_unchanged),
                    hard_rejection=not doc08_unchanged_check_passed,
                )
            ]
        )

        verdict: Doc12Verdict = (
            "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR"
            if frozen_verdict == "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR"
            and ext_verdict == "ACCEPTED AS TARGETED CORPUS REPAIR"
            and holdout_verdict == "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR"
            and doc08_unchanged_check_passed
            and all(check.passed for check in frozen_checks + ext_checks + holdout_checks)
            else "REJECTED"
        )

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

        return Doc12ThreatAtomicEvaluationRun(
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
            holdout_benchmark_fingerprint=self.holdout_benchmark_fingerprint,
            production_retrieval_config_hash=self.production_retrieval_config_hash,
            chunk_diff=Doc12ChunkDiffModel.from_dataclass(self.chunk_diff),
            doc08_fingerprint_unchanged=self.doc08_fingerprint_unchanged,
            expected_doc08_fingerprint=EXPECTED_DOC08_FINGERPRINT,
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
            primary_unreachable_baseline=primary_unreach_b,
            primary_unreachable_candidate=primary_unreach_c,
            fully_unreachable_baseline=fully_unreach_b,
            fully_unreachable_candidate=fully_unreach_c,
            promoted_cases_hit_at_4=case_delta["promoted_hit4"],
            regressed_cases_hit_at_4=case_delta["regressed_hit4"],
            promoted_cases_hit_at_12=case_delta["promoted_hit12"],
            regressed_cases_hit_at_12=case_delta["regressed_hit12"],
            extension=ExtensionEvaluationResult(
                benchmark_id=self.config["extension_benchmark"]["benchmark_id"],
                fingerprint=self.extension_benchmark_fingerprint,
                baseline=extension_outcome["baseline_metrics"],
                candidate=extension_outcome["candidate_metrics"],
                case_diagnostics=extension_outcome["diagnostics"],
                verdict=ext_verdict,
                acceptance_checks=ext_checks,
            ),
            holdout=HoldoutEvaluationResult(
                benchmark_id=self.config["holdout_benchmark"]["benchmark_id"],
                fingerprint=self.holdout_benchmark_fingerprint,
                baseline=holdout_outcome["baseline_metrics"],
                candidate=holdout_outcome["candidate_metrics"],
                case_diagnostics=holdout_outcome["diagnostics"],
                verdict=holdout_verdict,
                acceptance_checks=holdout_checks,
            ),
            replay_integrity=self.replay_integrity,
            replay_stability=self.replay_stability,
            replay_integrity_verdict=(
                self.replay_integrity.integrity_verdict
                if self.replay_integrity
                else (
                    self.replay_stability.integrity_verdict if self.replay_stability else None
                )
            ),
            ann_robustness=(
                self.replay_integrity.ann_robustness if self.replay_integrity else None
            ),
            ann_rebuild_stability=(
                self.replay_integrity.ann_rebuild_stability if self.replay_integrity else None
            ),
            exact_replay=(
                self.replay_integrity.exact_replay if self.replay_integrity else None
            ),
            environment=(
                self.replay_integrity.environment if self.replay_integrity else None
            ),
            experiment_verdict_source=self.evaluation_backend,
            acceptance_checks=all_checks,
            interpretation_boundary=[
                "Baseline arm is the accepted doc08 experimental candidate corpus; "
                "doc12 overlay must not mutate doc08 bytes.",
                "Frozen acceptance uses per-case hit@4/hit@12 non-regression against that baseline.",
                "Holdout threat-positive cases require doc12 to outrank or match doc08 when both appear.",
                "Extension threat failures where baseline already fails are not doc12 regressions.",
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
        index_path = self.baseline_index_dir if arm == "baseline" else self.candidate_index_dir
        return RetrievalArmMetadata(
            arm=arm,
            index_dir=repo_relative_path(index_path, self.project_root),
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
            chunk_payload_digest=manifest.chunk_payload_digest,
            embedding_digest=manifest.embedding_digest,
            collection_content_digest=manifest.collection_content_digest,
            build_run_id=manifest.build_run_id,
        )

    def _evaluate_cases(self, cases: list) -> list[Doc12ThreatAtomicCaseResult]:
        return [self._evaluate_single_case(case)["result"] for case in cases]

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

    def _evaluate_holdout_cases(self, cases: list) -> dict:
        baseline_cases = []
        candidate_cases = []
        diagnostics = []
        for case in cases:
            outcome = self._evaluate_single_case(case)
            baseline_cases.append(outcome["result"].baseline_case)
            candidate_cases.append(outcome["result"].candidate_case)
            baseline_case = outcome["result"].baseline_case
            candidate_case = outcome["result"].candidate_case
            if case.test_id in HOLDOUT_POSITIVE_IDS:
                delta = classify_holdout_positive_delta(
                    baseline=baseline_case,
                    candidate=candidate_case,
                )
            else:
                if baseline_case.primary_hit_at_4 and not candidate_case.primary_hit_at_4:
                    delta = "candidate_regression"
                elif not baseline_case.primary_hit_at_4 and candidate_case.primary_hit_at_4:
                    delta = "candidate_improves"
                elif baseline_case.primary_hit_at_4 == candidate_case.primary_hit_at_4:
                    delta = (
                        "unchanged_pass"
                        if baseline_case.primary_hit_at_4
                        else "both_fail_not_doc12_regression"
                    )
                else:
                    delta = "rank_order_changed"
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
                    baseline_case=baseline_case,
                    candidate_case=candidate_case,
                ).model_copy(update={"delta_classification": delta})
            )
        return {
            "baseline_metrics": compute_holdout_metrics(baseline_cases),
            "candidate_metrics": compute_holdout_metrics(candidate_cases),
            "diagnostics": diagnostics,
        }

    def _evaluate_single_case(self, case) -> dict:
        if self._uses_exact_backend():
            b_vector = self._exact_vector_search(case.query, arm="baseline")
            c_vector = self._exact_vector_search(case.query, arm="candidate")
        else:
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

        result = Doc12ThreatAtomicCaseResult(
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
        return {
            "result": result,
            "baseline_vector": b_vector,
            "candidate_vector": c_vector,
        }

    def _uses_exact_backend(self) -> bool:
        return (
            self.evaluation_backend == "exact_fixed_snapshot_evaluation"
            and self.exact_baseline_arm is not None
            and self.exact_candidate_arm is not None
        )

    def _exact_vector_search(self, query: str, *, arm: str) -> list:
        frozen_arm = (
            self.exact_baseline_arm if arm == "baseline" else self.exact_candidate_arm
        )
        if frozen_arm is None:
            raise RuntimeError(f"exact frozen arm missing for {arm}")
        query_vector = self.query_embedding_provider.embed_query(query)
        return frozen_arm.search(
            query_vector,
            fetch_k=self.fetch_k,
            threshold=self.threshold,
        )

    def with_ann_backend(self) -> Doc12ThreatAtomicAbEvaluator:
        """Return a shallow copy that evaluates through Chroma ANN (diagnostics only)."""
        baseline_store = create_vector_store(
            index_dir=self.baseline_index_dir,
            collection_name=self.collection_name,
        )
        candidate_store = create_vector_store(
            index_dir=self.candidate_index_dir,
            collection_name=self.collection_name,
        )
        baseline_retriever = BaselineRetriever(
            embedding_provider=self.baseline_retriever.embedding_provider,
            vector_store=baseline_store,
            index_dir=self.baseline_index_dir,
            top_k=self.fetch_k,
            fetch_k=self.fetch_k,
            similarity_threshold=self.threshold,
        )
        candidate_retriever = BaselineRetriever(
            embedding_provider=self.candidate_retriever.embedding_provider,
            vector_store=candidate_store,
            index_dir=self.candidate_index_dir,
            top_k=self.fetch_k,
            fetch_k=self.fetch_k,
            similarity_threshold=self.threshold,
        )
        clone = Doc12ThreatAtomicAbEvaluator(
            baseline_retriever=baseline_retriever,
            candidate_retriever=candidate_retriever,
            reranker=self.reranker,
            fetch_k=self.fetch_k,
            pool_k=self.pool_k,
            per_document_cap=self.per_document_cap,
            final_top_k=self.final_top_k,
            threshold=self.threshold,
            frozen_cases=self.frozen_cases,
            extension_cases=self.extension_cases,
            holdout_cases=self.holdout_cases,
            baseline_index_dir=self.baseline_index_dir,
            candidate_index_dir=self.candidate_index_dir,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            experiment_id=self.experiment_id,
            config=self.config,
            chunk_diff=self.chunk_diff,
            baseline_doc08_fingerprint=self.baseline_doc08_fingerprint,
            candidate_doc08_fingerprint=self.candidate_doc08_fingerprint,
            doc08_fingerprint_unchanged=self.doc08_fingerprint_unchanged,
            frozen_benchmark_fingerprint=self.frozen_benchmark_fingerprint,
            extension_benchmark_fingerprint=self.extension_benchmark_fingerprint,
            holdout_benchmark_fingerprint=self.holdout_benchmark_fingerprint,
            production_retrieval_config_hash=self.production_retrieval_config_hash,
            reference_artifact_path=self.reference_artifact_path,
            project_root=self.project_root,
            allow_dirty_source=self.allow_dirty_source,
            evaluation_backend="ann_chroma_evaluation",
        )
        return clone


def prepare_doc12_chunk_diff_and_fingerprints(
    *,
    canonical_dir: Path,
    doc08_overlay_path: Path,
    doc12_overlay_path: Path,
    permitted_root: Path,
    embedding_model: str,
    expected_doc08_fingerprint: str | None = None,
) -> tuple[DocumentChunkDiff, str, str, bool]:
    overlay_result = build_doc12_experimental_corpora(
        canonical_dir=canonical_dir,
        doc08_overlay_path=doc08_overlay_path,
        doc12_overlay_path=doc12_overlay_path,
        permitted_root=permitted_root,
    )
    try:
        verify_doc08_overlay_unchanged(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
            expected_doc08_fingerprint=expected_doc08_fingerprint,
        )
        diff = compute_doc12_chunk_diff(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
        )
        if not diff.non_target_byte_identical:
            raise ValueError("non-doc12 chunks are not byte-identical")
        baseline_doc08_fp = overlay_result.baseline_doc08_fingerprint
        candidate_doc08_fp = overlay_result.candidate_doc08_fingerprint
        doc08_unchanged = baseline_doc08_fp == candidate_doc08_fp
        return diff, baseline_doc08_fp, candidate_doc08_fp, doc08_unchanged
    finally:
        cleanup_overlay_temp_dir(overlay_result.temp_input_dir)


__all__ = [
    "Doc12ThreatAtomicAbEvaluator",
    "prepare_doc12_chunk_diff_and_fingerprints",
]
