"""Hybrid lexical + vector A/B evaluator with shared vector retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.git_state import read_git_state
from customer_claims_rag.evaluation.hybrid_metrics import (
    build_baseline_ranking_arm_result,
    build_candidate_ranking_arm_result,
    build_case_comparison_detail,
    build_channel_reachability_comparison,
    build_ranking_comparison,
    channel_flags_from_label,
    compute_channel_reachability,
    compute_evaluation_dataset_fingerprint,
    compute_hybrid_config_hash,
    evaluate_acceptance_criteria,
    lexical_pool_snapshot_from_hits,
    load_hybrid_config,
    prepare_vector_pool,
    rerank_baseline_to_final_top_k,
)
from customer_claims_rag.evaluation.hybrid_models import (
    ExperimentMetadata,
    FusionPoolSnapshot,
    HybridCaseResult,
    HybridEvaluationRun,
    HybridExperimentConfig,
    LexicalIndexMetadata,
    SharedContext,
    VectorPoolSnapshot,
)
from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    aggregate_category_metrics,
    aggregate_risk_metrics,
    compute_case_metrics,
    summarize_fallback_cases,
)
from customer_claims_rag.evaluation.models import CaseResult, EvaluationRun, RunMetadata
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.fusion import FusionCandidate, fuse_rankings
from customer_claims_rag.retrieval.lexical.bm25 import BM25Index
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk, load_lexical_corpus_from_chroma
from customer_claims_rag.retrieval.lexical.preprocessor import TOKENIZER_VERSION
from customer_claims_rag.retrieval.lexical.retriever import LexicalRetriever
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import (
    SourceAuthorityV1Reranker,
    compute_config_hash,
    load_reranker_config,
)
from customer_claims_rag.retrieval.rrf_adapter import fusion_to_search_results, rerank_fused_candidates
from customer_claims_rag.retrieval.retriever import BaselineRetriever


class HybridAbEvaluator:
    """Run hybrid lexical + vector A/B evaluation with one vector call per case."""

    def __init__(
        self,
        *,
        retriever: BaselineRetriever,
        lexical_retriever: LexicalRetriever,
        lexical_index: BM25Index,
        chunk_lookup: dict[str, LexicalChunk],
        reranker: SourceAuthorityV1Reranker,
        config: HybridExperimentConfig,
        index_dir: Path,
        cases: list,
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.lexical_retriever = lexical_retriever
        self.lexical_index = lexical_index
        self.chunk_lookup = chunk_lookup
        self.document_ids = {chunk_id: chunk.document_id for chunk_id, chunk in chunk_lookup.items()}
        self.reranker = reranker
        self.config = config
        self.index_dir = index_dir
        self.cases = cases
        self.questions_path = questions_path
        self.expected_path = expected_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.project_root = project_root

    @classmethod
    def from_paths(
        cls,
        *,
        retriever: BaselineRetriever,
        experiment_config_path: Path,
        reranker_config_path: Path,
        index_dir: Path,
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> "HybridAbEvaluator":
        config = load_hybrid_config(experiment_config_path)
        manifest = load_manifest(index_dir)
        chunks = load_lexical_corpus_from_chroma(
            index_dir=index_dir,
            collection_name=collection_name,
            expected_fingerprint=manifest.corpus_fingerprint,
        )
        lexical_index = BM25Index(chunks, k1=config.bm25_k1, b=config.bm25_b)
        lexical_retriever = LexicalRetriever(lexical_index)
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        reranker = SourceAuthorityV1Reranker(load_reranker_config(reranker_config_path))
        cases = load_evaluation_corpus(
            questions_path=questions_path,
            expected_path=expected_path,
        )
        return cls(
            retriever=retriever,
            lexical_retriever=lexical_retriever,
            lexical_index=lexical_index,
            chunk_lookup=chunk_lookup,
            reranker=reranker,
            config=config,
            index_dir=index_dir,
            cases=cases,
            questions_path=questions_path,
            expected_path=expected_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=project_root,
        )

    def validate_index(self) -> None:
        self.retriever.validate_index()

    def evaluate(self) -> HybridEvaluationRun:
        self.validate_index()
        manifest = load_manifest(self.index_dir)
        git_state = read_git_state(self.project_root)
        dataset_fingerprint = compute_evaluation_dataset_fingerprint(
            questions_path=self.questions_path,
            expected_path=self.expected_path,
        )

        case_results: list[HybridCaseResult] = []
        baseline_ranking_cases: list[CaseResult] = []
        candidate_ranking_cases: list[CaseResult] = []
        retrieval_calls = 0

        for case in self.cases:
            outcome = self._evaluate_case(case)
            retrieval_calls += 1
            case_results.append(outcome["case_result"])
            baseline_ranking_cases.append(outcome["baseline_ranking_case"])
            candidate_ranking_cases.append(outcome["candidate_ranking_case"])

        if retrieval_calls != len(self.cases):
            raise RuntimeError("retrieval call count mismatch")

        run_metadata_base = self._build_run_metadata(manifest=manifest, git_state=git_state)
        baseline_run = self._build_evaluation_run(baseline_ranking_cases, run_metadata_base)
        candidate_run = self._build_evaluation_run(candidate_ranking_cases, run_metadata_base)

        reachability = build_channel_reachability_comparison(case_results)
        ranking = build_ranking_comparison(
            baseline_cases=baseline_ranking_cases,
            candidate_cases=candidate_ranking_cases,
            case_results=case_results,
        )
        technical_errors_zero = candidate_run.aggregate_metrics.technical_error_count == 0
        acceptance = evaluate_acceptance_criteria(
            config=self.config,
            reranker_config_hash=compute_config_hash(self.reranker.config),
            reachability=reachability,
            ranking=ranking,
            case_results=case_results,
            single_retrieval_pass=True,
            technical_errors_zero=technical_errors_zero,
            lexical_index_valid=True,
        )

        experiment_config_payload = json.loads(
            Path(self.config.config_path or "configs/retrieval/hybrid_lexical_vector_v1.json").read_text(
                encoding="utf-8",
            )
        )
        experiment = ExperimentMetadata(
            experiment_id=self.config.experiment_id,
            version=self.config.version,
            experiment_mode=self.config.experiment_mode,
            config=experiment_config_payload,
            config_hash=compute_hybrid_config_hash(self.config),
            reranker_id=self.config.reranker_id,
            reranker_config_hash=self.config.reranker_config_hash,
            score_adapter_id=self.config.score_adapter_id,
            git_commit=git_state.commit,
            git_dirty=git_state.dirty,
            git_status_summary=git_state.status_summary,
        )
        shared_context = SharedContext(
            index_fingerprint=manifest.corpus_fingerprint,
            embedding_model=self.embedding_model,
            evaluation_dataset_fingerprint=dataset_fingerprint,
            case_count=len(self.cases),
            vector_k=self.config.vector_k,
            lexical_k=self.config.lexical_k,
            fusion_k=self.config.fusion_k,
            final_top_k=self.config.final_top_k,
            threshold=self.config.threshold,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
        )
        lexical_index_meta = LexicalIndexMetadata(
            lexical_algorithm=self.config.lexical_algorithm,
            tokenizer_version=TOKENIZER_VERSION,
            bm25_k1=self.config.bm25_k1,
            bm25_b=self.config.bm25_b,
            chunk_count=self.lexical_index.chunk_count,
            corpus_fingerprint=manifest.corpus_fingerprint,
            lexical_index_fingerprint=self.lexical_index.compute_fingerprint(
                corpus_fingerprint=manifest.corpus_fingerprint,
            ),
        )

        return HybridEvaluationRun(
            timestamp=datetime.now(timezone.utc),
            experiment=experiment,
            shared_context=shared_context,
            lexical_index=lexical_index_meta,
            baseline=baseline_run,
            candidate=candidate_run,
            candidate_generation_comparison=reachability,
            ranking_comparison=ranking,
            case_results=case_results,
            acceptance=acceptance,
        )

    def _evaluate_case(self, case) -> dict[str, object]:
        try:
            response = self.retriever.search(case.query)
        except SearchError as exc:
            return self._error_case(case, str(exc))

        vector_results = list(response.results)
        vector_pool = prepare_vector_pool(vector_results, self.config.vector_k)
        lexical_hits = self.lexical_retriever.search(case.query, k=self.config.lexical_k)

        fused = fuse_rankings(
            vector_results=vector_pool,
            lexical_results=lexical_hits,
            fusion_k=self.config.fusion_k,
            rrf_k=self.config.rrf_k,
            vector_weight=self.config.vector_weight,
            lexical_weight=self.config.lexical_weight,
            document_ids=self.document_ids,
        )
        fusion_pool = fusion_to_search_results(fused, chunk_lookup=self.chunk_lookup)

        baseline_reach, fusion_reach, channel, vec_only_flag, lex_only_flag, both_flag = (
            compute_channel_reachability(
                vector_pool=vector_pool,
                fusion_pool=fusion_pool,
                lexical_pool=lexical_hits,
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
                fallback_expected=case.fallback_expected,
            )
        )
        vector_reachable, lexical_reachable, _, _, _ = channel_flags_from_label(channel)

        baseline_final, baseline_reranked = rerank_baseline_to_final_top_k(
            self.reranker,
            case.query,
            vector_pool,
            final_top_k=self.config.final_top_k,
        )
        candidate_final, candidate_audits = rerank_fused_candidates(
            reranker=self.reranker,
            query=case.query,
            fused=fused,
            chunk_lookup=self.chunk_lookup,
            final_top_k=self.config.final_top_k,
        )

        baseline_ranking_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=self._chunks_from_results(baseline_final),
        )
        candidate_ranking_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=self._chunks_from_results(candidate_final),
        )

        baseline_ranking = build_baseline_ranking_arm_result(
            pool=vector_pool,
            final_results=baseline_final,
            reranked=baseline_reranked,
            case=baseline_ranking_case,
            pool_reachability=baseline_reach,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )
        candidate_ranking = build_candidate_ranking_arm_result(
            final_results=candidate_final,
            audits=candidate_audits,
            case=candidate_ranking_case,
            pool_reachability=fusion_reach,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )
        comparison = build_case_comparison_detail(
            baseline_metrics=baseline_ranking_case,
            candidate_metrics=candidate_ranking_case,
            baseline_ranking_ids=[item.chunk_id for item in baseline_final],
            candidate_ranking_ids=[item.chunk_id for item in candidate_final],
        )

        case_result = HybridCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_vector_candidate_ids=[item.chunk_id for item in vector_results],
            baseline_pool=VectorPoolSnapshot(
                pool_k=self.config.vector_k,
                candidate_ids=[item.chunk_id for item in vector_pool],
                reachability=baseline_reach,
            ),
            lexical_pool=lexical_pool_snapshot_from_hits(
                lexical_hits,
                pool_k=self.config.lexical_k,
                exact=True,
            ),
            candidate_fusion_pool=FusionPoolSnapshot(
                pool_k=self.config.fusion_k,
                candidate_ids=[item.chunk_id for item in fusion_pool],
                reachability=fusion_reach,
                vector_reachable=vector_reachable,
                lexical_reachable=lexical_reachable,
                retrieval_channel=channel,
                vector_only_reachable=vec_only_flag,
                lexical_only_reachable=lex_only_flag,
                both_reachable=both_flag,
            ),
            baseline_ranking=baseline_ranking,
            candidate_ranking=candidate_ranking,
            comparison=comparison,
        )
        return {
            "case_result": case_result,
            "baseline_ranking_case": baseline_ranking_case,
            "candidate_ranking_case": candidate_ranking_case,
        }

    def _error_case(self, case, message: str) -> dict[str, object]:
        from customer_claims_rag.evaluation.pool_expansion_metrics import compute_pool_reachability
        from customer_claims_rag.evaluation.hybrid_models import HybridRankingArmResult
        from customer_claims_rag.evaluation.ab_models import ArmMetrics

        error_case = CaseResult(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            status="technical_error",
            error_message=message,
        )
        empty_reach = compute_pool_reachability(
            [],
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )
        empty_ranking = HybridRankingArmResult(metrics=ArmMetrics())
        case_result = HybridCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_vector_candidate_ids=[],
            baseline_pool=VectorPoolSnapshot(pool_k=0, reachability=empty_reach),
            candidate_fusion_pool=FusionPoolSnapshot(pool_k=0, reachability=empty_reach),
            baseline_ranking=empty_ranking,
            candidate_ranking=empty_ranking,
            comparison=build_case_comparison_detail(
                baseline_metrics=error_case,
                candidate_metrics=error_case,
                baseline_ranking_ids=[],
                candidate_ranking_ids=[],
            ),
        )
        return {
            "case_result": case_result,
            "baseline_ranking_case": error_case,
            "candidate_ranking_case": error_case,
        }

    @staticmethod
    def _chunks_from_results(results: list[SearchResult]):
        from customer_claims_rag.evaluation.ab_metrics import chunks_from_search_results

        return chunks_from_search_results(results)

    def _build_run_metadata(self, *, manifest, git_state) -> RunMetadata:
        return RunMetadata(
            timestamp=datetime.now(timezone.utc),
            evaluation_result_id="pending",
            git_commit=git_state.commit,
            git_dirty=git_state.dirty,
            git_status_summary=git_state.status_summary,
            index_fingerprint=manifest.corpus_fingerprint,
            index_format_version=manifest.index_format_version,
            metadata_schema_version=manifest.metadata_schema_version,
            embedding_model=self.embedding_model,
            vector_dimension=manifest.vector_dimension,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
            evaluation_case_count=len(self.cases),
            threshold=self.config.threshold,
            top_k=self.config.final_top_k,
            fetch_k=self.config.vector_k,
        )

    def _build_evaluation_run(
        self,
        case_results: list[CaseResult],
        run_metadata_base: RunMetadata,
    ) -> EvaluationRun:
        technical_errors = [
            f"{case.test_id}: {case.error_message}"
            for case in case_results
            if case.status == "technical_error" and case.error_message
        ]
        return EvaluationRun(
            run_metadata=run_metadata_base,
            aggregate_metrics=aggregate_case_metrics(case_results),
            risk_metrics=aggregate_risk_metrics(case_results),
            category_metrics=aggregate_category_metrics(case_results),
            threshold_analysis=analyze_thresholds(case_results),
            case_results=case_results,
            technical_errors=technical_errors,
            fallback_analysis=summarize_fallback_cases(case_results),
        )
