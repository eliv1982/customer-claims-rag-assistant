"""Integration tests for vector pool expansion evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.cli import evaluate_pool_expansion as pool_cli
from customer_claims_rag.evaluation.models import EvaluationRun
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.pool_expansion_evaluator import PoolExpansionEvaluator
from customer_claims_rag.evaluation.pool_expansion_metrics import FROZEN_RERANKER_CONFIG_HASH, load_pool_expansion_config
from customer_claims_rag.evaluation.pool_expansion_models import PoolExpansionEvaluationRun
from customer_claims_rag.evaluation.pool_expansion_reporting import render_pool_expansion_markdown
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, compute_config_hash, load_reranker_config
from tests.frozen_fixtures import load_frozen_retrieval_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
RERANKER_CONFIG = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"


class DeepPoolRetriever:
    """Return 24-result pools based on frozen baseline top-12 plus synthetic tail."""

    def __init__(self, frozen_run: EvaluationRun) -> None:
        self._cases = {case.query: case for case in frozen_run.case_results}
        self.calls: list[str] = []

    def validate_index(self) -> None:
        return None

    def search(self, query: str) -> SearchResponse:
        self.calls.append(query)
        case = self._cases[query]
        results = [
            SearchResult(
                rank=chunk.rank,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=f"content for {chunk.document_id}",
                source_path=chunk.source_path,
                chunk_type=_chunk_type_for(chunk.chunk_id),
                topic=None,
                risk_level=None,
                heading=chunk.heading,
                heading_path=[chunk.heading],
                section=None,
                subsection=None,
                similarity=chunk.similarity,
                distance=chunk.distance,
            )
            for chunk in case.retrieved_chunks
        ]
        tail_start = len(results) + 1
        for index in range(tail_start, 25):
            doc = f"synthetic_doc_{index}"
            results.append(
                SearchResult(
                    rank=index,
                    chunk_id=f"synthetic_chunk_{index}",
                    document_id=doc,
                    content=f"content for {doc}",
                    source_path=f"data/02_clean_markdown/{doc}.md",
                    chunk_type="faq",
                    topic=None,
                    risk_level=None,
                    heading="Synthetic",
                    heading_path=["Synthetic"],
                    section=None,
                    subsection=None,
                    similarity=max(0.05, 0.30 - index * 0.005),
                    distance=0.7,
                )
            )
        return SearchResponse(
            query=query,
            top_k=24,
            fetch_k=24,
            similarity_threshold=0.0,
            results=results,
            candidates_fetched=len(results),
            candidates_above_threshold=len(results),
            embedding_model="text-embedding-3-small",
            collection_name="customer_claims",
        )


def _chunk_type_for(chunk_id: str) -> str:
    if chunk_id.startswith("10_customer_faq"):
        return "faq"
    if chunk_id.startswith("08_escalation"):
        return "escalation"
    if chunk_id.startswith("07_complaint"):
        return "procedure"
    if chunk_id.startswith("09_response"):
        return "templates"
    if chunk_id.startswith("01_service"):
        return "reference"
    return "policy"


@pytest.fixture
def frozen_run() -> EvaluationRun:
    return load_frozen_retrieval_baseline()


def _build_evaluator(frozen_run: EvaluationRun, cases_count: int | None = None) -> PoolExpansionEvaluator:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    if cases_count is not None:
        cases = cases[:cases_count]
    config = load_pool_expansion_config(EXPERIMENT_CONFIG)
    reranker = SourceAuthorityV1Reranker(load_reranker_config(RERANKER_CONFIG))
    return PoolExpansionEvaluator(
        retriever=DeepPoolRetriever(frozen_run),
        reranker=reranker,
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
    )


def test_one_retrieval_call_per_case(frozen_run: EvaluationRun, monkeypatch) -> None:
    retriever = DeepPoolRetriever(frozen_run)
    evaluator = _build_evaluator(frozen_run)
    evaluator.retriever = retriever
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.pool_expansion_evaluator.load_manifest",
        lambda _index_dir: MagicMock(
            corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
            index_format_version="1.0.0",
            metadata_schema_version="1.0.0",
            vector_dimension=1536,
            chunk_count=215,
            document_count=10,
        ),
    )
    run = evaluator.evaluate()
    assert len(retriever.calls) == 60
    assert run.acceptance.single_retrieval_pass is True


def test_baseline_pool_is_prefix_of_candidate_pool(frozen_run: EvaluationRun, monkeypatch) -> None:
    evaluator = _build_evaluator(frozen_run, cases_count=10)
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.pool_expansion_evaluator.load_manifest",
        lambda _index_dir: MagicMock(
            corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
            index_format_version="1.0.0",
            metadata_schema_version="1.0.0",
            vector_dimension=1536,
            chunk_count=215,
            document_count=10,
        ),
    )
    run = evaluator.evaluate()
    for case in run.case_results:
        assert case.comparison.baseline_pool_is_prefix_of_candidate_pool is True
        assert case.baseline_pool.candidate_ids == case.candidate_pool.candidate_ids[:12]
    assert run.acceptance.shared_prefix_pass is True


def test_frozen_reranker_hash_matches(frozen_run: EvaluationRun, monkeypatch) -> None:
    evaluator = _build_evaluator(frozen_run, cases_count=3)
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.pool_expansion_evaluator.load_manifest",
        lambda _index_dir: MagicMock(
            corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
            index_format_version="1.0.0",
            metadata_schema_version="1.0.0",
            vector_dimension=1536,
            chunk_count=215,
            document_count=10,
        ),
    )
    run = evaluator.evaluate()
    assert compute_config_hash(load_reranker_config(RERANKER_CONFIG)) == FROZEN_RERANKER_CONFIG_HASH
    assert run.acceptance.reranker_config_hash_match is True


def test_json_round_trip_and_markdown_from_json(frozen_run: EvaluationRun, monkeypatch) -> None:
    evaluator = _build_evaluator(frozen_run, cases_count=5)
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.pool_expansion_evaluator.load_manifest",
        lambda _index_dir: MagicMock(
            corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
            index_format_version="1.0.0",
            metadata_schema_version="1.0.0",
            vector_dimension=1536,
            chunk_count=215,
            document_count=10,
        ),
    )
    run = evaluator.evaluate()
    payload = run.model_dump(mode="json")
    restored = PoolExpansionEvaluationRun.model_validate(payload)
    markdown = render_pool_expansion_markdown(restored)
    assert "Candidate-generation verdict" in markdown
    assert "Primary hit@4 did not increase" in markdown or "did not increase" in markdown
    assert "exact frozen baseline identity confirmed" not in markdown.lower()


def test_protected_2c1_artifact_not_overwritten(temp_project: Path) -> None:
    from customer_claims_rag.evaluation.pool_expansion_reporting import write_pool_expansion_outputs
    from customer_claims_rag.exceptions import EvaluationOutputError

    mock_run = MagicMock()
    mock_run.model_dump.return_value = {"timestamp": "2026-01-01T00:00:00Z"}
    with pytest.raises(EvaluationOutputError, match="protected artifact"):
        write_pool_expansion_outputs(
            mock_run,
            output_json=temp_project / "data" / "05_evaluation" / "reranking_ab_source_authority_v1.json",
            output_markdown=temp_project / "tests" / "07_vector_pool_expansion_results.md",
            project_root=temp_project.resolve(),
        )
