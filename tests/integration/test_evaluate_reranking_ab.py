"""Integration tests for reranking A/B evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.cli import evaluate_reranking_ab as ab_cli
from customer_claims_rag.evaluation.ab_evaluator import RerankingAbEvaluator
from customer_claims_rag.evaluation.ab_reporting import PROTECTED_BASELINE_PATHS
from customer_claims_rag.evaluation.models import EvaluationRun
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult
from customer_claims_rag.retrieval.reranker import load_reranker_config
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker
from tests.frozen_fixtures import FROZEN_RETRIEVAL_BASELINE as FROZEN_BASELINE
from tests.frozen_fixtures import load_frozen_retrieval_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"
CONFIG = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"


class FrozenBaselineRetriever:
    """Return frozen baseline search results without calling Chroma."""

    def __init__(self, frozen_run: EvaluationRun) -> None:
        self._cases = {case.query: case for case in frozen_run.case_results}
        self.calls: list[str] = []
        self.vector_store = MagicMock()
        self.vector_store.collection_name = "customer_claims"
        self.vector_store.count.return_value = 215

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
        return SearchResponse(
            query=query,
            top_k=12,
            fetch_k=12,
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


def test_one_retrieval_call_per_case(frozen_run: EvaluationRun, monkeypatch) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
    assert run.shared_context.case_count == 60


def test_shared_candidate_ids_identical(frozen_run: EvaluationRun, monkeypatch) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)[:5]
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
        baseline_ids = [item.chunk_id for item in case.baseline.ordered_candidates]
        candidate_ids = sorted(
            item.chunk_id for item in case.candidate.ordered_candidates
        )
        assert sorted(baseline_ids) == candidate_ids
        assert case.comparison.shared_pool_match is True


def test_baseline_arm_matches_frozen_baseline(frozen_run: EvaluationRun, monkeypatch) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
    identity = run.baseline_identity
    assert identity.per_case_match is True
    assert identity.identity_status in {"exact", "equivalent_with_embedding_drift"}
    assert identity.evaluation_result_id_match is True


def test_candidate_output_contains_same_chunks(frozen_run: EvaluationRun, monkeypatch) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)[:10]
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
        baseline_set = {item.chunk_id for item in case.baseline.ordered_candidates}
        candidate_set = {item.chunk_id for item in case.candidate.ordered_candidates}
        assert baseline_set == candidate_set


def test_fallback_cases_preserve_status(frozen_run: EvaluationRun, monkeypatch) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
    baseline_map = {case.test_id: case.status for case in run.baseline.case_results}
    candidate_map = {case.test_id: case.status for case in run.candidate.case_results}
    assert baseline_map["T006"] == "no_grounding"
    assert candidate_map["T006"] == "no_grounding"
    assert baseline_map["T060"] == "no_grounding"
    assert candidate_map["T060"] == "no_grounding"


def test_t040_t047_and_unreachable_cases_marked(
    frozen_run: EvaluationRun,
    monkeypatch,
) -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    retriever = FrozenBaselineRetriever(frozen_run)
    config = load_reranker_config(CONFIG)
    evaluator = RerankingAbEvaluator(
        retriever=retriever,
        reranker=SourceAuthorityV1Reranker(config),
        config=config,
        index_dir=Path("data/04_index"),
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
        frozen_baseline_path=FROZEN_BASELINE,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.ab_evaluator.load_manifest",
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
    t004 = next(case for case in run.case_results if case.case_id == "T004")
    t040 = next(case for case in run.case_results if case.case_id == "T040")
    t047 = next(case for case in run.case_results if case.case_id == "T047")
    t023 = next(case for case in run.case_results if case.case_id == "T023")
    assert t040.comparison.reranker_not_applicable is True
    assert t047.comparison.reranker_not_applicable is True
    assert t004.comparison.reranker_not_applicable is True
    assert t023.comparison.reranker_not_applicable is False
    assert t023.comparison.primary_hit_at_4_delta > 0


def test_cli_rejects_overwriting_baseline_json(temp_project: Path, capsys) -> None:
    code = ab_cli.run_reranking_ab(
        config_path=CONFIG,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        frozen_baseline_path=FROZEN_BASELINE,
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        output_json=temp_project / "data" / "05_evaluation" / "retrieval_results.json",
        output_markdown=temp_project / "tests" / "06_reranking_ab_results.md",
        project_root_path=temp_project.resolve(),
    )[0]
    captured = capsys.readouterr()
    assert code == 2
    assert "must not overwrite baseline artifact" in captured.err


def test_protected_baseline_files_unchanged_after_ab_run(
    temp_project: Path, monkeypatch, dummy_openai_api_key: str
) -> None:
    for relative in PROTECTED_BASELINE_PATHS:
        source = PROJECT_ROOT / relative
        if not source.exists():
            continue
        target = temp_project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    frozen = temp_project / "data" / "05_evaluation" / "retrieval_results.json"
    before = {
        path: (temp_project / path).read_text(encoding="utf-8")
        for path in PROTECTED_BASELINE_PATHS
        if (temp_project / path).exists()
    }

    mock_evaluator = MagicMock()
    mock_run = MagicMock()
    mock_run.model_dump.return_value = {"timestamp": "2026-01-01T00:00:00Z"}
    mock_run.comparison.acceptance.candidate_accepted = False
    mock_run.experiment.reranker_id = "source-authority-v1"
    mock_run.experiment.version = "1.0.0"
    mock_run.experiment.config_hash = "abc"
    mock_run.baseline.aggregate_metrics.primary_source_hit_rate_at_4 = 0.621
    mock_run.candidate.aggregate_metrics.primary_source_hit_rate_at_4 = 0.650
    mock_run.comparison.aggregate.primary_source_hit_rate_at_4_delta = 0.029
    mock_run.baseline_identity = MagicMock()
    mock_run.baseline_identity.identity_status = "equivalent_with_embedding_drift"
    mock_evaluator.evaluate.return_value = mock_run
    monkeypatch.setattr(ab_cli, "RerankingAbEvaluator", MagicMock(from_paths=lambda **kwargs: mock_evaluator))
    monkeypatch.setattr(
        ab_cli,
        "write_ab_outputs",
        lambda run, **kwargs: (
            temp_project / "data" / "05_evaluation" / "reranking_ab_source_authority_v1.json",
            temp_project / "tests" / "06_reranking_ab_results.md",
        ),
    )

    code = ab_cli.run_reranking_ab(
        config_path=CONFIG,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        frozen_baseline_path=frozen,
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root_path=temp_project.resolve(),
    )[0]
    assert code == 0
    for path, content in before.items():
        assert (temp_project / path).read_text(encoding="utf-8") == content
