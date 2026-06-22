"""Integration tests for hybrid lexical + vector evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_claims_rag.cli import evaluate_hybrid_ab as hybrid_cli
from customer_claims_rag.evaluation.hybrid_evaluator import HybridAbEvaluator
from customer_claims_rag.evaluation.hybrid_metrics import FROZEN_RERANKER_CONFIG_HASH, load_hybrid_config
from customer_claims_rag.evaluation.hybrid_models import HybridEvaluationRun
from customer_claims_rag.evaluation.hybrid_reporting import render_hybrid_markdown
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.retrieval.lexical.bm25 import BM25Index
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.lexical.retriever import LexicalRetriever
from customer_claims_rag.retrieval.manifest import IndexManifest
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, load_reranker_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "hybrid_lexical_vector_v1.json"
RERANKER_CONFIG = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"
POOL_ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "vector_pool_expansion_v1.json"


def _fake_manifest() -> IndexManifest:
    return IndexManifest(
        index_format_version="1.0.0",
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
        chunk_count=215,
        document_count=10,
        metadata_schema_version="1.0.0",
        vector_dimension=1536,
    )


class SharedVectorRetriever:
    """Reuse frozen pool-expansion vector rankings with one call per query."""

    def __init__(self, artifact: dict) -> None:
        self._cases = {item["query"]: item for item in artifact["case_results"]}
        self.calls: list[str] = []

    def validate_index(self) -> None:
        return None

    def search(self, query: str) -> SearchResponse:
        self.calls.append(query)
        case = self._cases[query]
        results = []
        for rank, chunk_id in enumerate(case["shared_deep_candidate_ids"], start=1):
            document_id = chunk_id.split("::", 1)[0]
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=f"content for {chunk_id}",
                    source_path=f"data/02_clean_markdown/{document_id}.md",
                    chunk_type=_chunk_type_for(document_id),
                    topic=None,
                    risk_level=None,
                    heading="Heading",
                    heading_path=["Heading"],
                    section=None,
                    subsection=None,
                    similarity=max(0.05, 0.5 - rank * 0.005),
                    distance=0.5,
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


def _chunk_type_for(document_id: str) -> str:
    if document_id == "10_customer_faq":
        return "faq"
    if document_id == "08_escalation_and_risk_rules":
        return "escalation"
    if document_id == "07_complaint_handling_procedure":
        return "procedure"
    return "policy"


def _synthetic_lexical_chunks() -> list[LexicalChunk]:
    return [
        LexicalChunk(
            chunk_id="07_complaint_handling_procedure::chunk-019",
            document_id="07_complaint_handling_procedure",
            content="не запрашивать CVV или CVC",
            heading="Безопасность",
            chunk_type="procedure",
            source_path="data/02_clean_markdown/07_complaint_handling_procedure.md",
            metadata={"heading_path": "[\"Безопасность\"]"},
        ),
        LexicalChunk(
            chunk_id="06_food_quality_and_packaging::chunk-016",
            document_id="06_food_quality_and_packaging",
            content="металлическая осколка",
            heading="Качество",
            chunk_type="policy",
            source_path="data/02_clean_markdown/06_food_quality_and_packaging.md",
            metadata={"heading_path": "[\"Качество\"]"},
        ),
    ]


def _chunks_from_pool_artifact(artifact: dict) -> list[LexicalChunk]:
    seen: dict[str, LexicalChunk] = {}
    for case in artifact["case_results"]:
        for chunk_id in case["shared_deep_candidate_ids"]:
            if chunk_id in seen:
                continue
            document_id = chunk_id.split("::", 1)[0]
            seen[chunk_id] = LexicalChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=f"content for {chunk_id}",
                heading="Heading",
                chunk_type=_chunk_type_for(document_id),
                source_path=f"data/02_clean_markdown/{document_id}.md",
                metadata={"heading_path": "[\"Heading\"]"},
            )
    for extra in _synthetic_lexical_chunks():
        seen[extra.chunk_id] = extra
    return sorted(seen.values(), key=lambda item: item.chunk_id)


@pytest.fixture
def pool_artifact() -> dict:
    return json.loads(POOL_ARTIFACT.read_text(encoding="utf-8"))


def _build_evaluator(
    pool_artifact: dict,
    tmp_path: Path,
    *,
    case_limit: int | None = None,
) -> tuple[HybridAbEvaluator, SharedVectorRetriever]:
    retriever = SharedVectorRetriever(pool_artifact)
    chunks = _chunks_from_pool_artifact(pool_artifact)
    lexical_index = BM25Index(chunks)
    config = load_hybrid_config(EXPERIMENT_CONFIG)
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    if case_limit is not None:
        cases = cases[:case_limit]
    evaluator = HybridAbEvaluator(
        retriever=retriever,
        lexical_retriever=LexicalRetriever(lexical_index),
        lexical_index=lexical_index,
        chunk_lookup={chunk.chunk_id: chunk for chunk in chunks},
        reranker=SourceAuthorityV1Reranker(load_reranker_config(RERANKER_CONFIG)),
        config=config,
        index_dir=tmp_path,
        cases=cases,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        project_root=PROJECT_ROOT,
    )
    return evaluator, retriever


def test_evaluator_one_vector_call_per_case(pool_artifact: dict, tmp_path: Path) -> None:
    evaluator, retriever = _build_evaluator(pool_artifact, tmp_path)
    with patch.object(evaluator, "validate_index", return_value=None), patch(
        "customer_claims_rag.evaluation.hybrid_evaluator.load_manifest",
        return_value=_fake_manifest(),
    ):
        run = evaluator.evaluate()
    assert len(retriever.calls) == 60
    assert run.shared_context.vector_retrieval_calls_per_case == 1
    assert run.acceptance.reranker_config_hash_match
    assert all(len(case.baseline_ranking.ordered_candidates) <= 12 for case in run.case_results)


def test_markdown_rebuild_from_artifact_shape(pool_artifact: dict, tmp_path: Path) -> None:
    evaluator, _ = _build_evaluator(pool_artifact, tmp_path, case_limit=3)
    with patch.object(evaluator, "validate_index", return_value=None), patch(
        "customer_claims_rag.evaluation.hybrid_evaluator.load_manifest",
        return_value=_fake_manifest(),
    ):
        run = evaluator.evaluate()
    markdown = render_hybrid_markdown(run)
    assert "rrf-base-score-adapter-v1" in markdown
    assert FROZEN_RERANKER_CONFIG_HASH[:8] in markdown


def test_cli_writes_outputs(pool_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = PROJECT_ROOT / "data" / "05_evaluation" / "_pytest_hybrid_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / "hybrid.json"
    output_md = PROJECT_ROOT / "tests" / "_pytest_hybrid_tmp.md"

    def fake_from_paths(**kwargs):
        evaluator, _ = _build_evaluator(pool_artifact, tmp_path, case_limit=2)
        return evaluator

    class _Closer:
        def close(self) -> None:
            return None

    monkeypatch.setattr(hybrid_cli, "create_vector_store", lambda **kwargs: _Closer())
    monkeypatch.setattr(hybrid_cli, "create_embedding_provider", lambda **kwargs: object())
    monkeypatch.setattr(hybrid_cli.HybridAbEvaluator, "from_paths", staticmethod(fake_from_paths))
    monkeypatch.setattr(
        hybrid_cli,
        "validate_index_dir",
        lambda index_dir, project_root: tmp_path,
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.hybrid_evaluator.load_manifest",
        lambda index_dir: _fake_manifest(),
    )

    exit_code, _run = hybrid_cli.run_hybrid_ab(
        config_path=EXPERIMENT_CONFIG,
        reranker_config_path=RERANKER_CONFIG,
        questions_path=QUESTIONS,
        expected_path=EXPECTED,
        index_dir=tmp_path,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        output_json=output_json,
        output_markdown=output_md,
        project_root_path=PROJECT_ROOT,
    )
    assert exit_code == 0
    assert output_json.exists()
    assert output_md.exists()
    restored = HybridEvaluationRun.model_validate(json.loads(output_json.read_text(encoding="utf-8")))
    assert restored.experiment.experiment_id == "hybrid-lexical-vector-v1"
    output_json.unlink(missing_ok=True)
    output_md.unlink(missing_ok=True)
