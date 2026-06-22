"""Tests for exact lexical pool reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.evaluation.hybrid_lexical_replay import (
    LexicalReplayInvariantError,
    apply_exact_lexical_pools_to_run,
    build_frozen_lexical_retriever,
    replay_exact_lexical_pools,
    run_exact_lexical_pool_reconstruction,
    validate_lexical_replay_invariants,
)
from customer_claims_rag.evaluation.hybrid_metrics import (
    FROZEN_HYBRID_CONFIG_HASH,
    extract_immutability_snapshot,
    lexical_pool_snapshot_from_hits,
    rebuild_hybrid_diagnostics,
)
from customer_claims_rag.evaluation.hybrid_models import HybridEvaluationRun
from customer_claims_rag.evaluation.hybrid_reporting import rebuild_report_from_artifact
from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.lexical.preprocessor import TOKENIZER_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "hybrid_lexical_vector_v1.json"
INDEX_DIR = PROJECT_ROOT / "data" / "04_index"


@pytest.fixture
def frozen_run() -> HybridEvaluationRun:
    if not ARTIFACT.exists():
        pytest.skip("frozen hybrid artifact not present")
    return HybridEvaluationRun.model_validate(
        json.loads(ARTIFACT.read_text(encoding="utf-8"))
    )


def _chunk(chunk_id: str) -> LexicalChunk:
    document_id = chunk_id.split("::", 1)[0]
    return LexicalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        heading="Heading",
        chunk_type="policy",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        metadata={"heading_path": "[\"Heading\"]"},
    )


def test_validate_rejects_config_hash_mismatch(frozen_run: HybridEvaluationRun) -> None:
    bad = frozen_run.model_copy(
        update={
            "experiment": frozen_run.experiment.model_copy(
                update={"config_hash": "deadbeef"}
            )
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="config hash"):
        validate_lexical_replay_invariants(bad)


def test_validate_rejects_tokenizer_mismatch(frozen_run: HybridEvaluationRun) -> None:
    bad = frozen_run.model_copy(
        update={
            "lexical_index": frozen_run.lexical_index.model_copy(
                update={"tokenizer_version": "tokenizer-v2"}
            )
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="tokenizer"):
        validate_lexical_replay_invariants(bad)


def test_validate_rejects_bm25_mismatch(frozen_run: HybridEvaluationRun) -> None:
    bad = frozen_run.model_copy(
        update={
            "lexical_index": frozen_run.lexical_index.model_copy(update={"bm25_k1": 2.0})
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="BM25"):
        validate_lexical_replay_invariants(bad)


def test_lexical_top24_length_and_order_deterministic() -> None:
    from customer_claims_rag.retrieval.lexical.bm25 import BM25Index
    from customer_claims_rag.retrieval.lexical.retriever import LexicalRetriever

    chunks = [_chunk("alpha::chunk-001"), _chunk("beta::chunk-002")]
    retriever = LexicalRetriever(BM25Index(chunks, k1=1.5, b=0.75))
    first = retriever.search("alpha content", k=24)
    second = retriever.search("alpha content", k=24)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert all(item.lexical_rank == index + 1 for index, item in enumerate(first))


def test_exact_replay_uses_no_vector_retriever_or_openai(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("vector retrieval or OpenAI invoked")

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.retriever.BaselineRetriever.search",
        _boom,
    )
    monkeypatch.setattr(
        "customer_claims_rag.retrieval.adapters.openai_embeddings.OpenAIEmbeddingProvider.embed",
        _boom,
        raising=False,
    )
    hits = [
        LexicalHit(
            chunk_id="doc::chunk-001",
            document_id="doc",
            lexical_rank=1,
            bm25_score=1.0,
            matched_tokens=["term"],
        )
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = hits
    with patch(
        "customer_claims_rag.evaluation.hybrid_lexical_replay.build_frozen_lexical_retriever",
        return_value=mock_retriever,
    ):
        pools = replay_exact_lexical_pools(frozen_run, mock_retriever)
    assert len(pools) == len(frozen_run.case_results)
    assert mock_retriever.search.call_count == len(frozen_run.case_results)


def test_corpus_fingerprint_mismatch_rejects_replay(frozen_run: HybridEvaluationRun) -> None:
    with patch(
        "customer_claims_rag.evaluation.hybrid_lexical_replay.load_lexical_corpus_from_chroma",
        side_effect=__import__(
            "customer_claims_rag.exceptions", fromlist=["IndexManifestError"]
        ).IndexManifestError("corpus fingerprint mismatch"),
    ):
        with pytest.raises(__import__(
            "customer_claims_rag.exceptions", fromlist=["IndexManifestError"]
        ).IndexManifestError):
            build_frozen_lexical_retriever(
                index_dir=INDEX_DIR,
                collection_name="customer_claims",
                expected_corpus_fingerprint=frozen_run.lexical_index.corpus_fingerprint,
                bm25_k1=1.5,
                bm25_b=0.75,
                expected_lexical_index_fingerprint=frozen_run.lexical_index.lexical_index_fingerprint,
            )


@pytest.mark.skipif(not INDEX_DIR.exists(), reason="local Chroma index required")
def test_run_exact_lexical_pool_reconstruction_live(frozen_run: HybridEvaluationRun) -> None:
    if frozen_run.experiment.config_hash != FROZEN_HYBRID_CONFIG_HASH:
        pytest.skip("artifact config hash mismatch")
    updated, before, after = run_exact_lexical_pool_reconstruction(
        artifact_path=ARTIFACT,
        index_dir=INDEX_DIR,
    )
    assert before == after
    assert all(case.lexical_pool is not None and case.lexical_pool.exact for case in updated.case_results)
    assert all(len(case.lexical_pool.candidates) <= 24 for case in updated.case_results)


def test_json_contains_exact_lexical_pools(frozen_run: HybridEvaluationRun) -> None:
    if not _artifact_has_exact(frozen_run):
        pytest.skip("exact lexical pools not yet reconstructed")
    for case in frozen_run.case_results:
        assert case.lexical_pool is not None
        assert case.lexical_pool.exact
        assert len(case.lexical_pool.candidates) == len(case.lexical_pool.candidate_ids)
        assert case.lexical_pool.candidates[0].lexical_rank == 1


def test_diagnostics_rebuild_from_json_no_bm25(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _artifact_has_exact(frozen_run):
        pytest.skip("exact lexical pools not yet reconstructed")

    def _boom(*args, **kwargs):
        raise AssertionError("BM25 invoked during diagnostics rebuild")

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.lexical.bm25.BM25Index.search",
        _boom,
    )
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    again = rebuild_hybrid_diagnostics(rebuilt)
    assert again.model_dump() == rebuilt.model_dump()


def test_immutable_ranking_snapshot_unchanged_after_exact_apply(
    frozen_run: HybridEvaluationRun,
) -> None:
    if not _artifact_has_exact(frozen_run):
        pytest.skip("exact lexical pools not yet reconstructed")
    before = extract_immutability_snapshot(frozen_run)
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    after = extract_immutability_snapshot(rebuilt)
    assert before == after


def test_report_rebuild_from_json_no_bm25(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _artifact_has_exact(frozen_run):
        pytest.skip("exact lexical pools not yet reconstructed")

    def _boom(*args, **kwargs):
        raise AssertionError("BM25 invoked during report rebuild")

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.lexical.bm25.BM25Index.search",
        _boom,
    )
    out_dir = PROJECT_ROOT / "tests" / "_pytest_hybrid_report_tmp.md"
    try:
        rebuild_report_from_artifact(
            artifact_path=ARTIFACT,
            output_markdown=out_dir,
            project_root=PROJECT_ROOT,
        )
        text = out_dir.read_text(encoding="utf-8")
        assert "Exact lexical top-24 pools were reconstructed once" in text
    finally:
        out_dir.unlink(missing_ok=True)


def _artifact_has_exact(run: HybridEvaluationRun) -> bool:
    return all(
        case.lexical_pool is not None
        and case.lexical_pool.exact
        and case.lexical_pool.candidates
        for case in run.case_results
    )


def test_apply_exact_lexical_pools_updates_channel_fields_only(
    frozen_run: HybridEvaluationRun,
) -> None:
    hits = [
        LexicalHit(
            chunk_id="07_complaint_handling_procedure::chunk-019",
            document_id="07_complaint_handling_procedure",
            lexical_rank=1,
            bm25_score=2.5,
            matched_tokens=["cvv"],
        )
    ]
    pools = {case.case_id: hits for case in frozen_run.case_results}
    before = extract_immutability_snapshot(frozen_run)
    updated = apply_exact_lexical_pools_to_run(frozen_run, pools)
    after = extract_immutability_snapshot(updated)
    assert before == after
    t004 = next(case for case in updated.case_results if case.case_id == "T004")
    assert t004.lexical_pool is not None
    assert t004.lexical_pool.exact
    assert t004.lexical_pool.candidates[0].bm25_score == 2.5


def test_lexical_pool_snapshot_from_hits() -> None:
    hits = [
        LexicalHit("a::1", "a", 1, 1.5, ["x"]),
        LexicalHit("b::2", "b", 2, 1.0, ["y"]),
    ]
    pool = lexical_pool_snapshot_from_hits(hits, pool_k=24, exact=True)
    assert pool.exact is True
    assert pool.candidate_ids == ["a::1", "b::2"]
    assert pool.candidates[0].lexical_rank == 1
