"""Tests for exact lexical pool reconstruction."""

from __future__ import annotations

import json
import shutil
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
from customer_claims_rag.exceptions import IndexManifestError
from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.lexical.preprocessor import TOKENIZER_VERSION
from customer_claims_rag.retrieval.manifest import load_manifest
from tests.local_artifacts import requires_local_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "hybrid_lexical_vector_v1.json"
POSTURE = PROJECT_ROOT / "configs" / "release" / "production_posture.json"
# The frozen hybrid artifact was measured on the 10-document release index, which the
# release posture names as its active target (data/04_index holds the 15-document archive).
INDEX_DIR = PROJECT_ROOT / json.loads(POSTURE.read_text(encoding="utf-8"))["targets"]["active"]["index_path"]
EXACT_POOLS_MISSING = "tracked hybrid artifact lacks exact lexical pools"


@pytest.fixture
def frozen_run() -> HybridEvaluationRun:
    # Tracked artifact: a missing or unreadable file is a failure, not a skip.
    return HybridEvaluationRun.model_validate(
        json.loads(ARTIFACT.read_text(encoding="utf-8"))
    )


def assert_local_index_matches_frozen_artifact(run: HybridEvaluationRun, manifest) -> None:
    """A local index that exists but does not match the frozen artifact is stale state: fail."""
    assert run.experiment.config_hash == FROZEN_HYBRID_CONFIG_HASH, (
        f"tracked hybrid artifact config hash {run.experiment.config_hash} does not match "
        f"FROZEN_HYBRID_CONFIG_HASH {FROZEN_HYBRID_CONFIG_HASH}"
    )
    assert manifest.corpus_fingerprint == run.lexical_index.corpus_fingerprint, (
        f"stale local index: manifest corpus fingerprint {manifest.corpus_fingerprint} differs "
        f"from the frozen hybrid artifact's {run.lexical_index.corpus_fingerprint}; "
        "rebuild or restore the release index"
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


def test_validate_accepts_frozen_artifact_metadata(frozen_run: HybridEvaluationRun) -> None:
    validate_lexical_replay_invariants(frozen_run)
    assert frozen_run.shared_context.chunk_count == frozen_run.lexical_index.chunk_count
    assert frozen_run.shared_context.chunk_count == 215
    assert frozen_run.shared_context.document_count == 10


def test_validate_rejects_zero_shared_chunk_count(frozen_run: HybridEvaluationRun) -> None:
    bad = frozen_run.model_copy(
        update={
            "shared_context": frozen_run.shared_context.model_copy(update={"chunk_count": 0})
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="shared_context.chunk_count"):
        validate_lexical_replay_invariants(bad)


def test_validate_rejects_zero_lexical_chunk_count(frozen_run: HybridEvaluationRun) -> None:
    bad = frozen_run.model_copy(
        update={
            "lexical_index": frozen_run.lexical_index.model_copy(update={"chunk_count": 0})
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="lexical_index.chunk_count"):
        validate_lexical_replay_invariants(bad)


def test_validate_rejects_mismatched_artifact_chunk_counts(
    frozen_run: HybridEvaluationRun,
) -> None:
    bad = frozen_run.model_copy(
        update={
            "lexical_index": frozen_run.lexical_index.model_copy(update={"chunk_count": 216})
        }
    )
    with pytest.raises(LexicalReplayInvariantError, match="artifact chunk count mismatch"):
        validate_lexical_replay_invariants(bad)


def test_build_rejects_loaded_replay_count_mismatch(frozen_run: HybridEvaluationRun) -> None:
    short_corpus = [_chunk("doc::chunk-001")]
    with patch(
        "customer_claims_rag.evaluation.hybrid_lexical_replay.load_lexical_corpus_from_chroma",
        return_value=short_corpus,
    ):
        with pytest.raises(IndexManifestError, match="artifact replay corpus count mismatch"):
            build_frozen_lexical_retriever(
                index_dir=INDEX_DIR,
                collection_name="customer_claims",
                expected_corpus_fingerprint=frozen_run.lexical_index.corpus_fingerprint,
                expected_chunk_count=frozen_run.lexical_index.chunk_count,
                bm25_k1=1.5,
                bm25_b=0.75,
                expected_lexical_index_fingerprint=frozen_run.lexical_index.lexical_index_fingerprint,
                expected_document_count=frozen_run.shared_context.document_count,
            )


def test_build_rejects_loaded_document_count_mismatch(frozen_run: HybridEvaluationRun) -> None:
    corpus = [_chunk(f"doc-{index}::chunk-001") for index in range(frozen_run.lexical_index.chunk_count)]
    with patch(
        "customer_claims_rag.evaluation.hybrid_lexical_replay.load_lexical_corpus_from_chroma",
        return_value=corpus,
    ):
        with pytest.raises(IndexManifestError, match="artifact replay document count mismatch"):
            build_frozen_lexical_retriever(
                index_dir=INDEX_DIR,
                collection_name="customer_claims",
                expected_corpus_fingerprint=frozen_run.lexical_index.corpus_fingerprint,
                expected_chunk_count=frozen_run.lexical_index.chunk_count,
                bm25_k1=1.5,
                bm25_b=0.75,
                expected_lexical_index_fingerprint=frozen_run.lexical_index.lexical_index_fingerprint,
                expected_document_count=frozen_run.shared_context.document_count,
            )


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
                expected_chunk_count=frozen_run.lexical_index.chunk_count,
                bm25_k1=1.5,
                bm25_b=0.75,
                expected_lexical_index_fingerprint=frozen_run.lexical_index.lexical_index_fingerprint,
                expected_document_count=frozen_run.shared_context.document_count,
            )


@requires_local_artifacts(
    INDEX_DIR / "manifest.json",
    why="local Chroma index (built with live OpenAI embeddings) matching the frozen hybrid artifact",
)
def test_run_exact_lexical_pool_reconstruction_live(frozen_run: HybridEvaluationRun) -> None:
    assert_local_index_matches_frozen_artifact(frozen_run, load_manifest(INDEX_DIR))
    updated, before, after = run_exact_lexical_pool_reconstruction(
        artifact_path=ARTIFACT,
        index_dir=INDEX_DIR,
    )
    assert before == after
    assert all(case.lexical_pool is not None and case.lexical_pool.exact for case in updated.case_results)
    assert all(len(case.lexical_pool.candidates) <= 24 for case in updated.case_results)


def test_stale_local_index_fails_instead_of_skipping(frozen_run: HybridEvaluationRun) -> None:
    class Manifest:
        corpus_fingerprint = "0" * 64

    with pytest.raises(AssertionError, match="stale local index"):
        assert_local_index_matches_frozen_artifact(frozen_run, Manifest())


def test_stale_artifact_config_hash_fails_instead_of_skipping(
    frozen_run: HybridEvaluationRun,
) -> None:
    class Manifest:
        corpus_fingerprint = frozen_run.lexical_index.corpus_fingerprint

    stale = frozen_run.model_copy(
        update={"experiment": frozen_run.experiment.model_copy(update={"config_hash": "deadbeef"})}
    )
    with pytest.raises(AssertionError, match="config hash"):
        assert_local_index_matches_frozen_artifact(stale, Manifest())


def test_matching_local_index_is_accepted(frozen_run: HybridEvaluationRun) -> None:
    class Manifest:
        corpus_fingerprint = frozen_run.lexical_index.corpus_fingerprint

    assert_local_index_matches_frozen_artifact(frozen_run, Manifest())


def test_json_contains_exact_lexical_pools(frozen_run: HybridEvaluationRun) -> None:
    assert _artifact_has_exact(frozen_run), EXACT_POOLS_MISSING
    for case in frozen_run.case_results:
        assert case.lexical_pool is not None
        assert case.lexical_pool.exact
        assert len(case.lexical_pool.candidates) == len(case.lexical_pool.candidate_ids)
        assert case.lexical_pool.candidates[0].lexical_rank == 1


def test_diagnostics_rebuild_from_json_no_bm25(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _artifact_has_exact(frozen_run), EXACT_POOLS_MISSING

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
    assert _artifact_has_exact(frozen_run), EXACT_POOLS_MISSING
    before = extract_immutability_snapshot(frozen_run)
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    after = extract_immutability_snapshot(rebuilt)
    assert before == after


def test_report_rebuild_from_json_no_bm25(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert _artifact_has_exact(frozen_run), EXACT_POOLS_MISSING

    def _boom(*args, **kwargs):
        raise AssertionError("BM25 invoked during report rebuild")

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.lexical.bm25.BM25Index.search",
        _boom,
    )
    artifact = tmp_path / ARTIFACT.relative_to(PROJECT_ROOT)
    artifact.parent.mkdir(parents=True)
    shutil.copy2(ARTIFACT, artifact)
    output_markdown = tmp_path / "hybrid_report.md"
    rebuild_report_from_artifact(
        artifact_path=artifact,
        output_markdown=output_markdown,
        project_root=tmp_path,
    )
    text = output_markdown.read_text(encoding="utf-8")
    assert "Exact lexical top-24 pools were reconstructed once" in text


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
