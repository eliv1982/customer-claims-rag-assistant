"""One-time exact lexical top-24 reconstruction for frozen hybrid artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.evaluation.hybrid_metrics import (
    FROZEN_HYBRID_CONFIG_HASH,
    build_channel_reachability_comparison,
    extract_immutability_snapshot,
    lexical_pool_snapshot_from_hits,
    recompute_case_channel_diagnostics,
)
from customer_claims_rag.evaluation.hybrid_models import HybridEvaluationRun
from customer_claims_rag.exceptions import EvaluationOutputError, IndexManifestError
from customer_claims_rag.retrieval.lexical.bm25 import BM25Index, LexicalHit
from customer_claims_rag.retrieval.lexical.corpus_loader import (
    EXPECTED_CHUNK_COUNT,
    load_lexical_corpus_from_chroma,
)
from customer_claims_rag.retrieval.lexical.preprocessor import TOKENIZER_VERSION
from customer_claims_rag.retrieval.lexical.retriever import LexicalRetriever


class LexicalReplayInvariantError(EvaluationOutputError):
    """Raised when frozen lexical replay invariants do not hold."""


def validate_lexical_replay_invariants(run: HybridEvaluationRun) -> None:
    """Validate artifact metadata before exact lexical replay."""
    if run.experiment.config_hash != FROZEN_HYBRID_CONFIG_HASH:
        raise LexicalReplayInvariantError(
            f"hybrid config hash mismatch: {run.experiment.config_hash}"
        )
    lexical = run.lexical_index
    if lexical.chunk_count != EXPECTED_CHUNK_COUNT:
        raise LexicalReplayInvariantError(
            f"expected chunk_count={EXPECTED_CHUNK_COUNT}, got {lexical.chunk_count}"
        )
    if lexical.tokenizer_version != TOKENIZER_VERSION:
        raise LexicalReplayInvariantError(
            f"tokenizer version mismatch: {lexical.tokenizer_version}"
        )
    if lexical.bm25_k1 != 1.5 or lexical.bm25_b != 0.75:
        raise LexicalReplayInvariantError("BM25 parameters mismatch")
    if lexical.lexical_algorithm != "BM25":
        raise LexicalReplayInvariantError("lexical algorithm mismatch")
    if run.shared_context.lexical_k != 24:
        raise LexicalReplayInvariantError("lexical_k must be 24")
    if not run.case_results:
        raise LexicalReplayInvariantError("artifact has no case_results")


def build_frozen_lexical_retriever(
    *,
    index_dir: Path,
    collection_name: str,
    expected_corpus_fingerprint: str,
    bm25_k1: float,
    bm25_b: float,
    expected_lexical_index_fingerprint: str,
) -> LexicalRetriever:
    """Build BM25 index from Chroma corpus without embeddings or vector search."""
    chunks = load_lexical_corpus_from_chroma(
        index_dir=index_dir,
        collection_name=collection_name,
        expected_fingerprint=expected_corpus_fingerprint,
    )
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise IndexManifestError(f"expected {EXPECTED_CHUNK_COUNT} chunks, loaded {len(chunks)}")
    index = BM25Index(chunks, k1=bm25_k1, b=bm25_b)
    fingerprint = index.compute_fingerprint(corpus_fingerprint=expected_corpus_fingerprint)
    if fingerprint != expected_lexical_index_fingerprint:
        raise LexicalReplayInvariantError(
            "lexical index fingerprint mismatch after BM25 rebuild"
        )
    return LexicalRetriever(index)


def replay_exact_lexical_pools(
    run: HybridEvaluationRun,
    retriever: LexicalRetriever,
) -> dict[str, object]:
    """Replay lexical top-24 for every case query using frozen BM25 settings."""
    validate_lexical_replay_invariants(run)
    lexical_k = run.shared_context.lexical_k
    pools: dict[str, list[LexicalHit]] = {}
    for case in run.case_results:
        hits = retriever.search(case.query, k=lexical_k)
        if len(hits) > lexical_k:
            raise LexicalReplayInvariantError(
                f"{case.case_id}: lexical replay returned more than k={lexical_k} hits"
            )
        pools[case.case_id] = hits
    return pools


def apply_exact_lexical_pools_to_run(
    run: HybridEvaluationRun,
    pools: dict[str, list[LexicalHit]],
) -> HybridEvaluationRun:
    """Attach exact lexical pools and recompute channel diagnostics only."""
    lexical_k = run.shared_context.lexical_k
    updated_cases = []
    for case in run.case_results:
        hits = pools[case.case_id]
        lexical_pool = lexical_pool_snapshot_from_hits(hits, pool_k=lexical_k, exact=True)
        updated_case = case.model_copy(update={"lexical_pool": lexical_pool})
        updated_cases.append(recompute_case_channel_diagnostics(updated_case))

    reachability = build_channel_reachability_comparison(updated_cases)
    return run.model_copy(
        update={
            "case_results": updated_cases,
            "candidate_generation_comparison": reachability,
        }
    )


def run_exact_lexical_pool_reconstruction(
    *,
    artifact_path: Path,
    index_dir: Path,
    collection_name: str | None = None,
) -> tuple[HybridEvaluationRun, dict[str, object], dict[str, object]]:
    """Replay exact lexical pools and return updated run plus immutability snapshots."""
    run = HybridEvaluationRun.model_validate(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    validate_lexical_replay_invariants(run)
    before = extract_immutability_snapshot(run)

    resolved_collection = collection_name or run.shared_context.collection
    retriever = build_frozen_lexical_retriever(
        index_dir=index_dir,
        collection_name=resolved_collection,
        expected_corpus_fingerprint=run.lexical_index.corpus_fingerprint,
        bm25_k1=run.lexical_index.bm25_k1,
        bm25_b=run.lexical_index.bm25_b,
        expected_lexical_index_fingerprint=run.lexical_index.lexical_index_fingerprint,
    )
    pools = replay_exact_lexical_pools(run, retriever)
    updated = apply_exact_lexical_pools_to_run(run, pools)
    after = extract_immutability_snapshot(updated)
    if before != after:
        raise LexicalReplayInvariantError(
            "immutable experimental snapshot changed during lexical replay"
        )
    return updated, before, after
