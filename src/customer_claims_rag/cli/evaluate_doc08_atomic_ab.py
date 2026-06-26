"""CLI for doc08 atomic corpus A/B evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INPUT_DIR
from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.doc08_atomic_evaluator import (
    Doc08AtomicAbEvaluator,
    prepare_chunk_diff_and_fingerprints,
)
from customer_claims_rag.evaluation.doc08_atomic_reporting import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    write_doc08_outputs,
)
from customer_claims_rag.evaluation.diversity_metrics import (
    compute_vector_pool_cap_config_hash,
    load_vector_pool_cap_config,
)
from customer_claims_rag.evaluation.extension_parser import (
    compute_extension_dataset_fingerprint,
    load_extension_corpus,
)
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import compute_evaluation_dataset_fingerprint
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, load_reranker_config
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings

DEFAULT_CONFIG = Path("configs/experiments/doc08_atomic_risk_units_v1.json")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(description="Run doc08 atomic corpus A/B evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-index-dir", type=Path, default=None)
    parser.add_argument("--candidate-index-dir", type=Path, default=None)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    root = project_root()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = RetrievalSettings.from_env()

    retrieval = config["retrieval"]
    fetch_k = int(retrieval["fetch_k"])
    pool_k = int(retrieval["candidate_pool_k"])
    cap = int(retrieval["per_document_cap"])
    final_top_k = int(retrieval["final_top_k"])
    threshold = float(retrieval["threshold"])

    baseline_index = validate_index_dir(
        args.baseline_index_dir or root / config["baseline_index_dir"],
        project_root=root,
    )
    candidate_index = validate_index_dir(
        args.candidate_index_dir or root / config["candidate_index_dir"],
        project_root=root,
    )
    overlay_path = root / config["overlay_path"]

    diff, baseline_doc08_fp, candidate_doc08_fp = prepare_chunk_diff_and_fingerprints(
        canonical_dir=args.canonical_dir if args.canonical_dir.is_absolute() else root / args.canonical_dir,
        overlay_path=overlay_path,
        permitted_root=root,
        embedding_model=args.embedding_model,
    )

    frozen_q = root / config["frozen_benchmark"]["questions"]
    frozen_e = root / config["frozen_benchmark"]["expected"]
    ext_q = root / config["extension_benchmark"]["questions"]
    ext_e = root / config["extension_benchmark"]["expected"]

    frozen_cases = load_evaluation_corpus(questions_path=frozen_q, expected_path=frozen_e)
    extension_cases = load_extension_corpus(
        questions_path=ext_q,
        expected_path=ext_e,
        expected_ids=config["extension_benchmark"]["case_ids"],
    )

    emb = create_embedding_provider(
        model_name=args.embedding_model,
        api_key=settings.openai_api_key,
    )
    baseline_vs = create_vector_store(index_dir=baseline_index, collection_name=args.collection)
    candidate_vs = create_vector_store(index_dir=candidate_index, collection_name=args.collection)

    baseline_retriever = BaselineRetriever(
        embedding_provider=emb,
        vector_store=baseline_vs,
        index_dir=baseline_index,
        top_k=final_top_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
    )
    candidate_retriever = BaselineRetriever(
        embedding_provider=emb,
        vector_store=candidate_vs,
        index_dir=candidate_index,
        top_k=final_top_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
    )

    reranker_cfg_path = root / retrieval["reranker_config"]
    reranker = SourceAuthorityV1Reranker(load_reranker_config(reranker_cfg_path))
    pool_cfg = load_vector_pool_cap_config(root / "configs/retrieval/vector_pool_36_cap4_v1.json")

    evaluator = Doc08AtomicAbEvaluator(
        baseline_retriever=baseline_retriever,
        candidate_retriever=candidate_retriever,
        reranker=reranker,
        fetch_k=fetch_k,
        pool_k=pool_k,
        per_document_cap=cap,
        final_top_k=final_top_k,
        threshold=threshold,
        frozen_cases=frozen_cases,
        extension_cases=extension_cases,
        baseline_index_dir=baseline_index,
        candidate_index_dir=candidate_index,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        experiment_id=config["experiment_id"],
        config=config,
        chunk_diff=diff,
        baseline_doc08_fingerprint=baseline_doc08_fp,
        candidate_doc08_fingerprint=candidate_doc08_fp,
        frozen_benchmark_fingerprint=compute_evaluation_dataset_fingerprint(
            questions_path=frozen_q,
            expected_path=frozen_e,
        ),
        extension_benchmark_fingerprint=compute_extension_dataset_fingerprint(
            questions_path=ext_q,
            expected_path=ext_e,
            benchmark_id=config["extension_benchmark"]["benchmark_id"],
        ),
        production_retrieval_config_hash=compute_vector_pool_cap_config_hash(pool_cfg),
        project_root=root,
    )

    try:
        run = evaluator.evaluate()
        write_doc08_outputs(
            run,
            json_path=args.output_json if args.output_json.is_absolute() else root / args.output_json,
            markdown_path=args.output_markdown
            if args.output_markdown.is_absolute()
            else root / args.output_markdown,
        )
        print(f"verdict={run.verdict}")
        print(f"primary_hit@4={run.frozen_candidate_metrics.primary_source_hit_rate_at_4:.3f}")
        print(f"T044_reachable={next((d.candidate_primary_reachable for d in run.required_diagnostics if d.case_id=='T044'), None)}")
        return 0 if run.verdict == "ACCEPTED AS TARGETED CORPUS REPAIR" else 2
    finally:
        baseline_vs.close()
        candidate_vs.close()


if __name__ == "__main__":
    raise SystemExit(main())
