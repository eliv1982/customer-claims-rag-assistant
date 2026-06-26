"""CLI for doc12 threat atomic corpus A/B evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INPUT_DIR
from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.doc08_atomic_contract import DirtySourceTreeError
from customer_claims_rag.evaluation.doc12_threat_atomic_contract import (
    BaselineReproductionError,
    DEFAULT_REFERENCE_ARTIFACT,
    load_holdout_corpus,
    repo_relative_path,
)
from customer_claims_rag.evaluation.doc12_replay_integrity import build_replay_stability_result
from customer_claims_rag.evaluation.doc12_threat_atomic_evaluator import (
    Doc12ThreatAtomicAbEvaluator,
    prepare_doc12_chunk_diff_and_fingerprints,
)
from customer_claims_rag.evaluation.doc12_threat_atomic_reporting import (
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    write_doc12_outputs,
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

DEFAULT_CONFIG = Path("configs/experiments/doc12_threat_atomic_units_v1.json")


def compute_holdout_dataset_fingerprint(
    *,
    questions_path: Path,
    expected_path: Path,
    benchmark_id: str,
) -> str:
    payload = {
        "benchmark_id": benchmark_id,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(description="Run doc12 threat atomic corpus A/B evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-index-dir", type=Path, default=None)
    parser.add_argument("--candidate-index-dir", type=Path, default=None)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument(
        "--stability-matrix",
        action="store_true",
        help="Run replay stability matrix before final evaluation artifact",
    )
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

    doc08_overlay = root / config["baseline_corpus"]["doc08_overlay_path"]
    doc12_overlay = root / config["candidate_overlay"]["overlay_path"]
    expected_doc08_fp = config["baseline_corpus"].get("expected_doc08_fingerprint")

    diff, baseline_doc08_fp, candidate_doc08_fp, doc08_unchanged = (
        prepare_doc12_chunk_diff_and_fingerprints(
            canonical_dir=args.canonical_dir if args.canonical_dir.is_absolute() else root / args.canonical_dir,
            doc08_overlay_path=doc08_overlay,
            doc12_overlay_path=doc12_overlay,
            permitted_root=root,
            embedding_model=args.embedding_model,
            expected_doc08_fingerprint=expected_doc08_fp,
        )
    )

    frozen_q = root / config["frozen_benchmark"]["questions"]
    frozen_e = root / config["frozen_benchmark"]["expected"]
    ext_q = root / config["extension_benchmark"]["questions"]
    ext_e = root / config["extension_benchmark"]["expected"]
    holdout_q = root / config["holdout_benchmark"]["questions"]
    holdout_e = root / config["holdout_benchmark"]["expected"]

    frozen_cases = load_evaluation_corpus(questions_path=frozen_q, expected_path=frozen_e)
    extension_cases = load_extension_corpus(
        questions_path=ext_q,
        expected_path=ext_e,
        expected_ids=config["extension_benchmark"]["case_ids"],
    )
    holdout_cases = load_holdout_corpus(
        questions_path=holdout_q,
        expected_path=holdout_e,
        expected_ids=config["holdout_benchmark"]["case_ids"],
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
        top_k=fetch_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
    )
    candidate_retriever = BaselineRetriever(
        embedding_provider=emb,
        vector_store=candidate_vs,
        index_dir=candidate_index,
        top_k=fetch_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
    )

    reranker_cfg_path = root / retrieval["reranker_config"]
    reranker = SourceAuthorityV1Reranker(load_reranker_config(reranker_cfg_path))
    pool_cfg = load_vector_pool_cap_config(root / "configs/retrieval/vector_pool_36_cap4_v1.json")

    reference_artifact = root / config.get("reference_artifact", DEFAULT_REFERENCE_ARTIFACT)

    evaluator = Doc12ThreatAtomicAbEvaluator(
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
        holdout_cases=holdout_cases,
        baseline_index_dir=baseline_index,
        candidate_index_dir=candidate_index,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        experiment_id=config["experiment_id"],
        config=config,
        chunk_diff=diff,
        baseline_doc08_fingerprint=baseline_doc08_fp,
        candidate_doc08_fingerprint=candidate_doc08_fp,
        doc08_fingerprint_unchanged=doc08_unchanged,
        frozen_benchmark_fingerprint=compute_evaluation_dataset_fingerprint(
            questions_path=frozen_q,
            expected_path=frozen_e,
        ),
        extension_benchmark_fingerprint=compute_extension_dataset_fingerprint(
            questions_path=ext_q,
            expected_path=ext_e,
            benchmark_id=config["extension_benchmark"]["benchmark_id"],
        ),
        holdout_benchmark_fingerprint=compute_holdout_dataset_fingerprint(
            questions_path=holdout_q,
            expected_path=holdout_e,
            benchmark_id=config["holdout_benchmark"]["benchmark_id"],
        ),
        production_retrieval_config_hash=compute_vector_pool_cap_config_hash(pool_cfg),
        reference_artifact_path=reference_artifact,
        project_root=root,
    )

    replay_stability = None
    if args.stability_matrix:
        canonical_dir = args.canonical_dir if args.canonical_dir.is_absolute() else root / args.canonical_dir
        replay_stability = build_replay_stability_result(
            index_dir=candidate_index,
            project_root=root,
            config=config,
            canonical_dir=canonical_dir,
            evaluator=evaluator,
            rebuild_parent=root / ".tmp" / "doc12_replay_matrix",
        )
        evaluator.replay_stability = replay_stability

    try:
        run = evaluator.evaluate()
        json_path = args.output_json if args.output_json.is_absolute() else root / args.output_json
        markdown_path = (
            args.output_markdown if args.output_markdown.is_absolute() else root / args.output_markdown
        )
        write_doc12_outputs(run, json_path=json_path, markdown_path=markdown_path)
        print(f"verdict={run.verdict}")
        print(f"reference_artifact={repo_relative_path(reference_artifact, root)}")
        print(
            "primary_hit@4="
            f"{run.frozen_candidate_metrics.primary_source_hit_rate_at_4:.3f}"
        )
        print(f"doc08_unchanged={run.doc08_fingerprint_unchanged}")
        print(f"holdout_positive_hit4={run.holdout.candidate.positive_doc12_hit_at_4}/6")
        if run.replay_integrity_verdict:
            print(f"replay_integrity={run.replay_integrity_verdict}")
        return 0 if run.verdict == "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR" else 2
    except (BaselineReproductionError, DirtySourceTreeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    finally:
        baseline_vs.close()
        candidate_vs.close()


if __name__ == "__main__":
    raise SystemExit(main())
