"""CLI for vector pool cap A/B evaluation (Stage 4C.3B)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.diversity_evaluator import VectorPoolCapAbEvaluator
from customer_claims_rag.evaluation.diversity_metrics import STAGE_4C2_BASELINE_TARGETS
from customer_claims_rag.evaluation.diversity_models import DiversityEvaluationRun
from customer_claims_rag.evaluation.diversity_reporting import (
    DEFAULT_DIVERSITY_JSON,
    DEFAULT_DIVERSITY_MARKDOWN,
    write_diversity_outputs,
)
from customer_claims_rag.exceptions import (
    EmbeddingError,
    EvaluationCorpusError,
    EvaluationOutputError,
    IndexManifestError,
    RetrievalError,
)
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings

DEFAULT_QUESTIONS = Path("tests/01_test_questions.md")
DEFAULT_EXPECTED = Path("tests/02_expected_answers.md")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/retrieval/vector_pool_36_cap4_v1.json")
DEFAULT_RERANKER_CONFIG = Path("configs/reranking/source_authority_v1.json")

BASELINE_REPRODUCTION_TOLERANCE = 0.002


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Run vector pool cap A/B evaluation (fetch@24 vs fetch@48 cap@4 pool@36).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help=f"Experiment config JSON (default: {DEFAULT_EXPERIMENT_CONFIG})",
    )
    parser.add_argument(
        "--reranker-config",
        type=Path,
        default=DEFAULT_RERANKER_CONFIG,
        help=f"Frozen reranker config JSON (default: {DEFAULT_RERANKER_CONFIG})",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"Evaluation questions Markdown (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help=f"Expected answers Markdown (default: {DEFAULT_EXPECTED})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.index_dir,
        help=f"Vector index directory (default: {settings.index_dir})",
    )
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"Chroma collection name (default: {settings.collection_name})",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help=f"OpenAI embedding model (default: {settings.embedding_model})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_DIVERSITY_JSON,
        help=f"JSON output path (default: {DEFAULT_DIVERSITY_JSON})",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_DIVERSITY_MARKDOWN,
        help=f"Markdown report path (default: {DEFAULT_DIVERSITY_MARKDOWN})",
    )
    parser.add_argument(
        "--skip-baseline-check",
        action="store_true",
        help="Skip Stage 4C.2 baseline reproduction validation.",
    )
    return parser


def run_vector_pool_cap_ab(
    *,
    config_path: Path,
    reranker_config_path: Path,
    questions_path: Path,
    expected_path: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    output_json: Path = DEFAULT_DIVERSITY_JSON,
    output_markdown: Path = DEFAULT_DIVERSITY_MARKDOWN,
    project_root_path: Path | None = None,
    settings: RetrievalSettings | None = None,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
    skip_baseline_check: bool = False,
) -> tuple[int, DiversityEvaluationRun | None]:
    resolved_settings = settings or RetrievalSettings.from_env()
    root = (project_root_path or project_root()).resolve()
    vector_store = None
    try:
        resolved_index_dir = validate_index_dir(index_dir, project_root=root)
        resolved_questions = _resolve_project_path(questions_path, root)
        resolved_expected = _resolve_project_path(expected_path, root)
        resolved_config = _resolve_project_path(config_path, root)
        resolved_reranker_config = _resolve_project_path(reranker_config_path, root)

        vector_store = vector_store_factory(
            index_dir=resolved_index_dir,
            collection_name=collection_name,
        )
        embedding_provider = embedding_provider_factory(
            model_name=embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        baseline_retriever = BaselineRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            top_k=24,
            fetch_k=24,
            similarity_threshold=0.0,
        )
        candidate_retriever = BaselineRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            top_k=48,
            fetch_k=48,
            similarity_threshold=0.0,
        )
        evaluator = VectorPoolCapAbEvaluator.from_paths(
            baseline_retriever=baseline_retriever,
            candidate_retriever=candidate_retriever,
            experiment_config_path=resolved_config,
            reranker_config_path=resolved_reranker_config,
            index_dir=resolved_index_dir,
            questions_path=resolved_questions,
            expected_path=resolved_expected,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=root,
        )
        run = evaluator.evaluate()

        if not skip_baseline_check and not _baseline_reproduces_stage_4c2(run):
            print(
                "Error: baseline arm does not reproduce Stage 4C.2 metrics; "
                "aborting before writing artifacts.",
                file=sys.stderr,
            )
            _print_baseline_drift(run)
            return 1, run

        json_path, markdown_path = write_diversity_outputs(
            run,
            output_json=output_json,
            output_markdown=output_markdown,
            project_root=root,
        )
        DiversityEvaluationRun.model_validate(
            json.loads(json_path.read_text(encoding="utf-8")),
        )
    except EvaluationCorpusError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2, None
    except (IndexManifestError, EmbeddingError, EvaluationOutputError, RetrievalError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    finally:
        if vector_store is not None:
            vector_store.close()

    reach = run.reachability_comparison
    baseline = run.baseline_ranking.aggregate_metrics
    candidate = run.candidate_ranking.aggregate_metrics
    print("Vector pool cap A/B evaluation complete")
    print(f"Experiment: {run.experiment.experiment_id} ({run.experiment.version})")
    print(f"Config hash: {run.experiment.config_hash}")
    print(f"Verdict: {run.verdict}")
    print(
        f"Primary reachability: baseline {reach.baseline_primary_reachable}/"
        f"{reach.baseline_primary_total} -> candidate "
        f"{reach.candidate_primary_reachable}/{reach.candidate_primary_total}"
    )
    print(
        f"Primary hit@4: baseline={baseline.primary_source_hit_rate_at_4:.3f} "
        f"candidate={candidate.primary_source_hit_rate_at_4:.3f}"
    )
    print(f"JSON: {json_path.as_posix()}")
    print(f"Report: {markdown_path.as_posix()}")
    return 0, run


def _baseline_reproduces_stage_4c2(run: DiversityEvaluationRun) -> bool:
    baseline = run.baseline_ranking.aggregate_metrics
    reach = run.reachability_comparison
    faq = run.faq_comparison
    tol = BASELINE_REPRODUCTION_TOLERANCE
    targets = STAGE_4C2_BASELINE_TARGETS
    checks = [
        abs(baseline.primary_source_hit_rate_at_4 - targets["primary_hit_at_4"]) <= tol,
        abs(baseline.hit_rate_at_12 - targets["primary_hit_at_12"]) <= tol,
        abs(baseline.mrr - targets["mrr"]) <= tol,
        reach.baseline_primary_reachable == targets["primary_reach"][0],
        reach.high_baseline_primary_reachable == targets["high_risk_reach"][0],
        reach.critical_baseline_primary_reachable == targets["critical_reach"][0],
        faq.baseline_questions_with_faq_in_top4 == targets["faq_top4_count"],
    ]
    return all(checks)


def _print_baseline_drift(run: DiversityEvaluationRun) -> None:
    baseline = run.baseline_ranking.aggregate_metrics
    reach = run.reachability_comparison
    faq = run.faq_comparison
    targets = STAGE_4C2_BASELINE_TARGETS
    print(
        f"  primary hit@4: expected {targets['primary_hit_at_4']}, "
        f"got {baseline.primary_source_hit_rate_at_4}",
        file=sys.stderr,
    )
    print(
        f"  primary hit@12: expected {targets['primary_hit_at_12']}, "
        f"got {baseline.hit_rate_at_12}",
        file=sys.stderr,
    )
    print(f"  mrr: expected {targets['mrr']}, got {baseline.mrr}", file=sys.stderr)
    print(
        f"  primary reach: expected {targets['primary_reach']}, "
        f"got ({reach.baseline_primary_reachable}, {reach.baseline_primary_total})",
        file=sys.stderr,
    )
    print(
        f"  faq top-4: expected {targets['faq_top4_count']}, "
        f"got {faq.baseline_questions_with_faq_in_top4}",
        file=sys.stderr,
    )


def _resolve_project_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _ = run_vector_pool_cap_ab(
        config_path=args.config,
        reranker_config_path=args.reranker_config,
        questions_path=args.questions,
        expected_path=args.expected,
        index_dir=args.index_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        project_root_path=project_root(),
        settings=RetrievalSettings.from_env(),
        skip_baseline_check=args.skip_baseline_check,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
