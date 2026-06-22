"""CLI for hybrid lexical + vector A/B evaluation (stage 2C.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.hybrid_evaluator import HybridAbEvaluator
from customer_claims_rag.evaluation.hybrid_reporting import (
    DEFAULT_HYBRID_JSON,
    DEFAULT_HYBRID_MARKDOWN,
    write_hybrid_outputs,
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
DEFAULT_EXPERIMENT_CONFIG = Path("configs/retrieval/hybrid_lexical_vector_v1.json")
DEFAULT_RERANKER_CONFIG = Path("configs/reranking/source_authority_v1.json")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Run hybrid lexical + vector A/B evaluation.",
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
        default=DEFAULT_HYBRID_JSON,
        help=f"JSON output path (default: {DEFAULT_HYBRID_JSON})",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_HYBRID_MARKDOWN,
        help=f"Markdown report path (default: {DEFAULT_HYBRID_MARKDOWN})",
    )
    return parser


def run_hybrid_ab(
    *,
    config_path: Path,
    reranker_config_path: Path,
    questions_path: Path,
    expected_path: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    output_json: Path = DEFAULT_HYBRID_JSON,
    output_markdown: Path = DEFAULT_HYBRID_MARKDOWN,
    project_root_path: Path | None = None,
    settings: RetrievalSettings | None = None,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
) -> tuple[int, object | None]:
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
        retriever = BaselineRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            top_k=24,
            fetch_k=24,
            similarity_threshold=0.0,
        )
        evaluator = HybridAbEvaluator.from_paths(
            retriever=retriever,
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
        json_path, markdown_path = write_hybrid_outputs(
            run,
            output_json=output_json,
            output_markdown=output_markdown,
            project_root=root,
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

    acceptance = run.acceptance
    reach = run.candidate_generation_comparison
    ranking = run.ranking_comparison
    print("Hybrid lexical + vector evaluation complete")
    print(f"Experiment: {run.experiment.experiment_id} ({run.experiment.version})")
    print(f"Config hash: {run.experiment.config_hash}")
    print(
        f"Primary reachability: vector@24 {reach.baseline_primary_reachable}/{reach.baseline_primary_total} "
        f"-> fusion@24 {reach.candidate_primary_reachable}/{reach.candidate_primary_total}"
    )
    print(f"Candidate-generation verdict: {acceptance.reachability_verdict}")
    print(f"Final-ranking verdict: {acceptance.ranking_verdict}")
    print(
        f"Primary hit@4: baseline={run.baseline.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"candidate={run.candidate.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"delta={ranking.aggregate.primary_source_hit_rate_at_4_delta:+.3f}"
    )
    print(f"JSON: {json_path.as_posix()}")
    print(f"Report: {markdown_path.as_posix()}")
    return 0, run


def _resolve_project_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _ = run_hybrid_ab(
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
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
