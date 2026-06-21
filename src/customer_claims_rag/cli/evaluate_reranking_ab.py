"""CLI for reranking A/B evaluation (stage 2C.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.ab_evaluator import RerankingAbEvaluator
from customer_claims_rag.evaluation.ab_reporting import (
    DEFAULT_AB_JSON,
    DEFAULT_AB_MARKDOWN,
    write_ab_outputs,
)
from customer_claims_rag.cli.rebuild_reranking_ab_report import run_rebuild
from customer_claims_rag.exceptions import (
    EmbeddingError,
    EvaluationCorpusError,
    EvaluationOutputError,
    IndexManifestError,
    RetrievalError,
)
from customer_claims_rag.evaluation.path_helpers import DEFAULT_EVALUATION_OUTPUT
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings

DEFAULT_QUESTIONS = Path("tests/01_test_questions.md")
DEFAULT_EXPECTED = Path("tests/02_expected_answers.md")
DEFAULT_CONFIG = Path("configs/reranking/source_authority_v1.json")
DEFAULT_FROZEN_BASELINE = Path("data/05_evaluation/retrieval_results.json")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Run production-like reranking A/B evaluation (source-authority-v1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Reranker config JSON (default: {DEFAULT_CONFIG})",
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
        "--frozen-baseline",
        type=Path,
        default=DEFAULT_FROZEN_BASELINE,
        help=f"Frozen baseline JSON for identity check (default: {DEFAULT_FROZEN_BASELINE})",
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
        "--top-k",
        type=int,
        default=12,
        help="Maximum ranked results (default: 12)",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=12,
        help="Candidate pool size (default: 12)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity threshold (default: 0.0)",
    )
    parser.add_argument(
        "--from-artifact",
        type=Path,
        default=None,
        help="Rebuild derived fields from existing A/B JSON instead of live retrieval",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_AB_JSON,
        help=f"A/B JSON output path (default: {DEFAULT_AB_JSON})",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_AB_MARKDOWN,
        help=f"A/B Markdown report path (default: {DEFAULT_AB_MARKDOWN})",
    )
    return parser


def run_reranking_ab(
    *,
    config_path: Path,
    questions_path: Path,
    expected_path: Path,
    frozen_baseline_path: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    top_k: int = 12,
    fetch_k: int = 12,
    threshold: float = 0.0,
    output_json: Path = DEFAULT_AB_JSON,
    output_markdown: Path = DEFAULT_AB_MARKDOWN,
    from_artifact: Path | None = None,
    project_root_path: Path | None = None,
    settings: RetrievalSettings | None = None,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
) -> tuple[int, object | None]:
    root = (project_root_path or project_root()).resolve()
    if from_artifact is not None:
        return run_rebuild(
            artifact_path=from_artifact,
            frozen_baseline_path=frozen_baseline_path,
            output_json=output_json,
            output_markdown=output_markdown,
            project_root_path=root,
        )

    resolved_settings = settings or RetrievalSettings.from_env()
    vector_store = None
    try:
        if output_json.resolve() == (root / DEFAULT_EVALUATION_OUTPUT).resolve():
            print(
                f"Error: --output-json must not overwrite baseline artifact {DEFAULT_EVALUATION_OUTPUT.as_posix()}",
                file=sys.stderr,
            )
            return 2, None

        resolved_index_dir = validate_index_dir(index_dir, project_root=root)
        resolved_questions = _resolve_project_path(questions_path, root)
        resolved_expected = _resolve_project_path(expected_path, root)
        resolved_config = _resolve_project_path(config_path, root)
        resolved_frozen = _resolve_project_path(frozen_baseline_path, root)

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
            top_k=top_k,
            fetch_k=fetch_k,
            similarity_threshold=threshold,
        )
        evaluator = RerankingAbEvaluator.from_paths(
            retriever=retriever,
            config_path=resolved_config,
            index_dir=resolved_index_dir,
            questions_path=resolved_questions,
            expected_path=resolved_expected,
            top_k=top_k,
            fetch_k=fetch_k,
            threshold=threshold,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=root,
            frozen_baseline_path=resolved_frozen,
        )
        run = evaluator.evaluate()
        json_path, markdown_path = write_ab_outputs(
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

    acceptance = run.comparison.acceptance
    print("Reranking A/B evaluation complete")
    print(f"Reranker: {run.experiment.reranker_id} ({run.experiment.version})")
    print(f"Config hash: {run.experiment.config_hash}")
    print(f"Baseline identity pass: {acceptance.baseline_identity_pass}")
    print(f"Shared pool pass: {acceptance.shared_pool_pass}")
    print(
        f"Primary hit@4: baseline={run.baseline.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"candidate={run.candidate.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"delta={run.comparison.aggregate.primary_source_hit_rate_at_4_delta:+.3f}"
    )
    print(f"Verdict: {'candidate accepted' if acceptance.candidate_accepted else 'candidate rejected'}")
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
    exit_code, _ = run_reranking_ab(
        config_path=args.config,
        questions_path=args.questions,
        expected_path=args.expected,
        frozen_baseline_path=args.frozen_baseline,
        index_dir=args.index_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        threshold=args.threshold,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        from_artifact=args.from_artifact,
        project_root_path=project_root(),
        settings=RetrievalSettings.from_env(),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
