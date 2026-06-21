"""Evaluate baseline retrieval against the 60-case corpus."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.evaluator import RetrievalEvaluator
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.path_helpers import (
    COMMITTED_IMPROVEMENT_LOG,
    COMMITTED_RESULTS_REPORT,
    DEFAULT_EVALUATION_OUTPUT,
)
from customer_claims_rag.evaluation.reporting import write_evaluation_outputs
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


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Evaluate baseline retrieval on the 60-case FoodFlow corpus.",
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
        "--top-k",
        type=int,
        default=12,
        help="Maximum ranked results retained for evaluation metrics (default: 12)",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=12,
        help="Candidate pool fetched from vector store (default: 12)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity threshold for baseline run (default: 0.0)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT,
        help=f"Generated JSON output path (default: {DEFAULT_EVALUATION_OUTPUT})",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=COMMITTED_RESULTS_REPORT,
        help=f"Human-readable results report (fixed path: {COMMITTED_RESULTS_REPORT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sanitized traceback on error",
    )
    return parser


def run_evaluation(
    *,
    questions_path: Path,
    expected_path: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    top_k: int = 12,
    fetch_k: int = 12,
    threshold: float = 0.0,
    output_json: Path = DEFAULT_EVALUATION_OUTPUT,
    output_markdown: Path = COMMITTED_RESULTS_REPORT,
    project_root_path: Path | None = None,
    settings: RetrievalSettings | None = None,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
) -> tuple[int, object | None]:
    resolved_settings = settings or RetrievalSettings.from_env()
    root = (project_root_path or project_root()).resolve()
    vector_store = None
    run = None
    try:
        resolved_index_dir = validate_index_dir(index_dir, project_root=root)
        resolved_questions = _resolve_project_path(questions_path, root)
        resolved_expected = _resolve_project_path(expected_path, root)

        cases = load_evaluation_corpus(
            questions_path=resolved_questions,
            expected_path=resolved_expected,
        )

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
        evaluator = RetrievalEvaluator(
            retriever=retriever,
            index_dir=resolved_index_dir,
            cases=cases,
            top_k=top_k,
            fetch_k=fetch_k,
            threshold=threshold,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=root,
        )
        run = evaluator.evaluate()
        json_path, _, _ = write_evaluation_outputs(
            run,
            output_json=output_json,
            output_results=output_markdown,
            output_improvement_log=COMMITTED_IMPROVEMENT_LOG,
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

    aggregate = run.aggregate_metrics
    print("Baseline retrieval evaluation complete")
    print(f"Evaluation result ID: {run.run_metadata.evaluation_result_id}")
    print(f"Cases: {aggregate.total_cases}; technical errors: {aggregate.technical_error_count}")
    print(f"Hit@4: {aggregate.hit_rate_at_4:.3f}; Primary hit@4: {aggregate.primary_source_hit_rate_at_4:.3f}")
    print(f"MRR: {aggregate.mrr:.3f}; No-result rate: {aggregate.no_result_rate:.3f}")
    print(f"JSON: {json_path.as_posix()}")
    print(f"Results report: {COMMITTED_RESULTS_REPORT.as_posix()}")
    print(f"Improvement log: {COMMITTED_IMPROVEMENT_LOG.as_posix()}")
    print(f"Fingerprint: {run.run_metadata.index_fingerprint}")
    return 0, run


def _resolve_project_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_markdown.resolve() != (project_root() / COMMITTED_RESULTS_REPORT).resolve():
        print(
            f"Error: --output-markdown must be {COMMITTED_RESULTS_REPORT.as_posix()}",
            file=sys.stderr,
        )
        return 2

    try:
        exit_code, _ = run_evaluation(
            questions_path=args.questions,
            expected_path=args.expected,
            index_dir=args.index_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            threshold=args.threshold,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            settings=RetrievalSettings.from_env(),
            project_root_path=project_root(),
        )
        return exit_code
    except Exception:
        print("Error: evaluation failed", file=sys.stderr)
        if args.verbose:
            try:
                raise RetrievalError("evaluation failed") from None
            except RetrievalError:
                traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
