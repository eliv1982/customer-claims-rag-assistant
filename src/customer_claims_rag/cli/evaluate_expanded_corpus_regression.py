"""CLI for paired expanded-corpus frozen benchmark regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.expanded_corpus_regression import (
    build_expanded_corpus_regression,
    compute_evaluation_result_id,
)
from customer_claims_rag.evaluation.expanded_corpus_regression_reporting import (
    DEFAULT_REGRESSION_JSON,
    DEFAULT_REGRESSION_MARKDOWN,
    write_expanded_corpus_regression_outputs,
)
from customer_claims_rag.evaluation.pool_expansion_evaluator import PoolExpansionEvaluator
from customer_claims_rag.evaluation.pool_expansion_reporting import DEFAULT_POOL_EXPANSION_JSON
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
DEFAULT_EXPERIMENT_CONFIG = Path("configs/retrieval/vector_pool_expansion_v1.json")
DEFAULT_RERANKER_CONFIG = Path("configs/reranking/source_authority_v1.json")
DEFAULT_ARM_A_INDEX = Path("data/04_index_backup_10docs_215chunks")
DEFAULT_ARM_B_INDEX = Path("data/04_index")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description=(
            "Run paired frozen-benchmark regression for historical vs expanded corpus indexes."
        ),
    )
    parser.add_argument("--arm-a-index-dir", type=Path, default=DEFAULT_ARM_A_INDEX)
    parser.add_argument("--arm-b-index-dir", type=Path, default=DEFAULT_ARM_B_INDEX)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--reranker-config", type=Path, default=DEFAULT_RERANKER_CONFIG)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REGRESSION_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_REGRESSION_MARKDOWN)
    return parser


def run_expanded_corpus_regression(
    *,
    arm_a_index_dir: Path,
    arm_b_index_dir: Path,
    config_path: Path,
    reranker_config_path: Path,
    questions_path: Path,
    expected_path: Path,
    collection_name: str,
    embedding_model: str,
    output_json: Path = DEFAULT_REGRESSION_JSON,
    output_markdown: Path = DEFAULT_REGRESSION_MARKDOWN,
    project_root_path: Path | None = None,
    settings: RetrievalSettings | None = None,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
) -> tuple[int, object | None]:
    resolved_settings = settings or RetrievalSettings.from_env()
    root = (project_root_path or project_root()).resolve()
    if output_json.resolve() == (root / DEFAULT_POOL_EXPANSION_JSON).resolve():
        raise EvaluationOutputError(
            f"refusing to overwrite protected artifact: {DEFAULT_POOL_EXPANSION_JSON}",
        )

    arm_a_run = None
    arm_b_run = None
    try:
        arm_a_run = _evaluate_arm(
            index_dir=arm_a_index_dir,
            config_path=config_path,
            reranker_config_path=reranker_config_path,
            questions_path=questions_path,
            expected_path=expected_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            root=root,
            settings=resolved_settings,
            embedding_provider_factory=embedding_provider_factory,
            vector_store_factory=vector_store_factory,
        )
        arm_b_run = _evaluate_arm(
            index_dir=arm_b_index_dir,
            config_path=config_path,
            reranker_config_path=reranker_config_path,
            questions_path=questions_path,
            expected_path=expected_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            root=root,
            settings=resolved_settings,
            embedding_provider_factory=embedding_provider_factory,
            vector_store_factory=vector_store_factory,
        )
        regression = build_expanded_corpus_regression(
            arm_a_run=arm_a_run,
            arm_b_run=arm_b_run,
            arm_a_path=str(_resolve_project_path(arm_a_index_dir, root)),
            arm_b_path=str(_resolve_project_path(arm_b_index_dir, root)),
        )
        result_id = compute_evaluation_result_id(regression)
        json_path, markdown_path = write_expanded_corpus_regression_outputs(
            regression,
            output_json=output_json,
            output_markdown=output_markdown,
            project_root=root,
        )
    except EvaluationCorpusError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2, None
    except (IndexManifestError, EmbeddingError, EvaluationOutputError, RetrievalError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None

    print("Expanded corpus frozen regression complete")
    print(f"Evaluation ID: {regression.evaluation_id}")
    print(f"Result hash: {result_id}")
    print(f"Verdict: {regression.verdict}")
    print(
        f"Primary hit@4: Arm A {regression.arm_a.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"-> Arm B {regression.arm_b.aggregate_metrics.primary_source_hit_rate_at_4:.3f} "
        f"(delta {regression.metric_deltas.primary_source_hit_rate_at_4:+.3f})"
    )
    print(f"Harmful regressions: {regression.harmful_regressions or 'none'}")
    print(f"JSON: {json_path.as_posix()}")
    print(f"Report: {markdown_path.as_posix()}")
    return 0, regression


def _evaluate_arm(
    *,
    index_dir: Path,
    config_path: Path,
    reranker_config_path: Path,
    questions_path: Path,
    expected_path: Path,
    collection_name: str,
    embedding_model: str,
    root: Path,
    settings: RetrievalSettings,
    embedding_provider_factory,
    vector_store_factory,
):
    resolved_index_dir = validate_index_dir(index_dir, project_root=root)
    if not resolved_index_dir.is_dir():
        raise RetrievalError(f"index directory not found: {resolved_index_dir}")

    vector_store = vector_store_factory(
        index_dir=resolved_index_dir,
        collection_name=collection_name,
    )
    try:
        embedding_provider = embedding_provider_factory(
            model_name=embedding_model,
            api_key=settings.openai_api_key,
        )
        retriever = BaselineRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            top_k=24,
            fetch_k=24,
            similarity_threshold=0.0,
        )
        evaluator = PoolExpansionEvaluator.from_paths(
            retriever=retriever,
            experiment_config_path=_resolve_project_path(config_path, root),
            reranker_config_path=_resolve_project_path(reranker_config_path, root),
            index_dir=resolved_index_dir,
            questions_path=_resolve_project_path(questions_path, root),
            expected_path=_resolve_project_path(expected_path, root),
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=root,
        )
        return evaluator.evaluate()
    finally:
        vector_store.close()


def _resolve_project_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.arm_a_index_dir.exists():
        print(f"Error: Arm A index not found: {args.arm_a_index_dir}", file=sys.stderr)
        return 1
    if not args.arm_b_index_dir.exists():
        print(f"Error: Arm B index not found: {args.arm_b_index_dir}", file=sys.stderr)
        return 1
    exit_code, _ = run_expanded_corpus_regression(
        arm_a_index_dir=args.arm_a_index_dir,
        arm_b_index_dir=args.arm_b_index_dir,
        config_path=args.config,
        reranker_config_path=args.reranker_config,
        questions_path=args.questions,
        expected_path=args.expected,
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
