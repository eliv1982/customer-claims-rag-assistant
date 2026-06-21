"""Path safety checks for evaluation outputs."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.exceptions import EvaluationCorpusError, IngestionError, RetrievalError
from customer_claims_rag.retrieval.path_helpers import (
    RAW_MARKDOWN_DIRNAME,
    _resolve_under_root,
)
from customer_claims_rag.ingestion.path_helpers import CLEAN_MARKDOWN_DIRNAME, _is_under_or_equal

DEFAULT_EVALUATION_OUTPUT = Path("data/05_evaluation/retrieval_results.json")
COMMITTED_RESULTS_REPORT = Path("tests/03_test_results.md")
COMMITTED_IMPROVEMENT_LOG = Path("tests/04_improvement_log.md")
PROTECTED_SOURCE_PATHS = (
    Path("tests/01_test_questions.md"),
    Path("tests/02_expected_answers.md"),
)


def validate_evaluation_output_path(output_path: Path, *, project_root: Path) -> Path:
    """Validate generated evaluation JSON path before writing."""
    root = project_root.resolve()
    try:
        resolved = _resolve_under_root(output_path, root)
    except IngestionError as exc:
        raise RetrievalError(str(exc)) from None

    raw_dir = (root / RAW_MARKDOWN_DIRNAME).resolve()
    clean_dir = (root / CLEAN_MARKDOWN_DIRNAME).resolve()
    if _is_under_or_equal(resolved, raw_dir):
        raise RetrievalError("evaluation output must not be inside raw source directory")
    if _is_under_or_equal(resolved, clean_dir):
        raise RetrievalError("evaluation output must not be inside clean Markdown directory")

    for protected in PROTECTED_SOURCE_PATHS:
        if resolved == (root / protected).resolve():
            raise RetrievalError(
                f"evaluation output must not overwrite protected source file: {protected.as_posix()}"
            )

    index_dir = (root / "data/04_index").resolve()
    if _is_under_or_equal(resolved, index_dir):
        raise RetrievalError("evaluation output must not be inside index directory")

    return resolved


def validate_committed_report_path(path: Path, *, project_root: Path, expected: Path) -> Path:
    """Ensure committed report writes only go to fixed expected paths."""
    root = project_root.resolve()
    resolved = _resolve_under_root(path, root)
    expected_resolved = (root / expected).resolve()
    if resolved != expected_resolved:
        raise EvaluationCorpusError(
            f"report path must be {expected.as_posix()}, got {path.as_posix()}"
        )
    return resolved
