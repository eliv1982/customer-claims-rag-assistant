"""CLI to rebuild reranking A/B derived fields from an existing artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.ab_rebuild import (
    load_ab_artifact,
    load_frozen_baseline,
    rebuild_ab_artifact,
)
from customer_claims_rag.evaluation.ab_reporting import (
    DEFAULT_AB_JSON,
    DEFAULT_AB_MARKDOWN,
    write_ab_outputs,
)
from customer_claims_rag.exceptions import EvaluationOutputError


DEFAULT_FROZEN_BASELINE = Path("data/05_evaluation/retrieval_results.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild reranking A/B diagnostics and report from existing JSON.",
    )
    parser.add_argument(
        "--from-artifact",
        type=Path,
        default=DEFAULT_AB_JSON,
        help=f"Existing A/B JSON artifact (default: {DEFAULT_AB_JSON})",
    )
    parser.add_argument(
        "--frozen-baseline",
        type=Path,
        default=DEFAULT_FROZEN_BASELINE,
        help=f"Frozen baseline JSON (default: {DEFAULT_FROZEN_BASELINE})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: overwrite --from-artifact)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_AB_MARKDOWN,
        help=f"Markdown report path (default: {DEFAULT_AB_MARKDOWN})",
    )
    return parser


def run_rebuild(
    *,
    artifact_path: Path,
    frozen_baseline_path: Path,
    output_json: Path | None = None,
    output_markdown: Path = DEFAULT_AB_MARKDOWN,
    project_root_path: Path | None = None,
) -> tuple[int, object | None]:
    root = (project_root_path or project_root()).resolve()
    resolved_artifact = _resolve_project_path(artifact_path, root)
    resolved_frozen = _resolve_project_path(frozen_baseline_path, root)
    resolved_output_json = _resolve_project_path(output_json or artifact_path, root)
    resolved_output_markdown = _resolve_project_path(output_markdown, root)

    try:
        run = load_ab_artifact(resolved_artifact)
        frozen = load_frozen_baseline(resolved_frozen)
        rebuilt = rebuild_ab_artifact(run, frozen)
        json_path, markdown_path = write_ab_outputs(
            rebuilt,
            output_json=resolved_output_json,
            output_markdown=resolved_output_markdown,
            project_root=root,
        )
    except (EvaluationOutputError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None

    acceptance = rebuilt.comparison.acceptance
    identity = rebuilt.baseline_identity
    print("Reranking A/B artifact rebuild complete")
    print(f"Identity status: {identity.identity_status}")
    print(
        f"Identity counts: exact={identity.exact_count} "
        f"float_drift={identity.same_order_float_drift_count} "
        f"order_diff={identity.same_set_different_order_count} "
        f"mismatch={identity.substantive_mismatch_count}"
    )
    print(f"Baseline identity pass: {acceptance.baseline_identity_pass}")
    print(f"Verdict: {'candidate accepted' if acceptance.candidate_accepted else 'candidate rejected'}")
    print(f"JSON: {json_path.as_posix()}")
    print(f"Report: {markdown_path.as_posix()}")
    return 0, rebuilt


def _resolve_project_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _ = run_rebuild(
        artifact_path=args.from_artifact,
        frozen_baseline_path=args.frozen_baseline,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        project_root_path=project_root(),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
