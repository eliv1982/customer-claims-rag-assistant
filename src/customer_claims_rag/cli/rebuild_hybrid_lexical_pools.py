"""CLI for one-time exact lexical pool reconstruction (stage 2C.3)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.evaluation.hybrid_lexical_replay import (
    LexicalReplayInvariantError,
    run_exact_lexical_pool_reconstruction,
)
from customer_claims_rag.evaluation.hybrid_reporting import (
    DEFAULT_HYBRID_JSON,
    DEFAULT_HYBRID_MARKDOWN,
    write_hybrid_outputs,
)
from customer_claims_rag.exceptions import EvaluationOutputError
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval_config import RetrievalSettings

DEFAULT_INDEX_DIR = Path("data/04_index")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Replay exact lexical top-24 pools into a frozen hybrid artifact.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_HYBRID_JSON,
        help=f"Frozen hybrid JSON artifact (default: {DEFAULT_HYBRID_JSON})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.index_dir,
        help=f"Chroma index directory (default: {settings.index_dir})",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: from artifact shared_context)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_HYBRID_JSON,
        help=f"Output JSON path (default: {DEFAULT_HYBRID_JSON})",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_HYBRID_MARKDOWN,
        help=f"Output Markdown path (default: {DEFAULT_HYBRID_MARKDOWN})",
    )
    parser.add_argument(
        "--snapshot-before",
        type=Path,
        default=None,
        help="Optional path for immutability snapshot before replay",
    )
    return parser


def run_rebuild_lexical_pools(
    *,
    artifact_path: Path,
    index_dir: Path,
    collection_name: str | None,
    output_json: Path,
    output_markdown: Path,
    project_root_path: Path,
    snapshot_before: Path | None = None,
) -> int:
    root = project_root_path.resolve()
    resolved_artifact = artifact_path if artifact_path.is_absolute() else root / artifact_path
    resolved_index = validate_index_dir(index_dir, project_root=root)
    try:
        updated, before, after = run_exact_lexical_pool_reconstruction(
            artifact_path=resolved_artifact,
            index_dir=resolved_index,
            collection_name=collection_name,
        )
    except (LexicalReplayInvariantError, EvaluationOutputError) as exc:
        print(f"Lexical replay failed: {exc}", file=sys.stderr)
        return 1

    if snapshot_before is None:
        snapshot_before = Path(os.environ.get("TEMP", ".")) / "hybrid_lexical_replay_before.json"
    snapshot_before.write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    after_path = snapshot_before.with_name(snapshot_before.stem + "_after.json")
    after_path.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_hybrid_outputs(
        updated,
        output_json=output_json,
        output_markdown=output_markdown,
        project_root=root,
    )
    print(f"Exact lexical pools written to {output_json}")
    print(f"Immutability snapshots: {snapshot_before} / {after_path}")
    return 0


def main() -> int:
    load_project_env()
    args = build_parser().parse_args()
    root = project_root()
    return run_rebuild_lexical_pools(
        artifact_path=args.artifact,
        index_dir=args.index_dir,
        collection_name=args.collection,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        project_root_path=root,
        snapshot_before=args.snapshot_before,
    )


if __name__ == "__main__":
    raise SystemExit(main())
