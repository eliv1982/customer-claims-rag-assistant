"""CLI for production release posture validation and diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag import env_bootstrap
from customer_claims_rag.release.posture import (
    DEFAULT_DESCRIPTOR_PATH,
    resolve_production_release_posture,
    validate_production_release_posture,
)
from customer_claims_rag.exceptions import ReleasePostureError, RetrievalError
from customer_claims_rag.retrieval.factory import create_vector_store

DEFAULT_DESCRIPTOR_RELATIVE = Path("configs") / "release" / "production_posture.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the committed production release posture and emit diagnostics.",
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=DEFAULT_DESCRIPTOR_PATH,
        help=f"Release posture descriptor JSON (default: {DEFAULT_DESCRIPTOR_RELATIVE})",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Named release target to validate (default: descriptor default or RAG_RELEASE_TARGET)",
    )
    parser.add_argument(
        "--skip-vector-store",
        action="store_true",
        help="Validate manifest and frozen config only; do not open Chroma",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_bootstrap.load_project_env()
    vector_store = None
    diagnostics = None

    try:
        context = resolve_production_release_posture(
            descriptor_path=args.descriptor,
            project_root=env_bootstrap.project_root(),
            target_name=args.target.strip() if args.target is not None else None,
        )
        if not args.skip_vector_store:
            vector_store = create_vector_store(
                index_dir=context.resolved_target.index_dir,
                collection_name=context.resolved_target.collection_name,
                open_existing=True,
            )
        diagnostics = validate_production_release_posture(
            context,
            vector_store=vector_store,
        )
    except (ReleasePostureError, RetrievalError) as exc:
        print(f"release posture validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if vector_store is not None:
            vector_store.close()

    if diagnostics is None:
        return 1
    for line in diagnostics.format_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
