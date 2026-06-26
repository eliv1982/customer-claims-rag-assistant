"""CLI for production release posture validation and diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag import env_bootstrap
from customer_claims_rag.release.posture import (
    DEFAULT_DESCRIPTOR_PATH,
    load_production_release_context,
    load_release_posture_descriptor,
    resolve_release_target,
    validate_release_posture_for_production,
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
    root = env_bootstrap.project_root()
    vector_store = None

    try:
        descriptor = load_release_posture_descriptor(args.descriptor)
        if args.target is not None:
            target_name = args.target.strip()
            resolved_target = resolve_release_target(
                descriptor,
                target_name,
                project_root=root,
            )
        else:
            _descriptor, resolved_target = load_production_release_context(
                descriptor_path=args.descriptor,
                project_root=root,
            )

        vector_store = None
        if not args.skip_vector_store:
            vector_store = create_vector_store(
                index_dir=resolved_target.index_dir,
                collection_name=resolved_target.collection_name,
                open_existing=True,
            )

        diagnostics = validate_release_posture_for_production(
            descriptor,
            resolved_target,
            project_root=root,
            vector_store=vector_store,
        )
    except (ReleasePostureError, RetrievalError) as exc:
        print(f"release posture validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if vector_store is not None:
            vector_store.close()

    for line in diagnostics.format_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
