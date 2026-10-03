"""CLI for production release posture validation and diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from customer_claims_rag import env_bootstrap
from customer_claims_rag.release.posture import (
    DEFAULT_DESCRIPTOR_PATH,
    resolve_production_release_posture,
)
from customer_claims_rag.release.readiness import RELEASE_YES, assess_release_readiness
from customer_claims_rag.exceptions import ReleasePostureError, RetrievalError
from customer_claims_rag.retrieval.factory import create_vector_store

DEFAULT_DESCRIPTOR_RELATIVE = Path("configs") / "release" / "production_posture.json"

# The command is a release gate: automation may treat exit status 0 as "release may proceed" and
# nothing else. Static validation passing is a different fact from release readiness, so a run that
# could not establish readiness exits non-zero with its own code (2 stays argparse's usage error).
EXIT_RELEASE_READY = 0
EXIT_RELEASE_BLOCKED = 1
EXIT_READINESS_NOT_ESTABLISHED = 3

EXIT_STATUS_HELP = (
    "exit status:\n"
    f"  {EXIT_RELEASE_READY}  release can proceed: the index content was verified against the "
    "canonical corpus (release_can_proceed=yes)\n"
    f"  {EXIT_RELEASE_BLOCKED}  release cannot proceed: missing, invalid or stale index, source "
    "mismatch, or an invalid descriptor (release_can_proceed=no)\n"
    f"  {EXIT_READINESS_NOT_ESTABLISHED}  readiness not established: the static checks passed but the "
    "index content was not verified, as with --skip-vector-store "
    "(release_can_proceed=not_established)\n"
    "  2  command-line usage error"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the committed production release posture: the canonical corpus, the "
            "vector index and whether the index matches the corpus. Exit status 0 only when "
            "release readiness is established, that is, when the index content was verified "
            "against the canonical corpus."
        ),
        epilog=EXIT_STATUS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help=(
            "Static check only: validate the manifests and the frozen config without opening "
            "Chroma. Does not establish that the index content matches the corpus, so the "
            f"command exits {EXIT_READINESS_NOT_ESTABLISHED} (release_can_proceed=not_established) "
            "even when every static check passes: it is never a successful release gate"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_bootstrap.load_project_env()

    try:
        context = resolve_production_release_posture(
            descriptor_path=args.descriptor,
            project_root=env_bootstrap.project_root(),
            target_name=args.target.strip() if args.target is not None else None,
        )
        readiness = assess_release_readiness(
            context,
            check_index_content=not args.skip_vector_store,
            vector_store_factory=create_vector_store,
        )
    except (ReleasePostureError, RetrievalError) as exc:
        print(f"release posture validation failed: {exc}", file=sys.stderr)
        return EXIT_RELEASE_BLOCKED

    if readiness.diagnostics is None:
        for line in readiness.format_lines():
            print(line)
        print(f"release posture validation failed: {readiness.problem}", file=sys.stderr)
        if readiness.next_step:
            print(f"next step: {readiness.next_step}", file=sys.stderr)
        return EXIT_RELEASE_BLOCKED

    for line in readiness.diagnostics.format_lines():
        print(line)
    for line in readiness.format_status_lines():
        print(line)
    if readiness.release_status == RELEASE_YES:
        return EXIT_RELEASE_READY

    print(
        "release readiness not established: static validation passed, but the index content "
        "was not verified against the canonical corpus. Run validate-release-posture without "
        "--skip-vector-store to establish it.",
        file=sys.stderr,
    )
    return EXIT_READINESS_NOT_ESTABLISHED


if __name__ == "__main__":
    raise SystemExit(main())
