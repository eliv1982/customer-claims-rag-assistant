"""Build chunk corpus from clean Markdown documents."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from customer_claims_rag.config import (
    DEFAULT_ENCODING,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_STATS_PATH,
)
from customer_claims_rag.exceptions import IngestionError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.token_counter import TiktokenCounter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build RAG chunk corpus from clean Markdown documents.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with clean Markdown (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help=f"Stats JSON path (default: {DEFAULT_STATS_PATH})",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"tiktoken encoding name (default: {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include documents with status other than active",
    )
    parser.add_argument(
        "--fail-on-soft-limit",
        action="store_true",
        help="Fail when non-atomic chunks fall outside soft token limits",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed summary and diagnostic traceback on error",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    permitted_root = project_root.resolve()

    try:
        counter = TiktokenCounter(args.encoding)
        builder = CorpusBuilder(
            token_counter=counter,
            include_inactive=args.include_inactive,
            fail_on_soft_limit=args.fail_on_soft_limit,
            permitted_root=permitted_root,
        )
        documents, chunks, stats = builder.build_and_export(
            args.input_dir,
            args.output,
            args.stats_output,
        )
    except IngestionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print(
        f"Loaded {stats.documents_total} documents, "
        f"generated {stats.chunks_total} chunks."
    )
    print(f"Output: {args.output}")
    print(f"Stats: {args.stats_output}")

    if args.verbose:
        print("\nChunks by document:")
        for doc_id, count in stats.chunks_by_document.items():
            print(f"  {doc_id}: {count}")
        print("\nToken stats:")
        print(f"  min={stats.min_tokens}, max={stats.max_tokens}, "
              f"avg={stats.average_tokens:.1f}, median={stats.median_tokens:.1f}")
        print(f"  FAQ chunks: {stats.faq_chunk_count}")
        print(f"  template chunks: {stats.template_chunk_count}")
        print(f"  overlap chunks: {stats.overlap_chunks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
