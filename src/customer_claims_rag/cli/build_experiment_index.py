"""Build experiment vector index from canonical corpus with a single-document overlay."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INPUT_DIR
from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import (
    build_baseline_and_overlay_chunks,
    cleanup_overlay_temp_dir,
    compute_doc08_chunk_diff,
)
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval_config import RetrievalSettings
from customer_claims_rag.token_counter import TiktokenCounter


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Build experiment index with single-document corpus overlay.",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment identifier (e.g. doc08_atomic_risk_units_v1)",
    )
    parser.add_argument(
        "--overlay-document",
        type=Path,
        required=True,
        help="Path to overlay Markdown for the replaced document",
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Canonical clean markdown directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Candidate index output directory",
    )
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"Chroma collection name (default: {settings.collection_name})",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help=f"Embedding model (default: {settings.embedding_model})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help=f"Embedding batch size (default: {settings.embedding_batch_size})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Perform destructive full rebuild",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print traceback on error",
    )
    return parser


def run_build_experiment_index(
    *,
    experiment_id: str,
    overlay_document: Path,
    canonical_dir: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    rebuild: bool,
    permitted_root: Path,
    settings: RetrievalSettings | None = None,
) -> int:
    resolved_settings = settings or RetrievalSettings.from_env()
    root = permitted_root
    resolved_index = validate_index_dir(index_dir, project_root=root, input_dir=canonical_dir)

    overlay_result = build_baseline_and_overlay_chunks(
        canonical_dir=canonical_dir,
        overlay_document_path=overlay_document,
        permitted_root=root,
    )
    try:
        diff = compute_doc08_chunk_diff(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
        )
        if not diff.non_doc08_byte_identical:
            raise RetrievalError("non-doc08 chunks are not byte-identical after overlay")

        builder = CorpusBuilder(permitted_root=root, token_counter=TiktokenCounter())
        embedding_provider = create_embedding_provider(
            model_name=embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        vector_store = create_vector_store(
            index_dir=resolved_index,
            collection_name=collection_name,
        )
        index_builder = IndexBuilder(
            corpus_builder=builder,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index,
            batch_size=batch_size,
        )
        documents = sorted(
            {c.document_id for c in overlay_result.candidate_chunks},
        )
        report = index_builder.build_from_chunks(
            documents=[],  # document count from chunks
            chunks=overlay_result.candidate_chunks,
            rebuild=rebuild,
        )
        # Fix document count in report - IndexBuildReport uses len(documents) from param
        print(f"experiment_id={experiment_id}")
        print(f"index_dir={resolved_index}")
        print(f"chunks={report.chunks}")
        print(f"fingerprint={report.fingerprint}")
        print(f"doc08_baseline_chunks={len(overlay_result.baseline_doc08_chunks)}")
        print(f"doc08_candidate_chunks={len(overlay_result.candidate_doc08_chunks)}")
        print(f"doc08_added={diff.added_chunk_ids}")
        print(f"doc08_removed={diff.removed_chunk_ids}")
        print(f"doc08_changed={diff.changed_chunk_ids}")
        vector_store.close()
        return 0
    except RetrievalError:
        raise
    finally:
        cleanup_overlay_temp_dir(overlay_result.temp_input_dir)


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    root = project_root()
    parser = build_parser()
    args = parser.parse_args(argv)

    index_dir = args.index_dir
    if index_dir is None:
        index_dir = root / "data" / "04_index_experiments" / args.experiment_id

    try:
        return run_build_experiment_index(
            experiment_id=args.experiment_id,
            overlay_document=args.overlay_document,
            canonical_dir=args.canonical_dir,
            index_dir=index_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            permitted_root=root,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
