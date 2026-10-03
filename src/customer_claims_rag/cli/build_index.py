"""Build persistent vector index from ingestion corpus."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

from customer_claims_rag.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_INPUT_DIR,
)
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.exceptions import IngestionError, RetrievalError
from customer_claims_rag.ingestion.canonical_corpus import (
    CanonicalCorpusBuild,
    build_canonical_corpus,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval.ports import VectorStore
from customer_claims_rag.retrieval_config import RetrievalSettings
from customer_claims_rag.token_counter import TiktokenCounter


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Build persistent vector index from clean Markdown corpus.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory with clean Markdown; every .md file in it is indexed "
            f"(default: {DEFAULT_INPUT_DIR}). Not the production corpus: use --corpus-manifest"
        ),
    )
    source.add_argument(
        "--corpus-manifest",
        type=Path,
        default=None,
        help=(
            "Canonical corpus manifest (e.g. configs/corpus/foodflow_production_v1.json): index "
            "exactly the documents it selects, after verifying sources, chunk topology and "
            "fingerprint against it. This is how the production index is built"
        ),
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
        "--batch-size",
        type=int,
        default=settings.embedding_batch_size,
        help=f"Embedding batch size (default: {settings.embedding_batch_size})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Perform destructive full rebuild of the vector index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --corpus-manifest: verify the canonical corpus and print the build plan. "
            "Makes no embedding request, needs no API key and does not touch the index"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic traceback on error",
    )
    return parser


def _print_dry_run_plan(
    canonical: CanonicalCorpusBuild,
    *,
    index_dir: Path,
    collection_name: str,
    batch_size: int,
) -> None:
    identity = canonical.identity
    print("Dry run: canonical corpus verified; no embedding request was made.")
    print(f"Canonical corpus: {canonical.manifest.corpus_id}")
    print(f"Documents: {identity.document_count} ({', '.join(identity.document_ids)})")
    print(f"Chunks: {identity.chunk_count}")
    print(f"Chunk payload digest: {identity.chunk_payload_digest}")
    print(f"Corpus fingerprint ({identity.embedding_model}): {identity.corpus_fingerprint}")
    print(f"Embedding batches of {batch_size}: {math.ceil(identity.chunk_count / batch_size)}")
    print(f"Collection: {collection_name}")
    print(f"Index directory (not touched): {index_dir}")


def run_build(
    *,
    input_dir: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    rebuild: bool,
    permitted_root: Path,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
    settings: RetrievalSettings | None = None,
    corpus_manifest: Path | None = None,
    dry_run: bool = False,
) -> int:
    resolved_settings = settings or RetrievalSettings.from_env()
    vector_store: VectorStore | None = None
    canonical: CanonicalCorpusBuild | None = None
    try:
        batch_size = resolved_settings.validate_batch_size(batch_size)
        if dry_run and corpus_manifest is None:
            raise RetrievalError("--dry-run requires --corpus-manifest")
        if corpus_manifest is not None:
            # Verified before any provider, store or index file exists: a wrong corpus must
            # never reach the embedding step.
            manifest = load_canonical_corpus_manifest((permitted_root / corpus_manifest).resolve())
            canonical = build_canonical_corpus(
                manifest,
                project_root=permitted_root,
                token_counter=TiktokenCounter(manifest.tokenizer_encoding),
                embedding_model=embedding_model,
            )
            input_dir = canonical.source_dir
        resolved_index_dir = validate_index_dir(
            index_dir,
            project_root=permitted_root,
            input_dir=input_dir,
        )
        if dry_run and canonical is not None:
            _print_dry_run_plan(
                canonical,
                index_dir=resolved_index_dir,
                collection_name=collection_name,
                batch_size=batch_size,
            )
            return 0
        embedding_provider = embedding_provider_factory(
            model_name=embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        vector_store = vector_store_factory(
            index_dir=resolved_index_dir,
            collection_name=collection_name,
        )
        corpus_builder = CorpusBuilder(
            token_counter=TiktokenCounter(),
            permitted_root=permitted_root,
        )
        builder = IndexBuilder(
            corpus_builder=corpus_builder,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            batch_size=batch_size,
        )
        if canonical is not None:
            report = builder.build_from_chunks(
                canonical.documents,
                canonical.chunks,
                rebuild=rebuild,
            )
        else:
            report = builder.build_from_directory(input_dir, rebuild=rebuild)
    except (RetrievalError, IngestionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if vector_store is not None:
            vector_store.close()

    if canonical is not None:
        print(f"Canonical corpus: {canonical.manifest.corpus_id}")
        print(f"Chunk payload digest: {canonical.identity.chunk_payload_digest}")
    print(f"Status: {report.status}")
    print(f"Documents: {report.documents}")
    print(f"Chunks: {report.chunks}")
    print(f"Embedding model: {report.embedding_model}")
    print(f"Collection: {report.collection_name}")
    print(f"Index directory: {report.index_dir}")
    print(f"Vector dimension: {report.vector_dimension}")
    print(f"Fingerprint: {report.fingerprint}")
    print(f"Elapsed: {report.elapsed_seconds:.2f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = RetrievalSettings.from_env()

    try:
        return run_build(
            input_dir=args.input_dir,
            index_dir=args.index_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            permitted_root=Path.cwd().resolve(),
            settings=settings,
            corpus_manifest=args.corpus_manifest,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
