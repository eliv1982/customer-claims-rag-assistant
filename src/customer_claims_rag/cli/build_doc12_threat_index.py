"""Build doc12 threat atomic candidate index from stacked doc08+doc12 overlays."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INPUT_DIR
from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import (
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
    compute_doc12_chunk_diff,
    verify_doc08_overlay_unchanged,
)
from customer_claims_rag.retrieval.experiment_embedding_cache import experiment_cache_path
from customer_claims_rag.retrieval.experiment_index_publish import (
    experiment_staging_dir,
    publish_staged_index,
)
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval_config import RetrievalSettings
from customer_claims_rag.token_counter import TiktokenCounter

DEFAULT_CONFIG = Path("configs/experiments/doc12_threat_atomic_units_v1.json")


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Build doc12 threat atomic candidate index (doc08 exp. + doc12 overlay).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--build-run-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_build_doc12_threat_index(
    *,
    config: dict,
    canonical_dir: Path,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    rebuild: bool,
    permitted_root: Path,
    settings: RetrievalSettings | None = None,
    build_run_id: str | None = None,
) -> int:
    resolved_settings = settings or RetrievalSettings.from_env()
    resolved_index = validate_index_dir(index_dir, project_root=permitted_root, input_dir=canonical_dir)
    if not rebuild:
        raise RetrievalError("doc12 experiment index build requires --rebuild")

    doc08_overlay = permitted_root / config["baseline_corpus"]["doc08_overlay_path"]
    doc12_overlay = permitted_root / config["candidate_overlay"]["overlay_path"]
    expected_doc08_fp = config["baseline_corpus"].get("expected_doc08_fingerprint")

    overlay_result = build_doc12_experimental_corpora(
        canonical_dir=canonical_dir,
        doc08_overlay_path=doc08_overlay,
        doc12_overlay_path=doc12_overlay,
        permitted_root=permitted_root,
    )
    staging_dir, run_id = experiment_staging_dir(resolved_index, run_id=build_run_id or uuid.uuid4().hex)
    try:
        verify_doc08_overlay_unchanged(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
            expected_doc08_fingerprint=expected_doc08_fp,
        )
        diff = compute_doc12_chunk_diff(
            overlay_result.baseline_chunks,
            overlay_result.candidate_chunks,
        )
        if not diff.non_target_byte_identical:
            raise RetrievalError("non-doc12 chunks are not byte-identical after overlay")

        builder = CorpusBuilder(permitted_root=permitted_root, token_counter=TiktokenCounter())
        embedding_provider = create_embedding_provider(
            model_name=embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        vector_store = create_vector_store(
            index_dir=staging_dir,
            collection_name=collection_name,
        )
        index_builder = IndexBuilder(
            corpus_builder=builder,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=staging_dir,
            batch_size=batch_size,
        )
        report = index_builder.build_from_chunks(
            documents=[],
            chunks=overlay_result.candidate_chunks,
            rebuild=True,
            embedding_cache_path=experiment_cache_path(resolved_index),
            build_run_id=run_id,
        )
        vector_store.close()
        del vector_store
        publish_staged_index(staging_dir, resolved_index)
        manifest = load_manifest(resolved_index)
        print(f"experiment_id={config['experiment_id']}")
        print(f"index_dir={resolved_index}")
        print(f"chunks={report.chunks}")
        print(f"fingerprint={report.fingerprint}")
        print(f"build_run_id={run_id}")
        print(f"chunk_payload_digest={manifest.chunk_payload_digest}")
        print(f"embedding_digest={manifest.embedding_digest}")
        print(f"collection_content_digest={manifest.collection_content_digest}")
        print(f"doc08_fingerprint={overlay_result.candidate_doc08_fingerprint}")
        print(f"doc12_baseline_chunks={len(overlay_result.baseline_doc12_chunks)}")
        print(f"doc12_candidate_chunks={len(overlay_result.candidate_doc12_chunks)}")
        print(f"doc12_added={diff.added_chunk_ids}")
        print(f"doc12_removed={diff.removed_chunk_ids}")
        print(f"doc12_changed={diff.changed_chunk_ids}")
        return 0
    except RetrievalError:
        raise
    finally:
        cleanup_overlay_temp_dir(overlay_result.temp_input_dir)


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    root = project_root()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    index_dir = args.index_dir
    if index_dir is None:
        index_dir = root / config["candidate_index_dir"]

    canonical_dir = args.canonical_dir
    if not canonical_dir.is_absolute():
        canonical_dir = root / canonical_dir

    try:
        return run_build_doc12_threat_index(
            config=config,
            canonical_dir=canonical_dir,
            index_dir=index_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            permitted_root=root,
            build_run_id=args.build_run_id,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
