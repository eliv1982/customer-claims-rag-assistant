"""Export committed frozen embedding snapshot from experiment embedding cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INPUT_DIR
from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.ingestion.corpus_overlay import (
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
)
from customer_claims_rag.retrieval.embedding_snapshot import (
    SnapshotValidationError,
    export_snapshot_from_experiment_cache,
)
from customer_claims_rag.retrieval.experiment_embedding_cache import experiment_cache_path

DEFAULT_CONFIG = Path("configs/experiments/doc12_threat_atomic_units_v1.json")
DEFAULT_NPZ = Path(
    "data/05_evaluation/embedding_snapshots/doc12_threat_atomic_units_v1.npz"
)
DEFAULT_MANIFEST = Path(
    "data/05_evaluation/embedding_snapshots/doc12_threat_atomic_units_v1.manifest.json"
)


def _current_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export frozen embedding snapshot for doc12 threat experiment.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--canonical-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--build-run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    root = project_root()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    canonical_dir = args.canonical_dir
    if not canonical_dir.is_absolute():
        canonical_dir = root / canonical_dir

    index_dir = args.index_dir
    if index_dir is None:
        index_dir = root / config["candidate_index_dir"]
    elif not index_dir.is_absolute():
        index_dir = root / index_dir

    npz_path = args.npz_path if args.npz_path.is_absolute() else root / args.npz_path
    manifest_path = (
        args.manifest_path if args.manifest_path.is_absolute() else root / args.manifest_path
    )

    from customer_claims_rag.retrieval_config import RetrievalSettings

    settings = RetrievalSettings.from_env()
    embedding_model = args.embedding_model or settings.embedding_model

    doc08_overlay = root / config["baseline_corpus"]["doc08_overlay_path"]
    doc12_overlay = root / config["candidate_overlay"]["overlay_path"]
    overlay = build_doc12_experimental_corpora(
        canonical_dir=canonical_dir,
        doc08_overlay_path=doc08_overlay,
        doc12_overlay_path=doc12_overlay,
        permitted_root=root,
    )
    try:
        cache_path = experiment_cache_path(index_dir)
        lineage = {
            "build_run_id": args.build_run_id,
            "selected_before_independent_audit": True,
            "matches_post_r1_authoritative_index": True,
            "source": "experiment_embeddings_v1.json from post-R1 authoritative build",
        }
        manifest = export_snapshot_from_experiment_cache(
            chunks=overlay.candidate_chunks,
            cache_path=cache_path,
            npz_path=npz_path,
            manifest_path=manifest_path,
            embedding_model=embedding_model,
            candidate_experiment_id=config["experiment_id"],
            creation_source_commit=_current_commit(root),
            project_root=root,
            lineage=lineage,
        )
        print(f"snapshot_path={manifest['snapshot_path']}")
        print(f"embedding_digest={manifest['embedding_digest']}")
        print(f"snapshot_digest={manifest['snapshot_digest']}")
        print(f"chunk_payload_digest={manifest['chunk_payload_digest']}")
        print(f"corpus_fingerprint={manifest['corpus_fingerprint']}")
        return 0
    except (SnapshotValidationError, RetrievalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup_overlay_temp_dir(overlay.temp_input_dir)


if __name__ == "__main__":
    raise SystemExit(main())
