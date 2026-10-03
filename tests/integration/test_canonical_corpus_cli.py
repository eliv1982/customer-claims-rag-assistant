"""The supported build path for the canonical corpus, from clean Markdown to the release gate.

    clean Markdown -> canonical selection -> chunking -> embeddings -> Chroma -> index manifest

Everything before the embedding step is deterministic and needs no credentials; these tests prove
that it runs on a fresh clone, that a wrong corpus is refused *before* any embedding request or
index file exists, and that what the build writes satisfies the release validator. The embedding
step itself (the only call to OpenAI) is replaced by the fake provider.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.cli import build_chunks as chunks_cli
from customer_claims_rag.cli import build_index as index_cli
from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.release.posture import (
    load_release_posture_descriptor,
    resolve_release_target,
    validate_release_posture_for_production,
)
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.manifest import load_manifest
from tests.release_posture_helpers import PRODUCTION_FROZEN_CONFIG_HASH, stage_project_configs

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ARG = DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix()
MODEL = "text-embedding-3-small"


@pytest.fixture
def project(temp_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clone-like project root: sources, the committed manifest, empty output directories."""
    target = temp_project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE, target)
    monkeypatch.chdir(temp_project)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    return temp_project


def _refresh(project: Path) -> None:
    # The default lane's stand-in tokenizer cannot reproduce the committed (real cl100k_base)
    # identity, so the manifest is refreshed through the supported CLI path first.
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--refresh-manifest"]) == 0


def _chunk_document_ids(project: Path) -> set[str]:
    lines = (project / "data" / "03_chunks" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["document_id"] for line in lines}


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


def _must_not_embed(**_kwargs):
    raise AssertionError("an embedding provider was created for a corpus that failed verification")


def _must_not_open_a_store(**_kwargs):
    raise AssertionError("a vector store was opened for a corpus that failed verification")


def _build_index(project: Path, **overrides) -> int:
    arguments = dict(
        input_dir=project / "data" / "02_clean_markdown",
        index_dir=project / "data" / "04_index_production",
        collection_name="customer_claims",
        embedding_model=MODEL,
        batch_size=16,
        rebuild=True,
        permitted_root=project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_store_factory,
        corpus_manifest=Path(MANIFEST_ARG),
    )
    arguments.update(overrides)
    return index_cli.run_build(**arguments)


# --- build-chunks -----------------------------------------------------------------------------


def test_build_chunks_builds_exactly_the_selected_documents(project: Path, capsys) -> None:
    _refresh(project)
    capsys.readouterr()
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG]) == 0
    out = capsys.readouterr().out
    manifest = load_canonical_corpus_manifest(project / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    assert _chunk_document_ids(project) == set(manifest.document_ids)
    assert "Loaded 10 documents" in out
    assert f"Canonical corpus: {manifest.corpus_id}" in out
    assert f"Chunk payload digest: {manifest.expected.chunk_payload_digest}" in out
    assert f"Corpus fingerprint ({MODEL}): {manifest.expected.corpus_fingerprint}" in out
    assert "Manifest: verified" in out


def test_build_chunks_without_a_manifest_still_builds_the_whole_directory(project: Path) -> None:
    assert chunks_cli.main([]) == 0
    assert len(_chunk_document_ids(project)) == 15  # the legacy directory build is unchanged


def test_build_chunks_refuses_a_corpus_that_drifted_and_writes_nothing(project: Path, capsys) -> None:
    _refresh(project)
    source = project / "data" / "02_clean_markdown" / "05_compensation_policy.md"
    source.write_text(source.read_text(encoding="utf-8") + "\nAn unreviewed sentence.\n", encoding="utf-8")
    chunks_file = project / "data" / "03_chunks" / "chunks.jsonl"
    chunks_file.unlink(missing_ok=True)
    capsys.readouterr()
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG]) == 1
    err = capsys.readouterr().err
    assert "05_compensation_policy.md" in err and "--refresh-manifest" in err
    assert not chunks_file.exists()


def test_refresh_manifest_accepts_the_change_and_the_next_build_verifies(project: Path, capsys) -> None:
    _refresh(project)
    source = project / "data" / "02_clean_markdown" / "05_compensation_policy.md"
    source.write_text(source.read_text(encoding="utf-8") + "\nA reviewed sentence.\n", encoding="utf-8")
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG]) == 1
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--refresh-manifest"]) == 0
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG]) == 0
    assert "Manifest: verified" in capsys.readouterr().out


def test_build_chunks_refuses_another_tokenizer(project: Path, capsys, monkeypatch) -> None:
    from customer_claims_rag.token_counter import FakeTokenCounter

    class OtherEncoding(FakeTokenCounter):
        """Another encoding by name only: loading a second real vocabulary would need the network."""

        def __init__(self, encoding_name: str = "cl100k_base") -> None:
            self.encoding_name = encoding_name

    _refresh(project)
    monkeypatch.setattr(chunks_cli, "TiktokenCounter", OtherEncoding)
    capsys.readouterr()
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--encoding", "o200k_base"]) == 1
    assert "pinned to the 'cl100k_base' tokenizer" in capsys.readouterr().err


def test_corpus_manifest_and_input_dir_are_mutually_exclusive(project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--input-dir", "data/02_clean_markdown"])
    assert exc.value.code == 2


def test_refresh_manifest_requires_a_manifest(project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        chunks_cli.main(["--refresh-manifest"])
    assert exc.value.code == 2


def test_a_failed_export_leaves_the_manifest_unrefreshed(project: Path) -> None:
    manifest_path = project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    outside = project / "data" / "02_clean_markdown" / "out.jsonl"  # export path inside the sources: refused
    code = chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--refresh-manifest", "--output", str(outside)])
    assert code == 1
    assert manifest_path.read_bytes() == before


# --- build-index: the path up to the embedding boundary -------------------------------------------


def test_a_corpus_that_fails_verification_never_reaches_embedding_or_the_store(project: Path, capsys) -> None:
    """The manifest no longer describes what the sources build: refused before anything exists."""
    _refresh(project)
    manifest_path = project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["expected"]["chunk_payload_digest"] = "1" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    capsys.readouterr()
    code = _build_index(
        project,
        embedding_provider_factory=_must_not_embed,
        vector_store_factory=_must_not_open_a_store,
    )
    assert code == 1
    assert "identity mismatch" in capsys.readouterr().err
    assert not (project / "data" / "04_index_production").exists()


def test_dry_run_verifies_the_corpus_and_makes_no_request_and_no_index(project: Path, capsys) -> None:
    _refresh(project)
    capsys.readouterr()
    code = _build_index(
        project,
        dry_run=True,
        embedding_provider_factory=_must_not_embed,
        vector_store_factory=_must_not_open_a_store,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "no embedding request was made" in out
    assert "Documents: 10" in out and "Chunks: " in out and "Embedding batches of 16" in out
    assert not (project / "data" / "04_index_production").exists()


def test_dry_run_needs_a_manifest(project: Path, capsys) -> None:
    code = _build_index(project, corpus_manifest=None, dry_run=True)
    assert code == 1
    assert "--dry-run requires --corpus-manifest" in capsys.readouterr().err


def test_another_embedding_model_is_refused_for_the_canonical_corpus(project: Path, capsys) -> None:
    _refresh(project)
    code = _build_index(
        project,
        embedding_model="some-other-model",
        embedding_provider_factory=_must_not_embed,
        vector_store_factory=_must_not_open_a_store,
    )
    assert code == 1
    assert "pinned to embedding model" in capsys.readouterr().err


def test_build_index_cli_main_dry_run_end_to_end(project: Path, capsys, monkeypatch) -> None:
    _refresh(project)
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # a dry run needs no credentials
    code = index_cli.main(
        ["--corpus-manifest", MANIFEST_ARG, "--index-dir", "data/04_index_production", "--dry-run"]
    )
    assert code == 0
    assert "Dry run: canonical corpus verified" in capsys.readouterr().out


def test_corpus_manifest_and_input_dir_are_mutually_exclusive_for_build_index(project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        index_cli.main(["--corpus-manifest", MANIFEST_ARG, "--input-dir", "data/02_clean_markdown", "--rebuild"])
    assert exc.value.code == 2


def test_the_index_the_build_writes_satisfies_the_release_validator(project: Path, capsys) -> None:
    """Build -> manifest -> validator: the whole supported path, with the fake embedding step."""
    _refresh(project)
    capsys.readouterr()
    assert _build_index(project) == 0
    out = capsys.readouterr().out
    corpus = load_canonical_corpus_manifest(project / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    assert f"Canonical corpus: {corpus.corpus_id}" in out
    assert f"Fingerprint: {corpus.expected.corpus_fingerprint}" in out

    index_manifest = load_manifest(project / "data" / "04_index_production")
    assert index_manifest.corpus_fingerprint == corpus.expected.corpus_fingerprint
    assert index_manifest.chunk_payload_digest == corpus.expected.chunk_payload_digest
    assert index_manifest.chunk_count == corpus.expected.chunk_count
    assert index_manifest.document_count == 10
    assert index_manifest.embedding_digest and index_manifest.collection_content_digest

    stage_project_configs(project)
    descriptor_path = project / "configs" / "release" / "production_posture.json"
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "release_posture_id": "built-here",
                "default_target": "active",
                "limitations_doc": "docs/06_release_posture.md",
                "frozen_retrieval_config_path": "configs/retrieval/vector_pool_expansion_v1.json",
                "expected_frozen_config_hash": PRODUCTION_FROZEN_CONFIG_HASH,
                "targets": {
                    "active": {
                        "status": "selected_production_release",
                        "corpus_manifest": MANIFEST_ARG,
                        "index_path": "data/04_index_production",
                        "collection_name": "customer_claims",
                        "embedding_model": MODEL,
                        "vector_dimension": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    descriptor = load_release_posture_descriptor(descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=project)
    store = ChromaVectorStore(index_dir=resolved.index_dir, collection_name="customer_claims", open_existing=True)
    try:
        diagnostics = validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=project,
            vector_store=store,
            frozen_retrieval_config_path=project / "configs" / "retrieval" / "vector_pool_expansion_v1.json",
        )
    finally:
        store.close()
    assert diagnostics.integrity == "store_recomputed"
    assert diagnostics.chunk_count == corpus.expected.chunk_count


def test_the_indexed_documents_are_the_selection_not_the_directory(project: Path) -> None:
    _refresh(project)
    assert _build_index(project) == 0
    store = ChromaVectorStore(
        index_dir=project / "data" / "04_index_production",
        collection_name="customer_claims",
        open_existing=True,
    )
    try:
        indexed = {document_id[:2] for document_id in store.list_document_ids()}
    finally:
        store.close()
    assert indexed == {f"{n:02d}" for n in range(1, 11)}


def test_the_legacy_directory_build_is_unchanged(project: Path) -> None:
    code = _build_index(project, corpus_manifest=None, embedding_model="fake-embedding-model")
    assert code == 0
    assert load_manifest(project / "data" / "04_index_production").document_count == 15
