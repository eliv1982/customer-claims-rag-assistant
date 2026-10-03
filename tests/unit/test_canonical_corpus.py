"""The canonical production corpus: explicit selection, source integrity and identity.

Default lane: the offline stand-in tokenizer is active, so no test here claims anything about the
production chunk topology (``tests/unit/test_real_tokenizer_golden.py`` does, in the real lane).
What these tests establish is the machinery and the repository-level facts that need no tokenizer:
which documents are selected, that the sources still match what the manifest approved, that
nothing is selected by accident, and that every way of drifting is refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from customer_claims_rag.exceptions import CanonicalCorpusError
from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    CanonicalCorpusManifest,
    build_canonical_corpus,
    compute_corpus_identity,
    load_canonical_corpus_manifest,
    normalized_source_text,
    refresh_canonical_corpus_manifest,
    source_sha256,
    verify_corpus_sources,
    write_canonical_corpus_manifest,
)
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.token_counter import FakeTokenCounter, TiktokenCounter

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE
INCLUDED = tuple(f"{n:02d}" for n in range(1, 11))
EXCLUDED = tuple(f"{n:02d}" for n in range(11, 16))


def _committed() -> CanonicalCorpusManifest:
    return load_canonical_corpus_manifest(COMMITTED_MANIFEST)


# --- the committed definition ------------------------------------------------------------------


def test_committed_manifest_selects_documents_one_to_ten() -> None:
    manifest = _committed()
    assert [d[:2] for d in manifest.document_ids] == list(INCLUDED)
    assert manifest.corpus_id == "foodflow-10doc-corpus-v2"
    assert manifest.source_dir == "data/02_clean_markdown"
    assert manifest.tokenizer_encoding == "cl100k_base"
    assert manifest.expected.embedding_model == "text-embedding-3-small"
    assert manifest.expected.document_count == 10


def test_committed_manifest_declares_documents_eleven_to_fifteen_as_excluded() -> None:
    manifest = _committed()
    assert [d.document_id[:2] for d in manifest.excluded_documents] == list(EXCLUDED)
    for excluded in manifest.excluded_documents:
        assert (ROOT / excluded.evidence).is_file(), f"{excluded.document_id}: evidence file missing"
        assert "REJECT" in excluded.reason, excluded.document_id
    assert not set(manifest.document_ids) & {d.document_id for d in manifest.excluded_documents}


def test_every_markdown_source_is_declared_and_matches_the_manifest() -> None:
    """A document added, removed or edited without updating the manifest fails here."""
    manifest = _committed()
    report = verify_corpus_sources(manifest, project_root=ROOT)
    assert report.ok, report.describe()


def test_committed_manifest_identity_values_are_wellformed_and_not_historical() -> None:
    expected = _committed().expected
    assert expected.chunk_count >= 10
    for digest in (expected.chunk_payload_digest, expected.corpus_fingerprint):
        assert len(digest) == 64 and digest == digest.lower()
        assert set(digest) != {"0"}, "placeholder digest left in the manifest"


# --- source hashing ---------------------------------------------------------------------------


def test_source_hash_does_not_depend_on_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    lf.write_bytes("a\nb\n".encode("utf-8"))
    crlf.write_bytes("a\r\nb\r\n".encode("utf-8"))
    assert source_sha256(lf) == source_sha256(crlf)
    assert source_sha256(lf) == hashlib.sha256(b"a\nb\n").hexdigest()
    assert normalized_source_text("a\r\nb\rc") == "a\nb\nc"


def test_recorded_hashes_equal_the_files_in_this_checkout_whatever_their_line_endings() -> None:
    for document in _committed().documents:
        assert source_sha256(ROOT / "data" / "02_clean_markdown" / document.file) == document.source_sha256


# --- manifest validation ----------------------------------------------------------------------


def _payload() -> dict:
    return json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))


def _swap_documents(payload: dict) -> None:
    payload["documents"][0], payload["documents"][1] = payload["documents"][1], payload["documents"][0]


def _duplicate_document(payload: dict) -> None:
    payload["documents"][1] = copy.deepcopy(payload["documents"][0])


def _include_an_excluded_document(payload: dict) -> None:
    payload["excluded_documents"][0]["document_id"] = payload["documents"][0]["document_id"]
    payload["excluded_documents"][0]["file"] = payload["documents"][0]["file"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_swap_documents, "ascending document_id order"),
        (_duplicate_document, "unique"),
        (_include_an_excluded_document, "both included and excluded"),
        (lambda p: p["documents"][0].update(file="other.md"), "file must be"),
        (lambda p: p["documents"][0].update(source_sha256="abc"), "64 lowercase hex"),
        (lambda p: p["expected"].update(document_count=9), "document_count"),
        (lambda p: p["expected"].update(corpus_fingerprint="XYZ"), "64 lowercase hex"),
        (lambda p: p.update(source_dir="/abs/dir"), "repository-relative"),
        (lambda p: p.update(source_dir="../outside"), "traversal"),
        (lambda p: p.update(schema_version="9.9.9"), "unsupported canonical corpus schema_version"),
        (lambda p: p.update(documents=[]), "must not be empty"),
        (lambda p: p.update(surprise=True), "surprise"),
        (lambda p: p["excluded_documents"][0].pop("reason"), "reason"),
    ],
)
def test_invalid_manifest_is_rejected(tmp_path: Path, mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CanonicalCorpusError, match=message):
        load_canonical_corpus_manifest(path)


def test_missing_and_unparsable_manifests_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(CanonicalCorpusError, match="not found"):
        load_canonical_corpus_manifest(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(CanonicalCorpusError, match="invalid JSON"):
        load_canonical_corpus_manifest(broken)


# --- a project with a refreshed manifest (stand-in tokenizer) ---------------------------------


@pytest.fixture
def corpus_project(temp_project: Path) -> Path:
    """Project root with all 15 documents and the committed manifest (not yet refreshed).

    The committed manifest pins the production tokenizer's identity, which the default lane's
    stand-in tokenizer cannot reproduce, so tests refresh it first (``_refresh_and_write``).
    """
    manifest_path = temp_project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    shutil.copy(COMMITTED_MANIFEST, manifest_path)
    return temp_project


def _refresh_and_write(project: Path):
    manifest_path = project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    build = refresh_canonical_corpus_manifest(
        manifest_path,
        project_root=project,
        token_counter=TiktokenCounter(),
    )
    write_canonical_corpus_manifest(manifest_path, build.manifest)
    return manifest_path, build


def test_refresh_computes_but_does_not_write(temp_project: Path) -> None:
    manifest_path = temp_project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    shutil.copy(COMMITTED_MANIFEST, manifest_path)
    before = manifest_path.read_bytes()
    build = refresh_canonical_corpus_manifest(
        manifest_path, project_root=temp_project, token_counter=TiktokenCounter()
    )
    assert manifest_path.read_bytes() == before
    assert build.manifest.expected.chunk_count == len(build.chunks) > 0


def test_refresh_changes_only_derived_values_never_the_selection(corpus_project: Path) -> None:
    manifest_path = corpus_project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, build = _refresh_and_write(corpus_project)
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [d["document_id"] for d in after["documents"]] == [d["document_id"] for d in before["documents"]]
    assert after["excluded_documents"] == before["excluded_documents"]
    for key in ("schema_version", "corpus_id", "description", "source_dir", "tokenizer_encoding"):
        assert after[key] == before[key]
    assert after["expected"]["chunk_count"] == len(build.chunks)
    assert list(after["expected"]) == list(before["expected"])  # key order is stable


def test_a_refreshed_manifest_verifies_and_builds_only_the_selection(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    manifest = load_canonical_corpus_manifest(corpus_project / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    build = build_canonical_corpus(manifest, project_root=corpus_project, token_counter=TiktokenCounter())
    assert tuple(sorted({c.document_id for c in build.chunks})) == manifest.document_ids
    assert {c.document_id[:2] for c in build.chunks} == set(INCLUDED)
    assert build.identity.document_count == 10


def test_excluded_documents_are_never_read(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    excluded_file = corpus_project / "data" / "02_clean_markdown" / "12_staff_safety_and_threat_handling.md"
    excluded_file.write_text("not a document at all: no front matter", encoding="utf-8")
    manifest = load_canonical_corpus_manifest(corpus_project / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    build = build_canonical_corpus(manifest, project_root=corpus_project, token_counter=TiktokenCounter())
    assert "12_staff_safety_and_threat_handling" not in {c.document_id for c in build.chunks}


def test_selection_order_does_not_change_the_identity(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    manifest = load_canonical_corpus_manifest(corpus_project / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    builder = CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=corpus_project.resolve())
    source_dir = corpus_project / "data" / "02_clean_markdown"
    forward = builder.build_from_selection(source_dir, list(manifest.file_names))[1]
    backward = builder.build_from_selection(source_dir, list(reversed(manifest.file_names)))[1]
    model = manifest.expected.embedding_model
    assert compute_corpus_identity(forward, embedding_model=model) == compute_corpus_identity(
        backward, embedding_model=model
    )


def test_identity_is_stable_across_repeated_builds(corpus_project: Path) -> None:
    manifest_path, first = _refresh_and_write(corpus_project)
    manifest = load_canonical_corpus_manifest(manifest_path)
    second = build_canonical_corpus(manifest, project_root=corpus_project, token_counter=TiktokenCounter())
    assert second.identity == first.identity


def test_identity_does_not_depend_on_the_checkout_location_or_line_endings(
    corpus_project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    manifest_path, first = _refresh_and_write(corpus_project)
    other = tmp_path_factory.mktemp("elsewhere")
    shutil.copytree(corpus_project / "data", other / "data")
    for path in (other / "data" / "02_clean_markdown").glob("*.md"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    (other / "configs" / "corpus").mkdir(parents=True)
    shutil.copy(manifest_path, other / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    manifest = load_canonical_corpus_manifest(other / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    again = build_canonical_corpus(manifest, project_root=other, token_counter=TiktokenCounter())
    assert again.identity == first.identity


# --- every way of drifting is refused ----------------------------------------------------------


def _manifest_in(project: Path) -> CanonicalCorpusManifest:
    return load_canonical_corpus_manifest(project / DEFAULT_CORPUS_MANIFEST_RELATIVE)


def test_edited_source_is_refused(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    target = corpus_project / "data" / "02_clean_markdown" / "04_refund_policy.md"
    target.write_text(target.read_text(encoding="utf-8") + "\nA new sentence.\n", encoding="utf-8")
    with pytest.raises(CanonicalCorpusError, match=r"differ from the manifest.*04_refund_policy\.md"):
        build_canonical_corpus(_manifest_in(corpus_project), project_root=corpus_project, token_counter=TiktokenCounter())


def test_missing_source_is_refused(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    (corpus_project / "data" / "02_clean_markdown" / "10_customer_faq.md").unlink()
    with pytest.raises(CanonicalCorpusError, match=r"missing source files.*10_customer_faq\.md"):
        build_canonical_corpus(_manifest_in(corpus_project), project_root=corpus_project, token_counter=TiktokenCounter())


def test_undeclared_markdown_file_is_refused(corpus_project: Path) -> None:
    """A new document in the source directory forces a decision: include it or exclude it."""
    _refresh_and_write(corpus_project)
    source = corpus_project / "data" / "02_clean_markdown"
    shutil.copy(source / "01_service_overview.md", source / "16_new_document.md")
    manifest = _manifest_in(corpus_project)
    report = verify_corpus_sources(manifest, project_root=corpus_project)
    assert report.undeclared == ("16_new_document.md",)
    with pytest.raises(CanonicalCorpusError, match="neither included nor excluded"):
        build_canonical_corpus(manifest, project_root=corpus_project, token_counter=TiktokenCounter())
    with pytest.raises(CanonicalCorpusError, match="declare them first"):
        refresh_canonical_corpus_manifest(
            corpus_project / DEFAULT_CORPUS_MANIFEST_RELATIVE,
            project_root=corpus_project,
            token_counter=TiktokenCounter(),
        )


@pytest.mark.parametrize(
    "field",
    ["chunk_count", "chunk_payload_digest", "corpus_fingerprint"],
)
def test_a_manifest_that_no_longer_matches_the_built_corpus_is_refused(corpus_project: Path, field: str) -> None:
    manifest_path, _ = _refresh_and_write(corpus_project)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["expected"][field] = payload["expected"][field] + 1 if field == "chunk_count" else "1" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CanonicalCorpusError, match=f"identity mismatch.*{field}"):
        build_canonical_corpus(_manifest_in(corpus_project), project_root=corpus_project, token_counter=TiktokenCounter())


def test_a_change_of_chunk_text_changes_both_digests(corpus_project: Path) -> None:
    """The deterministic identity really covers the chunk content and the model-bound fingerprint."""
    manifest_path, first = _refresh_and_write(corpus_project)
    target = corpus_project / "data" / "02_clean_markdown" / "09_response_style_and_templates.md"
    target.write_text(target.read_text(encoding="utf-8").replace("Спасибо за ваш вопрос.", "Здравствуйте."), encoding="utf-8")
    _, second = _refresh_and_write(corpus_project)
    assert second.identity.chunk_payload_digest != first.identity.chunk_payload_digest
    assert second.identity.corpus_fingerprint != first.identity.corpus_fingerprint
    assert second.identity.document_ids == first.identity.document_ids


def test_the_fingerprint_is_bound_to_the_embedding_model_name_the_payload_digest_is_not(
    corpus_project: Path,
) -> None:
    _, build = _refresh_and_write(corpus_project)
    other = compute_corpus_identity(build.chunks, embedding_model="some-other-model")
    assert other.chunk_payload_digest == build.identity.chunk_payload_digest
    assert other.corpus_fingerprint != build.identity.corpus_fingerprint


def test_other_embedding_model_is_refused(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    with pytest.raises(CanonicalCorpusError, match="pinned to embedding model"):
        build_canonical_corpus(
            _manifest_in(corpus_project),
            project_root=corpus_project,
            token_counter=TiktokenCounter(),
            embedding_model="some-other-model",
        )


def test_other_tokenizer_is_refused(corpus_project: Path) -> None:
    _refresh_and_write(corpus_project)
    with pytest.raises(CanonicalCorpusError, match="pinned to the 'cl100k_base' tokenizer"):
        build_canonical_corpus(
            _manifest_in(corpus_project),
            project_root=corpus_project,
            token_counter=FakeTokenCounter(),
        )
    # another encoding by name only: loading a second real vocabulary would need the network
    class OtherEncoding(FakeTokenCounter):
        encoding_name = "o200k_base"

    with pytest.raises(CanonicalCorpusError, match="pinned to the 'cl100k_base' tokenizer"):
        build_canonical_corpus(
            _manifest_in(corpus_project),
            project_root=corpus_project,
            token_counter=OtherEncoding(),
        )
