"""The release documentation states what the committed configuration says, and nothing retired.

A release table that drifts from ``configs/`` or an instruction that names a command the tools no
longer accept is worse than no documentation: the provisioning instructions are the one thing an
operator follows literally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.release.posture import build_instruction, resolve_production_release_posture

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

RETIRED_PHRASES = (
    "instructor",
    "secure transfer",
    "approved external archive",
    "approved archive",
    "RAG_RELEASE_TARGET=rollback",
    "RAG_RELEASE_TARGET = \"rollback\"",
    "emergency_rollback",
    "no rebuild during provisioning",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def target():
    return resolve_production_release_posture(project_root=ROOT).resolved_target


def test_release_posture_doc_states_the_committed_identity(target) -> None:
    text = _read(DOCS / "06_release_posture.md")
    corpus = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    descriptor = resolve_production_release_posture(project_root=ROOT).descriptor
    for value in (
        descriptor.release_posture_id,
        corpus.corpus_id,
        target.index_path_relative,
        corpus.expected.chunk_payload_digest,
        corpus.expected.corpus_fingerprint,
        descriptor.expected_frozen_config_hash,
        target.embedding_model,
        str(target.vector_dimension),
        target.collection_name,
        DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix(),
        f"`{corpus.expected.chunk_count}` / `{corpus.expected.document_count}`",
    ):
        assert value in text, f"docs/06_release_posture.md does not state {value!r}"


def test_release_posture_doc_lists_every_document_as_included_or_excluded() -> None:
    text = _read(DOCS / "06_release_posture.md")
    corpus = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    for document in corpus.documents:
        assert f"| {document.document_id[:2]} |" in text and "| yes |" in text
    for document in corpus.excluded_documents:
        assert f"| {document.document_id[:2]} |" in text
    assert text.count("| yes |") == len(corpus.documents)
    assert text.count("| no, excluded |") == len(corpus.excluded_documents)


def test_provisioning_doc_contains_the_exact_build_command_the_validator_prints(target) -> None:
    text = _read(DOCS / "07_index_provisioning.md")
    assert build_instruction(target) in text


def test_provisioning_doc_has_a_dry_run_command_the_build_tool_accepts(target) -> None:
    from customer_claims_rag.cli import build_index

    text = _read(DOCS / "07_index_provisioning.md")
    dry_run = build_instruction(target).replace("--rebuild", "--dry-run")
    assert dry_run in text
    words = dry_run.split()[3:]
    args = build_index.build_parser().parse_args(words)
    assert args.dry_run is True and args.corpus_manifest == Path(target.corpus_manifest_relative)


@pytest.mark.parametrize("doc", ["06_release_posture.md", "07_index_provisioning.md", "08_docker_runbook.md"])
def test_release_docs_contain_no_retired_provisioning_instructions(doc: str) -> None:
    text = _read(DOCS / doc).lower()
    for phrase in RETIRED_PHRASES:
        assert phrase.lower() not in text, f"{doc} still says {phrase!r}"


@pytest.mark.parametrize("doc", ["07_index_provisioning.md", "08_docker_runbook.md"])
def test_operator_docs_never_point_at_the_superseded_backup_directory(doc: str) -> None:
    assert "04_index_backup" not in _read(DOCS / doc)


def test_readme_and_env_example_name_the_current_index_not_the_backup() -> None:
    for path in (ROOT / "README.md", ROOT / ".env.example"):
        text = _read(path)
        assert "04_index_backup" not in text, path.name
        assert "RAG_RELEASE_TARGET=rollback" not in text, path.name
    assert "data/04_index_production" in _read(ROOT / "README.md")


def test_release_posture_doc_marks_the_previous_release_as_superseded_history() -> None:
    text = _read(DOCS / "06_release_posture.md")
    assert "foodflow-10doc-release-v1" in text and "superseded" in text
    assert "experiments/corpus/historical_pre_2d2_15doc_v1" in text


def test_release_posture_doc_states_the_exit_status_of_each_verdict() -> None:
    from customer_claims_rag.cli import validate_release_posture as cli

    text = _read(DOCS / "06_release_posture.md")
    for code, verdict in (
        (cli.EXIT_RELEASE_READY, "yes"),
        (cli.EXIT_RELEASE_BLOCKED, "no"),
        (cli.EXIT_READINESS_NOT_ESTABLISHED, "not_established"),
    ):
        assert f"| `{code}` | `{verdict}` |" in text, f"docs/06 does not tie exit {code} to {verdict}"
    assert "never exits `0`" in text and "static_validation" in text
    assert "usage error" in text  # exit 2 stays argparse's


def test_provisioning_doc_says_that_skipping_the_store_is_not_a_release_gate() -> None:
    from customer_claims_rag.cli import validate_release_posture as cli

    text = _read(DOCS / "07_index_provisioning.md")
    assert f"exits `{cli.EXIT_READINESS_NOT_ESTABLISHED}`" in text
    assert "not a release gate" in text


def test_no_documentation_presents_skip_vector_store_as_a_passing_check() -> None:
    """Every line that mentions the flag also says it cannot establish readiness."""
    for doc in ("06_release_posture.md", "07_index_provisioning.md", "08_docker_runbook.md"):
        for line in _read(DOCS / doc).splitlines():
            if "--skip-vector-store" in line:
                assert any(
                    marker in line for marker in ("`3`", "not_established", "never exits `0`", "not a release gate")
                ), f"{doc}: {line!r}"


def test_no_operational_script_uses_skip_vector_store_as_its_gate() -> None:
    """The container preflight and the delivery files run the full validation or none at all."""
    for path in (
        ROOT / "scripts" / "docker-entrypoint.sh",
        ROOT / "compose.yaml",
        ROOT / "Dockerfile",
        ROOT / "README.md",
    ):
        assert "--skip-vector-store" not in _read(path), path.name


RELEASE_PATH_SOURCES = (
    *sorted((ROOT / "src" / "customer_claims_rag").glob("application/*.py")),
    *sorted((ROOT / "src" / "customer_claims_rag").glob("release/*.py")),
    *sorted((ROOT / "src" / "customer_claims_rag").glob("ui/*.py")),
    *sorted((ROOT / "src" / "customer_claims_rag").glob("generation/*.py")),
    *sorted((ROOT / "src" / "customer_claims_rag").glob("retrieval/*.py")),
    *sorted((ROOT / "src" / "customer_claims_rag").glob("ingestion/*.py")),
    *(
        ROOT / "src" / "customer_claims_rag" / "cli" / f"{name}.py"
        for name in ("answer_claim", "build_chunks", "build_index", "validate_release_posture")
    ),
    *sorted((ROOT / "configs" / "release").glob("*.json")),
    *sorted((ROOT / "configs" / "corpus").glob("*.json")),
    ROOT / "compose.yaml",
    ROOT / "Dockerfile",
    ROOT / "scripts" / "docker-entrypoint.sh",
)


def test_the_release_path_never_depends_on_a_backup_directory_or_a_rollback_copy() -> None:
    """Build, validate and serve touch the canonical corpus and the production index only."""
    assert len(RELEASE_PATH_SOURCES) > 40
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in RELEASE_PATH_SOURCES
        if any(word in path.read_text(encoding="utf-8").lower() for word in ("04_index_backup", "rollback"))
    ]
    assert offenders == []
