"""Credential source of the canonical release build (``build-index --corpus-manifest``).

A canonical build embeds the whole release corpus with a paid provider, so its credential must be
one somebody put in the process environment on purpose, never one the tool found in a repository
``.env``. Ordinary development builds keep reading ``.env``.

Nothing here calls OpenAI: the provider class behind the factory is replaced by a recorder, and
the refusal cases assert that no provider is ever created.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.cli import build_chunks as chunks_cli
from customer_claims_rag.cli import build_index as index_cli
from customer_claims_rag.ingestion.canonical_corpus import DEFAULT_CORPUS_MANIFEST_RELATIVE
from customer_claims_rag.retrieval import factory as retrieval_factory
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ARG = DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix()

PROCESS_KEY = "sk-test-process-environment-key-0001"
DOTENV_KEY = "sk-test-repository-dotenv-key-0002"

CANONICAL_BUILD = [
    "--corpus-manifest", MANIFEST_ARG,
    "--index-dir", "data/04_index_production",
    "--collection", "customer_claims",
    "--embedding-model", "text-embedding-3-small",
    "--rebuild",
]  # fmt: skip


@pytest.fixture
def project(temp_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clone-like root with the committed corpus manifest, refreshed for the default lane."""
    target = temp_project / DEFAULT_CORPUS_MANIFEST_RELATIVE
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE, target)
    monkeypatch.chdir(temp_project)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    # The stand-in tokenizer of the default lane cannot reproduce the committed (cl100k_base)
    # identity; the supported refresh path rewrites it for this sandbox.
    assert chunks_cli.main(["--corpus-manifest", MANIFEST_ARG, "--refresh-manifest"]) == 0
    # A real load_dotenv() runs for this root (the suite only neutralises the repository's .env),
    # and it writes os.environ directly: register the variable so the test's end unsets it again.
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(env_bootstrap, "_loaded", False)
    return temp_project


@pytest.fixture
def provider_keys(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Records the key every embedding provider was created with; no network, no OpenAI."""
    created: list[str | None] = []

    def provider(*, model_name: str, api_key: str | None = None):
        created.append(api_key)
        return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)

    monkeypatch.setattr(retrieval_factory, "OpenAIEmbeddingProvider", provider)
    return created


def _write_dotenv(project: Path) -> None:
    (project / ".env").write_text(f"OPENAI_API_KEY={DOTENV_KEY}\n", encoding="utf-8")


def _assert_no_secret_printed(capsys) -> str:
    captured = capsys.readouterr()
    output = captured.out + captured.err
    for secret in (PROCESS_KEY, DOTENV_KEY):
        assert secret not in output
        assert secret[:12] not in output  # no partial key either
    return output


# --- the resolution rule ----------------------------------------------------------------------


def test_a_process_environment_key_is_reported_by_source_and_never_by_value() -> None:
    credential = index_cli.resolve_release_build_credential(PROCESS_KEY)
    assert credential.available and credential.api_key == PROCESS_KEY
    assert credential.source == "process_environment"
    assert PROCESS_KEY not in repr(credential) and PROCESS_KEY not in str(credential)


@pytest.mark.parametrize("absent", [None, "", "   ", "\n"])
def test_an_absent_or_blank_process_key_is_missing(absent: str | None) -> None:
    credential = index_cli.resolve_release_build_credential(absent)
    assert not credential.available and credential.api_key is None
    assert credential.source == "missing"


# --- the canonical build ----------------------------------------------------------------------


def test_canonical_build_without_any_key_stops_before_a_provider_exists(
    project: Path, provider_keys: list, capsys
) -> None:
    capsys.readouterr()
    assert index_cli.main(CANONICAL_BUILD) == 1
    output = _assert_no_secret_printed(capsys)
    assert "credential_source=missing" in output and "OPENAI_API_KEY" in output
    assert provider_keys == []
    assert not (project / "data" / "04_index_production").exists()


@pytest.mark.parametrize("process_value", [None, "", "  "])
def test_canonical_build_does_not_accept_a_key_only_a_repository_dotenv_provides(
    project: Path, provider_keys: list, monkeypatch: pytest.MonkeyPatch, capsys, process_value
) -> None:
    _write_dotenv(project)
    if process_value is not None:
        monkeypatch.setenv("OPENAI_API_KEY", process_value)
    capsys.readouterr()

    assert index_cli.main(CANONICAL_BUILD) == 1

    output = _assert_no_secret_printed(capsys)
    assert "credential_source=missing" in output
    assert ".env is not used as the credential source" in output
    assert provider_keys == []
    assert not (project / "data" / "04_index_production").exists()


def test_canonical_build_uses_the_process_environment_key(
    project: Path, provider_keys: list, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PROCESS_KEY)
    capsys.readouterr()

    assert index_cli.main(CANONICAL_BUILD) == 0

    assert provider_keys == [PROCESS_KEY]
    output = _assert_no_secret_printed(capsys)
    assert "credential_source=process_environment" in output
    assert "Status: success" in output


def test_the_provider_receives_the_resolved_credential_not_the_raw_environment_value(
    project: Path, provider_keys: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", f"  {PROCESS_KEY}\r\n")  # a pasted secret's trailing newline

    assert index_cli.main(CANONICAL_BUILD) == 0

    assert provider_keys == [PROCESS_KEY]


def test_a_repository_dotenv_never_displaces_the_process_environment_key(
    project: Path, provider_keys: list, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _write_dotenv(project)
    monkeypatch.setenv("OPENAI_API_KEY", PROCESS_KEY)
    capsys.readouterr()

    assert index_cli.main(CANONICAL_BUILD) == 0

    assert provider_keys == [PROCESS_KEY]
    assert "credential_source=process_environment" in _assert_no_secret_printed(capsys)


def test_dry_run_needs_no_credential_even_with_a_dotenv_key(
    project: Path, provider_keys: list, capsys
) -> None:
    _write_dotenv(project)
    capsys.readouterr()

    code = index_cli.main(
        ["--corpus-manifest", MANIFEST_ARG, "--index-dir", "data/04_index_production", "--dry-run"]
    )

    assert code == 0 and provider_keys == []
    assert "credential_source" not in _assert_no_secret_printed(capsys)


# --- development builds keep their .env convenience --------------------------------------------


def test_a_development_build_still_reads_the_repository_dotenv(
    project: Path, provider_keys: list, capsys
) -> None:
    _write_dotenv(project)
    capsys.readouterr()

    code = index_cli.main(
        [
            "--input-dir", "data/02_clean_markdown",
            "--index-dir", "data/04_index_dev",
            "--embedding-model", "text-embedding-3-small",
            "--rebuild",
        ]  # fmt: skip
    )

    assert code == 0
    assert provider_keys == [DOTENV_KEY]
    assert "credential_source" not in capsys.readouterr().out
