"""Environment bootstrap and .env.example tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.retrieval_config import RetrievalSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

DOCUMENTED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "RAG_INDEX_DIR",
    "RAG_COLLECTION_NAME",
    "RAG_TOP_K",
    "RAG_FETCH_K",
    "RAG_SIMILARITY_THRESHOLD",
    "RAG_EMBEDDING_BATCH_SIZE",
}


@pytest.fixture(autouse=True)
def reset_env_bootstrap() -> None:
    env_bootstrap.reset_project_env()
    yield
    env_bootstrap.reset_project_env()


def test_env_example_contains_documented_settings_without_secrets() -> None:
    assert ENV_EXAMPLE.is_file()
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    keys = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        keys.add(key)
        assert "sk-" not in value
        assert "secret" not in value.lower()
    assert DOCUMENTED_ENV_KEYS.issubset(keys)


def test_dotenv_loaded_when_file_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    (tmp_path / ".env").write_text("RAG_TOP_K=7\n", encoding="utf-8")

    env_bootstrap.load_project_env(force=True)

    assert os.environ.get("RAG_TOP_K") == "7"
    settings = RetrievalSettings.from_env()
    assert settings.top_k == 7


def test_process_environment_has_priority_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_TOP_K", "9")
    (tmp_path / ".env").write_text("RAG_TOP_K=7\n", encoding="utf-8")

    env_bootstrap.load_project_env(force=True)

    assert os.environ.get("RAG_TOP_K") == "9"
    settings = RetrievalSettings.from_env()
    assert settings.top_k == 9


def test_missing_dotenv_is_allowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RAG_TOP_K", raising=False)

    env_bootstrap.load_project_env(force=True)
    settings = RetrievalSettings.from_env()

    assert settings.top_k == 4


def test_env_override_has_priority_over_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RAG_SIMILARITY_THRESHOLD", raising=False)
    (tmp_path / ".env").write_text("RAG_SIMILARITY_THRESHOLD=0.55\n", encoding="utf-8")

    env_bootstrap.load_project_env(force=True)
    settings = RetrievalSettings.from_env()

    assert settings.similarity_threshold == 0.55


def test_fake_provider_build_does_not_require_env_or_api_key(
    temp_project: Path,
    monkeypatch,
) -> None:
    from customer_claims_rag.cli import build_index as cli_module
    from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    monkeypatch.chdir(temp_project)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = cli_module.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=16,
        rebuild=True,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=lambda **kwargs: FakeEmbeddingProvider(
            model_name=kwargs["model_name"],
            vector_dimension=8,
        ),
        vector_store_factory=lambda **kwargs: ChromaVectorStore(
            index_dir=kwargs["index_dir"],
            collection_name=kwargs["collection_name"],
        ),
        settings=RetrievalSettings.from_env(),
    )
    assert code == 0
