"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_MARKDOWN_DIR = PROJECT_ROOT / "data" / "02_clean_markdown"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def clean_markdown_dir() -> Path:
    return CLEAN_MARKDOWN_DIR


@pytest.fixture
def temp_project(tmp_path: Path, clean_markdown_dir: Path) -> Path:
    """Isolated project root with a copy of clean Markdown corpus."""
    data_dir = tmp_path / "data"
    clean_dest = data_dir / "02_clean_markdown"
    clean_dest.mkdir(parents=True)
    for source in sorted(clean_markdown_dir.glob("*.md")):
        shutil.copy(source, clean_dest / source.name)
    (data_dir / "03_chunks").mkdir()
    (data_dir / "04_index").mkdir()
    return tmp_path


@pytest.fixture
def builder(temp_project: Path) -> "CorpusBuilder":
    from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
    from customer_claims_rag.token_counter import TiktokenCounter

    return CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=temp_project.resolve(),
    )


@pytest.fixture
def fake_embedding_provider():
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider

    return FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)


@pytest.fixture
def sample_metadata() -> dict:
    return {
        "document_id": "99_test_doc",
        "title": "Тестовый документ",
        "category": "general",
        "document_type": "reference",
        "version": "1.0.0",
        "status": "active",
        "effective_date": "2026-06-01",
        "last_updated": "2026-06-20",
        "audience": "support",
        "confidentiality": "internal",
        "source_type": "internal_reference",
        "language": "ru",
        "priority": "medium",
    }
