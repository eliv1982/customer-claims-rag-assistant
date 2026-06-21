"""Unit tests for Markdown loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.exceptions import DuplicateDocumentIdError, FrontMatterError, IngestionError, MetadataValidationError
from customer_claims_rag.ingestion.markdown_loader import MarkdownLoader
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path


def _write_md(tmp_path: Path, name: str, front_matter: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"---\n{front_matter}---\n\n{body}", encoding="utf-8")
    return path


@pytest.fixture
def loader(tmp_path: Path) -> MarkdownLoader:
    return MarkdownLoader(permitted_root=tmp_path.resolve())


def test_valid_front_matter(loader: MarkdownLoader, tmp_path: Path, sample_metadata: dict) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    _write_md(tmp_path, "99_test_doc.md", fm_lines + "\n", "## Раздел\n\nТекст раздела.")
    record = loader.load_file(tmp_path / "99_test_doc.md")
    assert record is not None
    assert record.metadata.document_id == "99_test_doc"
    assert record.source_path == "99_test_doc.md"
    assert "Текст раздела" in record.content


def test_relative_posix_source_path(loader: MarkdownLoader, tmp_path: Path, sample_metadata: dict) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    _write_md(
        tmp_path,
        "99_test_doc.md",
        fm_lines + "\n",
        "## X\n\nY.",
    )
    record = loader.load_file(tmp_path / "99_test_doc.md")
    assert record is not None
    assert not Path(record.source_path).is_absolute()
    assert "/" not in record.source_path or record.source_path.count(":") == 0


def test_source_path_stable_across_roots(tmp_path: Path, sample_metadata: dict) -> None:
    paths: list[str] = []
    for idx in range(2):
        root = tmp_path / f"root{idx}"
        clean = root / "data" / "02_clean_markdown"
        clean.mkdir(parents=True)
        fm = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items()) + "\n"
        (clean / "99_test_doc.md").write_text(f"---\n{fm}---\n\n## X\n\nY.", encoding="utf-8")
        loader = MarkdownLoader(permitted_root=root.resolve())
        record = loader.load_file(clean / "99_test_doc.md")
        assert record is not None
        paths.append(record.source_path)
    assert paths[0] == paths[1] == "data/02_clean_markdown/99_test_doc.md"


def test_missing_front_matter(loader: MarkdownLoader, tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(FrontMatterError, match="front matter"):
        loader.load_file(path)


def test_malformed_yaml(loader: MarkdownLoader, tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("---\n: invalid: yaml: [\n---\n\nbody", encoding="utf-8")
    with pytest.raises(FrontMatterError, match="malformed YAML"):
        loader.load_file(path)


def test_empty_body(loader: MarkdownLoader, tmp_path: Path, sample_metadata: dict) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    _write_md(tmp_path, "99_test_doc.md", fm_lines + "\n", "   ")
    with pytest.raises(FrontMatterError, match="empty"):
        loader.load_file(tmp_path / "99_test_doc.md")


def test_utf8_russian_text(loader: MarkdownLoader, tmp_path: Path, sample_metadata: dict) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    body = "## Раздел\n\nОбращение клиента о доставке еды."
    _write_md(tmp_path, "99_test_doc.md", fm_lines + "\n", body)
    record = loader.load_file(tmp_path / "99_test_doc.md")
    assert record is not None
    assert "Обращение клиента" in record.content


def test_deterministic_file_order(loader: MarkdownLoader, tmp_path: Path, sample_metadata: dict) -> None:
    for name in ("02_b.md", "01_a.md"):
        meta = dict(sample_metadata)
        meta["document_id"] = Path(name).stem
        fm = "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n"
        _write_md(tmp_path, name, fm, "## X\n\nY.")
    files = loader.discover_files(tmp_path)
    assert [f.name for f in files] == ["01_a.md", "02_b.md"]


def test_inactive_filtering_empty_corpus(tmp_path: Path, sample_metadata: dict) -> None:
    loader = MarkdownLoader(permitted_root=tmp_path.resolve())
    meta = dict(sample_metadata)
    meta["status"] = "draft"
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n"
    _write_md(tmp_path, "99_test_doc.md", fm, "## X\n\nY.")
    with pytest.raises(IngestionError, match="no active documents"):
        loader.load_directory(tmp_path)


def test_include_inactive(tmp_path: Path, sample_metadata: dict) -> None:
    loader = MarkdownLoader(include_inactive=True, permitted_root=tmp_path.resolve())
    meta = dict(sample_metadata)
    meta["status"] = "draft"
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n"
    path = _write_md(tmp_path, "99_test_doc.md", fm, "## X\n\nY.")
    records = loader.load_directory(tmp_path)
    assert len(records) == 1
    assert records[0].metadata.status == "draft"


def test_empty_directory_error(tmp_path: Path) -> None:
    loader = MarkdownLoader(permitted_root=tmp_path.resolve())
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngestionError, match="no Markdown documents"):
        loader.load_directory(empty)


def test_duplicate_document_id_via_validator(tmp_path: Path, sample_metadata: dict) -> None:
    loader = MarkdownLoader(permitted_root=tmp_path.resolve())
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items()) + "\n"
    path = _write_md(tmp_path, "99_test_doc.md", fm_lines, "## X\n\nY.")
    original_discover = loader.discover_files

    def discover_duplicated(input_dir: Path) -> list[Path]:
        files = original_discover(input_dir)
        return files + files

    loader.discover_files = discover_duplicated  # type: ignore[method-assign]
    with pytest.raises(DuplicateDocumentIdError, match="duplicate document_id"):
        loader.load_directory(tmp_path)


def test_raw_directory_not_read(tmp_path: Path) -> None:
    raw_like = tmp_path / "raw_sim"
    raw_like.mkdir()
    (raw_like / "file.txt").write_text("raw content", encoding="utf-8")
    loader = MarkdownLoader(permitted_root=tmp_path.resolve())
    assert loader.discover_files(raw_like) == []


def test_path_outside_permitted_root_rejected(tmp_path: Path) -> None:
    loader = MarkdownLoader(permitted_root=tmp_path.resolve())
    outside = tmp_path.parent / "outside.md"
    outside.write_text("---\ndocument_id: x\n---\n\nbody", encoding="utf-8")
    with pytest.raises(IngestionError, match="path traversal"):
        loader.load_file(outside)
