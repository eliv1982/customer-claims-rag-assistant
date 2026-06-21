"""CLI tests."""

from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path

import pytest

from customer_claims_rag.cli import build_chunks as cli_module


def _write_doc(clean_dir: Path, metadata: dict, body: str) -> None:
    fm_lines = "\n".join(f"{k}: {v}" for k, v in metadata.items())
    name = f"{metadata['document_id']}.md"
    (clean_dir / name).write_text(
        f"---\n{fm_lines}\n---\n\n{body}",
        encoding="utf-8",
    )


@pytest.fixture
def cli_project(tmp_path: Path, sample_metadata: dict) -> Path:
    data = tmp_path / "data"
    clean = data / "02_clean_markdown"
    clean.mkdir(parents=True)
    meta = dict(sample_metadata)
    _write_doc(clean, meta, "## Section\n\nContent paragraph.\n")
    (data / "03_chunks").mkdir()
    return tmp_path


def test_cli_success(cli_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/03_chunks/chunks.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
            "--verbose",
        ]
    )
    assert code == 0
    assert (cli_project / "data" / "03_chunks" / "chunks.jsonl").exists()


def test_cli_empty_corpus_nonzero(cli_project: Path, monkeypatch) -> None:
    shutil.rmtree(cli_project / "data" / "02_clean_markdown")
    (cli_project / "data" / "02_clean_markdown").mkdir()
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/03_chunks/chunks.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
        ]
    )
    assert code != 0
    assert not (cli_project / "data" / "03_chunks" / "chunks.jsonl").exists()


def test_cli_unsafe_output_nonzero(cli_project: Path, monkeypatch) -> None:
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/02_clean_markdown/out.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
        ]
    )
    assert code != 0


def test_cli_unsafe_output_no_traceback_without_verbose(
    cli_project: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/02_clean_markdown/out.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_invalid_encoding_nonzero(cli_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(cli_project)
    out = cli_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = cli_project / "data" / "03_chunks" / "chunk_stats.json"
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/03_chunks/chunks.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
            "--encoding",
            "nonexistent_encoding_xyz",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()
    assert not stats.exists()


def test_cli_verbose_shows_traceback(cli_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/02_clean_markdown/out.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" in captured.err


def test_cli_include_inactive(cli_project: Path, monkeypatch, sample_metadata: dict) -> None:
    clean = cli_project / "data" / "02_clean_markdown"
    meta = dict(sample_metadata)
    meta["status"] = "draft"
    _write_doc(clean, meta, "## Draft\n\nDraft body paragraph here.\n")
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/03_chunks/chunks.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
            "--include-inactive",
        ]
    )
    assert code == 0
    assert (cli_project / "data" / "03_chunks" / "chunks.jsonl").exists()


def test_cli_fail_on_soft_limit_nonzero(cli_project: Path, monkeypatch) -> None:
    monkeypatch.chdir(cli_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--output",
            "data/03_chunks/chunks.jsonl",
            "--stats-output",
            "data/03_chunks/chunk_stats.json",
            "--fail-on-soft-limit",
        ]
    )
    assert code != 0
    assert not (cli_project / "data" / "03_chunks" / "chunks.jsonl").exists()
