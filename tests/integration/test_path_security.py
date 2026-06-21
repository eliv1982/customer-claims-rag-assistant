"""Tests for export path security."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from customer_claims_rag.exceptions import IngestionError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.token_counter import TiktokenCounter


@pytest.fixture
def mini_project(tmp_path: Path, sample_metadata: dict) -> Path:
    data = tmp_path / "data"
    clean = data / "02_clean_markdown"
    clean.mkdir(parents=True)
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    (clean / "99_test_doc.md").write_text(
        f"---\n{fm_lines}\n---\n\n## Section\n\nParagraph one.\n",
        encoding="utf-8",
    )
    (data / "03_chunks").mkdir()
    return tmp_path


def _builder(root: Path) -> CorpusBuilder:
    return CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=root.resolve())


def test_output_outside_root_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    outside = mini_project.parent / "outside.jsonl"
    with pytest.raises(IngestionError, match="outside permitted root"):
        builder.build_and_export(
            clean,
            outside,
            mini_project / "data" / "03_chunks" / "stats.json",
        )


def test_stats_outside_root_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    outside_stats = mini_project.parent / "stats.json"
    with pytest.raises(IngestionError, match="outside permitted root"):
        builder.build_and_export(
            clean,
            mini_project / "data" / "03_chunks" / "chunks.jsonl",
            outside_stats,
        )


def test_output_equals_source_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    source = clean / "99_test_doc.md"
    before = source.read_bytes()
    with pytest.raises(IngestionError, match="overwrite"):
        builder.build_and_export(
            clean,
            source,
            mini_project / "data" / "03_chunks" / "stats.json",
        )
    assert source.read_bytes() == before


def test_output_inside_clean_dir_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    with pytest.raises(IngestionError, match="clean Markdown"):
        builder.build_and_export(
            clean,
            clean / "chunks.jsonl",
            mini_project / "data" / "03_chunks" / "stats.json",
        )


def test_output_equals_stats_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    same = mini_project / "data" / "03_chunks" / "same.json"
    with pytest.raises(IngestionError, match="same file"):
        builder.build_and_export(clean, same, same)


def test_output_directory_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out_dir = mini_project / "data" / "03_chunks"
    with pytest.raises(IngestionError, match="directory"):
        builder.build_and_export(
            clean,
            out_dir,
            out_dir / "stats.json",
        )


def test_stats_directory_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats_dir = mini_project / "data" / "03_chunks" / "stats_dir"
    stats_dir.mkdir()
    with pytest.raises(IngestionError, match="directory"):
        builder.build_and_export(clean, out, stats_dir)
    assert not out.exists()


def test_stats_inside_clean_dir_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = clean / "chunk_stats.json"
    before = {p.name: p.read_bytes() for p in clean.glob("*.md")}
    with pytest.raises(IngestionError, match="clean Markdown"):
        builder.build_and_export(clean, out, stats)
    assert not stats.exists()
    after = {p.name: p.read_bytes() for p in clean.glob("*.md")}
    assert before == after
    assert not out.exists()


def test_output_inside_custom_input_dir_rejected(mini_project: Path, sample_metadata: dict) -> None:
    custom_input = mini_project / "custom_corpus"
    custom_input.mkdir()
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    (custom_input / "99_test_doc.md").write_text(
        f"---\n{fm_lines}\n---\n\n## Section\n\nParagraph one.\n",
        encoding="utf-8",
    )
    builder = _builder(mini_project)
    out = custom_input / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"
    with pytest.raises(IngestionError, match="inside input directory"):
        builder.build_and_export(custom_input, out, stats)
    assert not out.exists()
    assert not stats.exists()


def test_stats_inside_custom_input_dir_rejected(mini_project: Path, sample_metadata: dict) -> None:
    custom_input = mini_project / "custom_corpus"
    custom_input.mkdir()
    fm_lines = "\n".join(f"{k}: {v}" for k, v in sample_metadata.items())
    (custom_input / "99_test_doc.md").write_text(
        f"---\n{fm_lines}\n---\n\n## Section\n\nParagraph one.\n",
        encoding="utf-8",
    )
    builder = _builder(mini_project)
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = custom_input / "chunk_stats.json"
    with pytest.raises(IngestionError, match="inside input directory"):
        builder.build_and_export(custom_input, out, stats)
    assert not out.exists()
    assert not stats.exists()


def test_stats_traversal_outside_root_rejected(mini_project: Path) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    outside_stats = mini_project / "data" / "03_chunks" / ".." / ".." / ".." / "outside_stats.json"
    with pytest.raises(IngestionError, match="outside permitted root"):
        builder.build_and_export(clean, out, outside_stats)
    assert not out.exists()


def test_documented_dual_export_pair_not_atomic_on_stats_replace_failure(
    mini_project: Path, monkeypatch
) -> None:
    """JSONL+stats pair is not one transaction; new JSONL may persist if stats replace fails."""
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"

    builder.build_and_export(clean, out, stats)
    stats.write_text('{"marker": "old-stats"}\n', encoding="utf-8")
    old_stats = stats.read_bytes()

    real_replace = Path.replace
    replace_calls = 0

    def patched_replace(self: Path, target: Path) -> Path:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated stats replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", patched_replace)

    with pytest.raises(OSError, match="simulated stats replace failure"):
        builder.build_and_export(clean, out, stats)

    assert replace_calls == 2
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert lines
    for line in lines:
        json.loads(line)
    assert stats.read_bytes() == old_stats
    chunks_dir = mini_project / "data" / "03_chunks"
    assert not list(chunks_dir.glob("*.tmp"))


def test_existing_output_not_corrupted_on_failure(mini_project: Path, monkeypatch) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"
    builder.build_and_export(clean, out, stats)
    good = out.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(builder, "_write_stats_temp", fail_write)
    with pytest.raises(OSError):
        builder.build_and_export(clean, out, stats)
    assert out.read_bytes() == good
    assert not list((mini_project / "data" / "03_chunks").glob("*.tmp"))


def test_jsonl_temp_write_failure_leaves_no_partial(mini_project: Path, monkeypatch) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"
    chunks_dir = mini_project / "data" / "03_chunks"
    before = {p.name for p in chunks_dir.iterdir()}

    def fail_jsonl(*args, **kwargs):
        raise OSError("simulated jsonl failure")

    monkeypatch.setattr(builder, "_write_jsonl_temp", fail_jsonl)
    with pytest.raises(OSError):
        builder.build_and_export(clean, out, stats)
    after = {p.name for p in chunks_dir.iterdir()}
    assert after == before


def test_stats_temp_write_failure_leaves_no_partial(mini_project: Path, monkeypatch) -> None:
    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = mini_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"
    chunks_dir = mini_project / "data" / "03_chunks"

    def fail_stats(*args, **kwargs):
        raise OSError("simulated stats failure")

    monkeypatch.setattr(builder, "_write_stats_temp", fail_stats)
    with pytest.raises(OSError):
        builder.build_and_export(clean, out, stats)
    assert not list(chunks_dir.glob("*.tmp"))
    assert not stats.exists()


def test_symlink_escape_outside_root_rejected(mini_project: Path) -> None:
    outside = mini_project.parent / "outside_escape"
    outside.mkdir(exist_ok=True)
    link = mini_project / "escape_link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation not available in this environment: {exc}")

    builder = _builder(mini_project)
    clean = mini_project / "data" / "02_clean_markdown"
    out = link / "chunks.jsonl"
    stats = mini_project / "data" / "03_chunks" / "stats.json"
    with pytest.raises(IngestionError, match="outside permitted root"):
        builder.build_and_export(clean, out, stats)
    assert not out.exists()
    assert not stats.exists()
