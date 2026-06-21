"""Markdown document loading from clean corpus."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from customer_claims_rag.exceptions import (
    FrontMatterError,
    IngestionError,
    MetadataValidationError,
)
from customer_claims_rag.ingestion.metadata_validator import MetadataValidator
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path, to_relative_posix_path
from customer_claims_rag.models import DocumentRecord

_FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def normalize_body(text: str) -> str:
    """Normalize line endings and trailing whitespace without altering Markdown."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def split_front_matter(raw: str, file_path: str) -> tuple[dict, str]:
    """Split YAML front matter and body."""
    match = _FRONT_MATTER_PATTERN.match(raw)
    if not match:
        raise FrontMatterError(
            "missing or invalid YAML front matter delimiters (---)",
            file_path=file_path,
        )
    yaml_text = match.group(1)
    body = raw[match.end() :]
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise FrontMatterError(
            f"malformed YAML in front matter: {exc}",
            file_path=file_path,
        ) from exc
    if not isinstance(parsed, dict):
        raise FrontMatterError(
            "front matter must be a YAML mapping",
            file_path=file_path,
        )
    return parsed, body


def count_words(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())


class MarkdownLoader:
    """Load and validate clean Markdown documents."""

    def __init__(
        self,
        *,
        include_inactive: bool = False,
        allowed_statuses: set[str] | None = None,
        validator: MetadataValidator | None = None,
        permitted_root: Path | None = None,
    ) -> None:
        self.include_inactive = include_inactive
        self.allowed_statuses = allowed_statuses or (
            {"draft", "active", "superseded", "archived"} if include_inactive else {"active"}
        )
        self.validator = validator or MetadataValidator()
        self.permitted_root = permitted_root

    def discover_files(self, input_dir: Path) -> list[Path]:
        """Discover *.md files in deterministic sorted order."""
        safe_dir = resolve_safe_path(input_dir, root=self.permitted_root)
        if not safe_dir.is_dir():
            raise IngestionError(
                f"input directory does not exist: {safe_dir}",
                file_path=str(safe_dir),
            )
        return sorted(safe_dir.glob("*.md"), key=lambda p: p.name)

    def load_file(self, file_path: Path) -> DocumentRecord | None:
        """Load a single Markdown file. Returns None if status is filtered out."""
        if self.permitted_root is None:
            raise IngestionError(
                "permitted_root is required for loading Markdown files",
                file_path=str(file_path),
            )

        safe_path = resolve_safe_path(file_path, root=self.permitted_root)
        if safe_path.suffix.lower() != ".md":
            raise IngestionError(
                "only .md files are supported",
                file_path=str(safe_path),
            )
        try:
            raw = safe_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(
                f"cannot read file: {exc}",
                file_path=str(safe_path),
            ) from exc

        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        relative_path = to_relative_posix_path(safe_path, self.permitted_root)
        front_matter, body = split_front_matter(raw, relative_path)
        body = normalize_body(body)
        if not body:
            raise FrontMatterError(
                "document body is empty after front matter",
                file_path=relative_path,
            )

        metadata = self.validator.validate_front_matter(
            front_matter,
            filename_stem=safe_path.stem,
            file_path=relative_path,
        )
        if metadata.status not in self.allowed_statuses:
            return None

        return DocumentRecord(
            metadata=metadata,
            source_path=relative_path,
            content=body,
            word_count=count_words(body),
            char_count=len(body),
        )

    def load_directory(self, input_dir: Path) -> list[DocumentRecord]:
        """Load all Markdown files, enforcing unique document_id."""
        safe_input = resolve_safe_path(input_dir, root=self.permitted_root)
        md_files = self.discover_files(safe_input)
        if not md_files:
            raise IngestionError(
                f"no Markdown documents found in {safe_input.as_posix()}",
                file_path=str(safe_input),
            )

        records: list[DocumentRecord] = []
        skipped_inactive = 0
        for file_path in md_files:
            record = self.load_file(file_path)
            if record is None:
                skipped_inactive += 1
                continue
            records.append(record)

        if not records:
            raise IngestionError(
                f"no active documents loaded from {safe_input.as_posix()} "
                f"({skipped_inactive} file(s) skipped by status filter)",
                file_path=str(safe_input),
            )

        self.validator.check_duplicate_document_ids(
            [(record.metadata.document_id, record.source_path) for record in records]
        )
        return records
