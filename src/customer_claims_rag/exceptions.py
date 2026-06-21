"""Ingestion and chunking exceptions."""


class IngestionError(Exception):
    """Base error for ingestion pipeline."""

    def __init__(self, message: str, *, file_path: str | None = None) -> None:
        self.file_path = file_path
        if file_path:
            super().__init__(f"{file_path}: {message}")
        else:
            super().__init__(message)


class FrontMatterError(IngestionError):
    """YAML front matter parsing or structure error."""


class MetadataValidationError(IngestionError):
    """Document metadata failed validation."""


class MarkdownStructureError(IngestionError):
    """Markdown body structure is invalid for parsing."""


class ChunkingError(IngestionError):
    """Chunking rule violation or failure."""


class DuplicateDocumentIdError(IngestionError):
    """Duplicate document_id across corpus."""


class DuplicateChunkIdError(IngestionError):
    """Duplicate chunk_id in generated corpus."""
