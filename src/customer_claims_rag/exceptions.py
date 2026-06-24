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


class RetrievalError(Exception):
    """Base error for retrieval and indexing pipeline."""


class EmbeddingError(RetrievalError):
    """Embedding provider validation or API failure."""


class VectorStoreError(RetrievalError):
    """Vector store operation failure."""


class IndexBuildError(RetrievalError):
    """Index build validation or persistence failure."""


class IndexManifestError(RetrievalError):
    """Index manifest missing, corrupt, or inconsistent."""


class SearchError(RetrievalError):
    """Search query validation or execution failure."""


class EvaluationCorpusError(Exception):
    """Evaluation corpus parsing or validation failure."""


class EvaluationOutputError(RetrievalError):
    """Evaluation artifact write failure."""


class GenerationError(Exception):
    """Base error for grounded generation pipeline."""


class GenerationParseError(GenerationError):
    """Failed to parse model output into a generation draft."""


class GenerationValidationError(GenerationError):
    """Generation draft or context failed structural validation."""


class LLMCallError(GenerationError):
    """Chat model invocation failure."""


class RiskAssessmentError(Exception):
    """Base error for deterministic risk assessment pipeline."""


class RiskValidationError(RiskAssessmentError):
    """Deterministic risk result failed cross-field validation."""
