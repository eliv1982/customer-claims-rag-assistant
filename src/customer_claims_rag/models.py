"""Pydantic data models for documents and chunks."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocumentType = Literal["policy", "procedure", "reference", "faq", "templates", "guideline"]
DocumentStatus = Literal["draft", "active", "superseded", "archived"]
Confidentiality = Literal["public", "internal", "restricted"]
DocumentPriority = Literal["critical", "high", "medium", "low"]
SourceType = Literal[
    "internal_policy",
    "internal_reference",
    "internal_procedure",
    "internal_guideline",
    "internal_faq",
    "sop",
    "faq",
    "brand_guide",
]
Language = Literal["ru"]
ChunkRiskLevel = Literal["low", "medium", "high", "critical"]

ALLOWED_METADATA_KEYS = frozenset(
    {
        "document_id",
        "title",
        "category",
        "document_type",
        "version",
        "status",
        "effective_date",
        "last_updated",
        "audience",
        "confidentiality",
        "source_type",
        "language",
        "priority",
        "related_documents",
        "supersedes",
    }
)


class DocumentMetadata(BaseModel):
    """Canonical document metadata from YAML front matter."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str
    title: str
    category: str
    document_type: DocumentType
    version: str
    status: DocumentStatus
    effective_date: date
    last_updated: date
    audience: str | list[str]
    confidentiality: Confidentiality
    source_type: SourceType
    language: Language
    priority: DocumentPriority
    related_documents: list[str] | None = None
    supersedes: str | None = None

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or " " in value:
            raise ValueError("document_id must be non-empty snake_case identifier")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("version must be semver format, e.g. 1.0.0")
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> DocumentMetadata:
        if self.last_updated < self.effective_date:
            raise ValueError("last_updated must not be earlier than effective_date")
        return self


class DocumentRecord(BaseModel):
    """Loaded Markdown document with validated metadata."""

    metadata: DocumentMetadata
    source_path: str
    content: str
    word_count: int
    char_count: int


class HeadingBlock(BaseModel):
    """Semantic block bounded by Markdown headings."""

    level: int
    heading: str
    heading_path: list[str]
    content: str
    order: int


class ChunkMetadata(BaseModel):
    """Inherited document metadata plus optional chunk-level fields."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    category: str
    document_type: DocumentType
    version: str
    status: DocumentStatus
    effective_date: date
    last_updated: date
    audience: str | list[str]
    confidentiality: Confidentiality
    source_type: SourceType
    language: Language
    priority: DocumentPriority
    related_documents: list[str] | None = None
    supersedes: str | None = None
    section: str | None = None
    subsection: str | None = None
    topic: str | None = None
    risk_level: ChunkRiskLevel | None = None
    keywords: list[str] | None = None
    source_file: str | None = None

    @classmethod
    def from_document(
        cls,
        metadata: DocumentMetadata,
        *,
        source_path: str,
        section: str | None = None,
        subsection: str | None = None,
        topic: str | None = None,
        risk_level: ChunkRiskLevel | None = None,
    ) -> ChunkMetadata:
        return cls(
            document_id=metadata.document_id,
            title=metadata.title,
            category=metadata.category,
            document_type=metadata.document_type,
            version=metadata.version,
            status=metadata.status,
            effective_date=metadata.effective_date,
            last_updated=metadata.last_updated,
            audience=metadata.audience,
            confidentiality=metadata.confidentiality,
            source_type=metadata.source_type,
            language=metadata.language,
            priority=metadata.priority,
            related_documents=metadata.related_documents,
            supersedes=metadata.supersedes,
            section=section,
            subsection=subsection,
            topic=topic,
            risk_level=risk_level,
            source_file=source_path,
        )


class ChunkRecord(BaseModel):
    """Deterministic chunk ready for JSONL export."""

    chunk_id: str
    document_id: str
    source_path: str
    title: str
    category: str
    document_type: DocumentType
    source_type: SourceType
    document_priority: DocumentPriority
    chunk_index: int
    heading: str
    heading_path: list[str]
    content: str
    token_count: int
    word_count: int
    char_count: int
    strategy: str
    overlap_tokens: int
    metadata: ChunkMetadata


class CorpusStats(BaseModel):
    """Aggregated statistics for generated chunk corpus."""

    documents_total: int
    chunks_total: int
    chunks_by_document: dict[str, int]
    chunks_by_document_type: dict[str, int]
    strategy_counts: dict[str, int]
    min_tokens: int
    max_tokens: int
    average_tokens: float
    median_tokens: float
    chunks_below_soft_min: int
    chunks_above_soft_max: int
    chunks_at_or_above_hard_max: int
    overlap_chunks: int
    faq_chunk_count: int
    template_chunk_count: int
