"""Data models for grounded generation contract layer."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CITATION_KEY_PATTERN = re.compile(r"^S[1-9][0-9]*$")
ResponseMode = Literal["grounded_answer", "insufficient_context"]


class ContextItem(BaseModel):
    """Single retrieved context block exposed to the generation layer."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    rank: int
    document_id: str
    chunk_id: str
    heading: str
    source_path: str
    content: str

    @field_validator("citation_key")
    @classmethod
    def validate_citation_key(cls, value: str) -> str:
        if not _CITATION_KEY_PATTERN.fullmatch(value):
            raise ValueError("citation_key must match S1, S2, ...")
        return value

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rank must be >= 1")
        return value

    @field_validator(
        "document_id",
        "chunk_id",
        "heading",
        "source_path",
        "content",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value


class ContextPackage(BaseModel):
    """Package of context blocks passed to grounded generation."""

    model_config = ConfigDict(extra="forbid")

    items: list[ContextItem] = Field(default_factory=list)


class GroundedGenerationRequest(BaseModel):
    """Production input for grounded answer generation."""

    model_config = ConfigDict(extra="forbid")

    customer_query: str
    context_package: ContextPackage

    @field_validator("customer_query")
    @classmethod
    def validate_customer_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_query must not be empty or whitespace-only")
        return value


class RawGenerationDraft(BaseModel):
    """Parsed model output before structural validation."""

    model_config = ConfigDict(extra="forbid")

    response_mode: ResponseMode
    answer: str


class Citation(BaseModel):
    """Application-owned citation metadata derived from context items."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    document_id: str
    chunk_id: str
    heading: str
    source_path: str

    @field_validator("citation_key")
    @classmethod
    def validate_citation_key(cls, value: str) -> str:
        if not _CITATION_KEY_PATTERN.fullmatch(value):
            raise ValueError("citation_key must match S1, S2, ...")
        return value

    @field_validator("document_id", "chunk_id", "heading", "source_path")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value


class GroundedGenerationResult(BaseModel):
    """Client-safe grounded generation output."""

    model_config = ConfigDict(extra="forbid")

    response_mode: ResponseMode
    customer_response: str
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response_mode_citations(self) -> Self:
        if self.response_mode == "grounded_answer":
            if not self.citations:
                raise ValueError("grounded_answer requires at least one citation")
        elif self.response_mode == "insufficient_context":
            if self.citations:
                raise ValueError("insufficient_context requires empty citations")
        return self
