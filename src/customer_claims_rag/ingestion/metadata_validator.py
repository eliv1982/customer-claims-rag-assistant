"""Strict metadata validation against canonical schema."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from customer_claims_rag.exceptions import DuplicateDocumentIdError, MetadataValidationError
from customer_claims_rag.models import ALLOWED_METADATA_KEYS, DocumentMetadata


class MetadataValidator:
    """Validate YAML front matter before Pydantic parsing."""

    def validate_front_matter(
        self,
        data: dict[str, Any],
        *,
        filename_stem: str,
        file_path: str,
    ) -> DocumentMetadata:
        unknown = set(data.keys()) - ALLOWED_METADATA_KEYS
        if unknown:
            raise MetadataValidationError(
                f"unknown metadata fields: {sorted(unknown)}",
                file_path=file_path,
            )

        required = {
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
        }
        absent = required - set(data.keys())
        if absent:
            raise MetadataValidationError(
                f"missing required metadata fields: {sorted(absent)}",
                file_path=file_path,
            )

        document_id = data.get("document_id")
        if document_id != filename_stem:
            raise MetadataValidationError(
                f"document_id {document_id!r} does not match filename {filename_stem!r}",
                file_path=file_path,
            )

        try:
            return DocumentMetadata.model_validate(data)
        except ValidationError as exc:
            raise MetadataValidationError(
                f"metadata validation failed: {exc.errors()[0]['msg']}",
                file_path=file_path,
            ) from exc

    def check_duplicate_document_ids(
        self, records: list[tuple[str, str]]
    ) -> None:
        """Raise if duplicate document_id values are present."""
        seen: dict[str, str] = {}
        for doc_id, path in records:
            if doc_id in seen:
                raise DuplicateDocumentIdError(
                    f"duplicate document_id {doc_id!r} (also in {seen[doc_id]})",
                    file_path=path,
                )
            seen[doc_id] = path
