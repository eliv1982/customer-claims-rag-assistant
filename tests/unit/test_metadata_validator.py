"""Unit tests for metadata validation."""

from __future__ import annotations

from datetime import date

import pytest

from customer_claims_rag.exceptions import DuplicateDocumentIdError, MetadataValidationError
from customer_claims_rag.ingestion.metadata_validator import MetadataValidator
from customer_claims_rag.models import DocumentMetadata


@pytest.fixture
def validator() -> MetadataValidator:
    return MetadataValidator()


def test_valid_metadata(validator: MetadataValidator, sample_metadata: dict) -> None:
    meta = validator.validate_front_matter(
        sample_metadata,
        filename_stem="99_test_doc",
        file_path="test.md",
    )
    assert isinstance(meta, DocumentMetadata)
    assert meta.document_id == "99_test_doc"


def test_invalid_semver(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["version"] = "1.0"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_malformed_related_documents(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["related_documents"] = "not-a-list"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_unknown_status(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["status"] = "deleted"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_unknown_priority(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["priority"] = "urgent"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_unknown_source_type(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["source_type"] = "external"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_unknown_document_type(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["document_type"] = "memo"
    with pytest.raises(MetadataValidationError, match="validation failed"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_yaml_date_parsed(validator: MetadataValidator, sample_metadata: dict) -> None:
    meta = validator.validate_front_matter(
        sample_metadata,
        filename_stem="99_test_doc",
        file_path="t.md",
    )
    assert meta.effective_date == date(2026, 6, 1)


def test_missing_required_field(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    del data["title"]
    with pytest.raises(MetadataValidationError, match="missing required"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")


def test_filename_mismatch(validator: MetadataValidator, sample_metadata: dict) -> None:
    with pytest.raises(MetadataValidationError, match="does not match filename"):
        validator.validate_front_matter(
            sample_metadata,
            filename_stem="wrong_name",
            file_path="t.md",
        )


def test_category_accepts_extension_values(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["category"] = "unknown_cat"
    meta = validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")
    assert meta.category == "unknown_cat"


def test_duplicate_document_id(validator: MetadataValidator) -> None:
    with pytest.raises(DuplicateDocumentIdError, match="duplicate document_id"):
        validator.check_duplicate_document_ids(
            [("doc_a", "a.md"), ("doc_a", "b.md")]
        )


def test_unknown_metadata_field(validator: MetadataValidator, sample_metadata: dict) -> None:
    data = dict(sample_metadata)
    data["extra_field"] = "value"
    with pytest.raises(MetadataValidationError, match="unknown metadata fields"):
        validator.validate_front_matter(data, filename_stem="99_test_doc", file_path="t.md")
