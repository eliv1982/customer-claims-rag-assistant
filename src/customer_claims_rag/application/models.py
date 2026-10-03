"""Application-layer request and result contracts."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS

QUERY_TOO_LONG_ERROR_TYPE = "customer_query_too_long"


def is_query_too_long_error(error: ValidationError) -> bool:
    """True when a request validation error is the oversized-message error."""
    return any(item["type"] == QUERY_TOO_LONG_ERROR_TYPE for item in error.errors())


class CustomerClaimsRequest(BaseModel):
    """Framework-independent input for the customer claims application pipeline.

    The message is stripped and must then be 1..``MAX_CUSTOMER_QUERY_CHARS`` characters. The
    bound lives here, below any UI, so every caller (Streamlit, CLI, library) gets the same
    protection before retrieval or any safety rule runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_query: str

    @field_validator("customer_query")
    @classmethod
    def validate_customer_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("customer_query must not be empty or whitespace-only")
        if len(stripped) > MAX_CUSTOMER_QUERY_CHARS:
            raise PydanticCustomError(
                QUERY_TOO_LONG_ERROR_TYPE,
                "customer_query must not exceed {max_chars} characters",
                {"max_chars": MAX_CUSTOMER_QUERY_CHARS},
            )
        return stripped


class RetrievedItemMeta(BaseModel):
    """Display-safe metadata for a retrieved context item passed to the UI layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str
    rank: int
    heading: str
    document_id: str


class CustomerClaimsResult(BaseModel):
    """Framework-independent output from the customer claims application pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response: RiskAwareGroundedGenerationResult
    customer_query: str = ""
    retrieved_items: tuple[RetrievedItemMeta, ...] = ()
    # True when retrieval failed: the result then carries the deterministic safety assessment
    # and a degraded customer response only, with no retrieved context and no grounded answer.
    retrieval_failed: bool = False
    # True when retrieval succeeded but building the generation context from its results failed.
    # Same degraded shape as above; the two failure sources are mutually exclusive.
    context_build_failed: bool = False

    @model_validator(mode="after")
    def validate_single_failure_source(self) -> Self:
        if self.retrieval_failed and self.context_build_failed:
            raise ValueError("retrieval_failed and context_build_failed are mutually exclusive")
        return self
