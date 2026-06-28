"""Application-layer request and result contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)


class CustomerClaimsRequest(BaseModel):
    """Framework-independent input for the customer claims application pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_query: str

    @field_validator("customer_query")
    @classmethod
    def validate_customer_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("customer_query must not be empty or whitespace-only")
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
