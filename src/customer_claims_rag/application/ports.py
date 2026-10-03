"""Application-layer port interfaces."""

from __future__ import annotations

from typing import Protocol, Sequence

from customer_claims_rag.generation.models import GroundedGenerationRequest
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.risk.models import DeterministicRiskResult


class RetrievalPort(Protocol):
    """Final ordered retrieval results for a normalized customer query."""

    def search(self, query: str) -> Sequence[SearchResult]:
        """Return final ordered retrieval results for the normalized query."""
        ...


class RiskAwareGenerationPort(Protocol):
    """Risk-aware grounded generation contract at the application boundary.

    The pipeline computes the deterministic risk assessment once per request, before
    retrieval, and hands that single authoritative result to the generator.
    """

    def generate(
        self,
        request: GroundedGenerationRequest,
        risk_assessment: DeterministicRiskResult,
    ) -> RiskAwareGroundedGenerationResult:
        ...
