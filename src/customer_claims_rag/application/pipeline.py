"""End-to-end customer claims application orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.application.ports import RetrievalPort, RiskAwareGenerationPort
from customer_claims_rag.generation.context_builder import build_context_package
from customer_claims_rag.generation.models import ContextPackage, GroundedGenerationRequest
from customer_claims_rag.retrieval.models import SearchResult

ContextBuilder = Callable[[Sequence[SearchResult]], ContextPackage]


class CustomerClaimsPipeline:
    """Framework-independent application pipeline for customer claims handling."""

    def __init__(
        self,
        retrieval: RetrievalPort,
        generator: RiskAwareGenerationPort,
        context_builder: ContextBuilder = build_context_package,
    ) -> None:
        self._retrieval = retrieval
        self._generator = generator
        self._context_builder = context_builder

    def handle(self, request: CustomerClaimsRequest) -> CustomerClaimsResult:
        query = request.customer_query

        search_results = self._retrieval.search(query)

        context_package = self._context_builder(search_results)

        generation_request = GroundedGenerationRequest(
            customer_query=query,
            context_package=context_package,
        )

        generation_result = self._generator.generate(generation_request)

        return CustomerClaimsResult(response=generation_result)
