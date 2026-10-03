"""End-to-end customer claims application orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from customer_claims_rag.application.models import (
    CustomerClaimsRequest,
    CustomerClaimsResult,
    RetrievedItemMeta,
)
from customer_claims_rag.application.ports import RetrievalPort, RiskAwareGenerationPort
from customer_claims_rag.exceptions import GenerationValidationError, RetrievalError
from customer_claims_rag.generation.context_builder import build_context_package
from customer_claims_rag.generation.fallback import generation_failure_result
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import ContextPackage, GroundedGenerationRequest
from customer_claims_rag.generation.risk_aware_generator import RiskAssessor
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskAssessmentRequest

logger = logging.getLogger(__name__)

ContextBuilder = Callable[[Sequence[SearchResult]], ContextPackage]


class CustomerClaimsPipeline:
    """Framework-independent application pipeline for customer claims handling."""

    def __init__(
        self,
        retrieval: RetrievalPort,
        generator: RiskAwareGenerationPort,
        context_builder: ContextBuilder = build_context_package,
        risk_assessor: RiskAssessor = assess_deterministic_risk,
    ) -> None:
        self._retrieval = retrieval
        self._generator = generator
        self._context_builder = context_builder
        self._risk_assessor = risk_assessor

    def handle(self, request: CustomerClaimsRequest) -> CustomerClaimsResult:
        # 1. Input: the request model has already stripped and bounded the message.
        query = request.customer_query

        # 2. Deterministic safety assessment: exactly one authoritative result per request,
        #    computed before any embedding / vector work so that no later failure can erase it.
        risk_assessment = self._risk_assessor(RiskAssessmentRequest(customer_query=query))

        # 3. Retrieval. A retrieval failure keeps the safety result and degrades the answer.
        try:
            search_results = self._retrieval.search(query)
        except RetrievalError as exc:
            logger.warning(
                "Retrieval failed with %s: %s",
                type(exc).__name__,
                str(exc),
            )
            return self._degraded_result(query, risk_assessment, retrieval_failed=True)

        # 3b. Context building. Only the builder's own typed data-validation error degrades the
        #     answer (e.g. duplicate rank / chunk_id in the retrieved results); any other
        #     exception is a programming error and propagates.
        try:
            context_package = self._context_builder(search_results)
        except GenerationValidationError as exc:
            logger.warning(
                "Context building failed with %s: %s",
                type(exc).__name__,
                str(exc),
            )
            return self._degraded_result(query, risk_assessment, context_build_failed=True)

        generation_request = GroundedGenerationRequest(
            customer_query=query,
            context_package=context_package,
        )

        # 4. Grounded generation, reusing the assessment computed above. The generator keeps
        #    the assessment (and its handoff) even when generation itself fails.
        generation_result = self._generator.generate(generation_request, risk_assessment)

        retrieved_items = tuple(
            RetrievedItemMeta(
                citation_key=item.citation_key,
                rank=item.rank,
                heading=item.heading,
                document_id=item.document_id,
            )
            for item in context_package.items
        )

        return CustomerClaimsResult(
            response=generation_result,
            customer_query=query,
            retrieved_items=retrieved_items,
        )

    @staticmethod
    def _degraded_result(
        query: str,
        risk_assessment: DeterministicRiskResult,
        *,
        retrieval_failed: bool = False,
        context_build_failed: bool = False,
    ) -> CustomerClaimsResult:
        """Degraded result built from the safety assessment alone (no context, no answer)."""
        return CustomerClaimsResult(
            response=RiskAwareGroundedGenerationResult(
                generation=generation_failure_result(),
                risk_assessment=risk_assessment,
                handoff_notice=build_handoff_notice(risk_assessment),
                generation_outcome="generation_error_fallback",
            ),
            customer_query=query,
            retrieved_items=(),
            retrieval_failed=retrieval_failed,
            context_build_failed=context_build_failed,
        )
