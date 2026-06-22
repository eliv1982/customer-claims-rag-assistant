"""Grounded answer generation orchestration."""

from __future__ import annotations

from customer_claims_rag.generation.models import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from customer_claims_rag.generation.parser import parse_generation_draft
from customer_claims_rag.generation.ports import ChatMessage, ChatModel
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.validator import (
    insufficient_context_for_empty_package,
    validate_generation_draft,
)


class GroundedGenerator:
    """Orchestrate prompt build, chat completion, parse, and validation."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._chat_model = chat_model
        self._prompt_builder = prompt_builder

    def generate(self, request: GroundedGenerationRequest) -> GroundedGenerationResult:
        if not request.context_package.items:
            return insufficient_context_for_empty_package()

        prompt_messages = self._prompt_builder.build_messages(request)
        chat_messages = [
            ChatMessage(role="system", content=prompt_messages.system_message),
            ChatMessage(role="user", content=prompt_messages.user_message),
        ]
        raw_response = self._chat_model.complete(chat_messages)
        draft = parse_generation_draft(raw_response)
        return validate_generation_draft(draft, request.context_package)
