"""Prompt construction for grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from customer_claims_rag.generation.models import ContextItem, GroundedGenerationRequest

_CONTEXT_BLOCK_BEGIN = "<<<CONTEXT_BLOCK_BEGIN>>>"
_CONTEXT_BLOCK_END = "<<<CONTEXT_BLOCK_END>>>"
_CUSTOMER_QUERY_BEGIN = "<<<CUSTOMER_QUERY_BEGIN>>>"
_CUSTOMER_QUERY_END = "<<<CUSTOMER_QUERY_END>>>"


@dataclass(frozen=True)
class GroundedPromptMessages:
    """System and user messages for grounded generation."""

    system_message: str
    user_message: str


class PromptBuilder:
    """Build deterministic grounded-generation prompts from a request."""

    def __init__(self, *, prompt_path: Path) -> None:
        """Load the system prompt from an explicit filesystem path."""
        resolved = Path(prompt_path)
        if not resolved.exists():
            raise FileNotFoundError(f"prompt file not found: {resolved}")
        if not resolved.is_file():
            raise IsADirectoryError(f"prompt path is not a file: {resolved}")
        self._prompt_path = resolved
        self._system_message = resolved.read_text(encoding="utf-8").strip()

    @property
    def system_message(self) -> str:
        return self._system_message

    def build_messages(self, request: GroundedGenerationRequest) -> GroundedPromptMessages:
        """Render system and user messages for the request."""
        return GroundedPromptMessages(
            system_message=self._system_message,
            user_message=self._render_user_message(request),
        )

    def _render_user_message(self, request: GroundedGenerationRequest) -> str:
        blocks = [
            self._render_context_block(item)
            for item in request.context_package.items
        ]
        sections = ["## Retrieved context", *blocks, self._render_customer_query(request.customer_query)]
        return "\n\n".join(section for section in sections if section).strip()

    def _render_context_block(self, item: ContextItem) -> str:
        escaped_content = escape(item.content, quote=False)
        return "\n".join(
            [
                _CONTEXT_BLOCK_BEGIN,
                f"citation_key: {item.citation_key}",
                f"rank: {item.rank}",
                f"document_id: {escape(item.document_id, quote=False)}",
                f"chunk_id: {escape(item.chunk_id, quote=False)}",
                f"heading: {escape(item.heading, quote=False)}",
                f"source_path: {escape(item.source_path, quote=False)}",
                "content:",
                escaped_content,
                _CONTEXT_BLOCK_END,
            ]
        )

    @staticmethod
    def _render_customer_query(customer_query: str) -> str:
        return "\n".join(
            [
                "## Customer query",
                _CUSTOMER_QUERY_BEGIN,
                escape(customer_query, quote=False),
                _CUSTOMER_QUERY_END,
            ]
        )
