"""Integration test for grounded generation pipeline without live OpenAI."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.models import (
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
)
from customer_claims_rag.generation.parser import parse_generation_draft
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.validator import validate_generation_draft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


def test_context_package_to_result_pipeline() -> None:
    package = ContextPackage(
        items=[
            ContextItem(
                citation_key="S1",
                rank=1,
                document_id="doc-a",
                chunk_id="doc-a::1",
                heading="Heading A",
                source_path="data/02_clean_markdown/doc-a.md",
                content="Policy text",
            )
        ]
    )
    request = GroundedGenerationRequest(
        customer_query="Когда придет заказ?",
        context_package=package,
    )
    prompt_builder = PromptBuilder(prompt_path=PROMPT_PATH)
    prompt_messages = prompt_builder.build_messages(request)
    assert prompt_messages.system_message
    assert "Когда придет заказ?" in prompt_messages.user_message

    raw = (
        '{"response_mode":"grounded_answer",'
        '"answer":"Срок зависит от правил доставки [S1]."}'
    )
    chat_model = FakeChatModel(response=raw)
    generator = GroundedGenerator(chat_model=chat_model, prompt_builder=prompt_builder)
    result = generator.generate(request)

    draft = parse_generation_draft(raw)
    expected = validate_generation_draft(draft, package)
    assert result == expected
    assert result.response_mode == "grounded_answer"
    assert "[S1]" in result.customer_response
    assert result.citations[0].document_id == "doc-a"
    assert chat_model.call_count == 1
