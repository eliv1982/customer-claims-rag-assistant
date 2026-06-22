"""Factory helpers for grounded generation wiring."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from customer_claims_rag.generation.adapters.openai_chat import OpenAIChatAdapter
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.ports import ChatModel
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation_config import GenerationSettings


def build_chat_model(settings: GenerationSettings) -> ChatModel:
    client = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        max_tokens=settings.max_output_tokens,
    )
    return OpenAIChatAdapter(client=client)


def build_grounded_generator(settings: GenerationSettings) -> GroundedGenerator:
    return GroundedGenerator(
        chat_model=build_chat_model(settings),
        prompt_builder=PromptBuilder(prompt_path=settings.prompt_path),
    )
