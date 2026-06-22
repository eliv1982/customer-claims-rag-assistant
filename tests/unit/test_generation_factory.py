"""Unit tests for generation factory wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.generation.adapters.openai_chat import OpenAIChatAdapter
from customer_claims_rag.generation.factory import build_chat_model, build_grounded_generator
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation_config import GenerationSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "grounded_answer_v1.md"


def _settings() -> GenerationSettings:
    return GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=1024,
        prompt_path=PROMPT_PATH.resolve(),
    )


def test_build_chat_model_creates_chat_openai_with_expected_params() -> None:
    settings = _settings()
    fake_client = MagicMock()

    with patch(
        "customer_claims_rag.generation.factory.ChatOpenAI",
        return_value=fake_client,
    ) as chat_openai_cls:
        adapter = build_chat_model(settings)

    chat_openai_cls.assert_called_once_with(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60.0,
        max_retries=2,
        max_tokens=1024,
    )
    assert isinstance(adapter, OpenAIChatAdapter)
    assert adapter._client is fake_client


def test_build_grounded_generator_wires_dependencies() -> None:
    settings = _settings()
    fake_client = MagicMock()

    with patch(
        "customer_claims_rag.generation.factory.ChatOpenAI",
        return_value=fake_client,
    ):
        generator = build_grounded_generator(settings)

    assert isinstance(generator, GroundedGenerator)
    assert isinstance(generator._chat_model, OpenAIChatAdapter)
    assert generator._chat_model._client is fake_client
    assert isinstance(generator._prompt_builder, PromptBuilder)
    assert generator._prompt_builder._prompt_path == PROMPT_PATH.resolve()


def test_factory_does_not_invoke_client() -> None:
    settings = _settings()
    fake_client = MagicMock()

    with patch(
        "customer_claims_rag.generation.factory.ChatOpenAI",
        return_value=fake_client,
    ):
        build_grounded_generator(settings)

    fake_client.invoke.assert_not_called()
