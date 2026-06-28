"""Unit tests for generation settings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.exceptions import GenerationError
from customer_claims_rag.generation_config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
    DEFAULT_GENERATION_MAX_RETRIES,
    DEFAULT_GENERATION_PROMPT_PATH,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    GenerationSettings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


@pytest.fixture(autouse=True)
def reset_env_bootstrap() -> None:
    env_bootstrap.reset_project_env()
    yield
    env_bootstrap.reset_project_env()


def _clear_generation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPENAI_CHAT_MODEL",
        "GENERATION_TEMPERATURE",
        "GENERATION_TIMEOUT_SECONDS",
        "GENERATION_MAX_RETRIES",
        "GENERATION_MAX_OUTPUT_TOKENS",
        "GENERATION_PROMPT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)


def test_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    _clear_generation_env(monkeypatch)

    settings = GenerationSettings.from_env()

    assert settings.model_name == DEFAULT_CHAT_MODEL
    assert settings.temperature == DEFAULT_GENERATION_TEMPERATURE
    assert settings.timeout_seconds == DEFAULT_GENERATION_TIMEOUT_SECONDS
    assert settings.max_retries == DEFAULT_GENERATION_MAX_RETRIES
    assert settings.max_output_tokens == DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
    assert settings.prompt_path == DEFAULT_GENERATION_PROMPT_PATH.resolve()
    assert settings.prompt_path.is_absolute()
    assert settings.prompt_path == PROMPT_PATH.resolve()


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("GENERATION_TEMPERATURE", "0.5")
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "1")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv(
        "GENERATION_PROMPT_PATH",
        str(PROMPT_PATH.resolve()),
    )

    settings = GenerationSettings.from_env()

    assert settings.model_name == "gpt-4o"
    assert settings.temperature == 0.5
    assert settings.timeout_seconds == 45.0
    assert settings.max_retries == 1
    assert settings.max_output_tokens == 512
    assert settings.prompt_path == PROMPT_PATH.resolve()


def test_model_name_whitespace_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "  gpt-4o-mini  ")

    settings = GenerationSettings.from_env()

    assert settings.model_name == "gpt-4o-mini"


def test_empty_model_name_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "   ")

    with pytest.raises(GenerationError, match="model_name"):
        GenerationSettings.from_env()


@pytest.mark.parametrize("temperature", ["0.0", "2.0"])
def test_temperature_boundaries(monkeypatch: pytest.MonkeyPatch, temperature: str) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_TEMPERATURE", temperature)

    settings = GenerationSettings.from_env()

    assert settings.temperature == float(temperature)


def test_invalid_temperature_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_TEMPERATURE", "2.1")

    with pytest.raises(GenerationError, match="temperature"):
        GenerationSettings.from_env()


def test_invalid_timeout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "0")

    with pytest.raises(GenerationError, match="timeout_seconds"):
        GenerationSettings.from_env()


def test_invalid_max_retries_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "-1")

    with pytest.raises(GenerationError, match="max_retries"):
        GenerationSettings.from_env()


def test_invalid_max_output_tokens_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "0")

    with pytest.raises(GenerationError, match="max_output_tokens"):
        GenerationSettings.from_env()


def test_relative_prompt_path_resolved_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GENERATION_PROMPT_PATH", "prompts/system_prompt.md")

    settings = GenerationSettings.from_env()

    assert settings.prompt_path == PROMPT_PATH.resolve()


def test_absolute_prompt_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_PROMPT_PATH", str(PROMPT_PATH.resolve()))

    settings = GenerationSettings.from_env()

    assert settings.prompt_path == PROMPT_PATH.resolve()


def test_missing_prompt_file_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv(
        "GENERATION_PROMPT_PATH",
        str(tmp_path / "missing_prompt.md"),
    )

    with pytest.raises(GenerationError, match="prompt file not found"):
        GenerationSettings.from_env()


def test_directory_prompt_path_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_PROMPT_PATH", str(tmp_path))

    with pytest.raises(GenerationError, match="not a file"):
        GenerationSettings.from_env()


def test_openai_api_key_is_not_settings_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    settings = GenerationSettings.from_env()

    assert not hasattr(settings, "openai_api_key")
    assert "sk-test-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("GENERATION_TEMPERATURE", "nan"),
        ("GENERATION_TEMPERATURE", "inf"),
        ("GENERATION_TEMPERATURE", "-inf"),
        ("GENERATION_TIMEOUT_SECONDS", "nan"),
        ("GENERATION_TIMEOUT_SECONDS", "inf"),
        ("GENERATION_TIMEOUT_SECONDS", "-inf"),
    ],
)
def test_non_finite_float_env_rejected(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv(env_name, value)

    with pytest.raises(GenerationError, match="finite float"):
        GenerationSettings.from_env()


def test_temperature_true_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_TEMPERATURE", "True")

    with pytest.raises(GenerationError, match="must be a float"):
        GenerationSettings.from_env()


def test_timeout_true_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "True")

    with pytest.raises(GenerationError, match="must be a float"):
        GenerationSettings.from_env()


def test_max_retries_true_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "True")

    with pytest.raises(GenerationError, match="must be an integer"):
        GenerationSettings.from_env()


def test_max_output_tokens_true_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "True")

    with pytest.raises(GenerationError, match="must be an integer"):
        GenerationSettings.from_env()


@pytest.mark.parametrize(
    ("env_name", "value", "expected_type"),
    [
        ("GENERATION_TEMPERATURE", "abc", "float"),
        ("GENERATION_TIMEOUT_SECONDS", "1.2.3", "float"),
        ("GENERATION_MAX_RETRIES", "abc", "integer"),
        ("GENERATION_MAX_OUTPUT_TOKENS", "1.2.3", "integer"),
    ],
)
def test_malformed_numeric_env_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    value: str,
    expected_type: str,
) -> None:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    monkeypatch.setenv(env_name, value)

    with pytest.raises(GenerationError, match=expected_type) as exc_info:
        GenerationSettings.from_env()

    assert env_name in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_direct_construction_rejects_non_finite_temperature() -> None:
    settings = GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=float("nan"),
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=1024,
        prompt_path=PROMPT_PATH.resolve(),
    )

    with pytest.raises(GenerationError, match="finite float"):
        settings.validate()


def test_direct_construction_rejects_bool_max_retries() -> None:
    settings = GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout_seconds=60.0,
        max_retries=True,  # type: ignore[arg-type]
        max_output_tokens=1024,
        prompt_path=PROMPT_PATH.resolve(),
    )

    with pytest.raises(GenerationError, match="must be an integer"):
        settings.validate()
