"""Generation configuration with env overrides."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.env_bootstrap import load_project_env, project_root
from customer_claims_rag.exceptions import GenerationError

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_TIMEOUT_SECONDS = 60.0
DEFAULT_GENERATION_MAX_RETRIES = 2
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 1024
DEFAULT_GENERATION_PROMPT_PATH = project_root() / "prompts" / "grounded_answer_v1.md"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise GenerationError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise GenerationError(f"{name} must be a float") from exc


def _resolve_prompt_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _validate_prompt_path(path: Path) -> Path:
    if not path.exists():
        raise GenerationError(f"generation prompt file not found: {path}")
    if not path.is_file():
        raise GenerationError(f"generation prompt path is not a file: {path}")
    return path


def _env_model_name(default: str) -> str:
    value = os.environ.get("OPENAI_CHAT_MODEL")
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        raise GenerationError("model_name must not be empty")
    return stripped


def _validate_finite_float(
    field_name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    must_be_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationError(f"{field_name} must be a finite float")

    numeric = float(value)
    if not math.isfinite(numeric):
        raise GenerationError(f"{field_name} must be a finite float")

    if must_be_positive and numeric <= 0:
        raise GenerationError(f"{field_name} must be > 0")

    if minimum is not None and numeric < minimum:
        raise GenerationError(f"{field_name} must be in [{minimum}, {maximum}]")

    if maximum is not None and numeric > maximum:
        raise GenerationError(f"{field_name} must be in [{minimum}, {maximum}]")

    return numeric


def _validate_exact_int(
    field_name: str,
    value: object,
    *,
    minimum: int,
) -> int:
    if type(value) is not int:
        raise GenerationError(f"{field_name} must be an integer")

    if value < minimum:
        raise GenerationError(f"{field_name} must be >= {minimum}")

    return value


@dataclass(frozen=True)
class GenerationSettings:
    """Validated settings for grounded answer generation."""

    model_name: str
    temperature: float
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int
    prompt_path: Path

    @classmethod
    def from_env(cls) -> GenerationSettings:
        load_project_env()
        prompt_env = os.environ.get("GENERATION_PROMPT_PATH")
        if prompt_env is None or not prompt_env.strip():
            prompt_path = DEFAULT_GENERATION_PROMPT_PATH.resolve()
        else:
            prompt_path = _resolve_prompt_path(prompt_env.strip())

        settings = cls(
            model_name=_env_model_name(DEFAULT_CHAT_MODEL),
            temperature=_env_float("GENERATION_TEMPERATURE", DEFAULT_GENERATION_TEMPERATURE),
            timeout_seconds=_env_float(
                "GENERATION_TIMEOUT_SECONDS",
                DEFAULT_GENERATION_TIMEOUT_SECONDS,
            ),
            max_retries=_env_int("GENERATION_MAX_RETRIES", DEFAULT_GENERATION_MAX_RETRIES),
            max_output_tokens=_env_int(
                "GENERATION_MAX_OUTPUT_TOKENS",
                DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
            ),
            prompt_path=prompt_path,
        )
        return settings.validate()

    def validate(self) -> GenerationSettings:
        model_name = self.model_name.strip()
        if not model_name:
            raise GenerationError("model_name must not be empty")

        temperature = _validate_finite_float(
            "temperature",
            self.temperature,
            minimum=0.0,
            maximum=2.0,
        )
        timeout_seconds = _validate_finite_float(
            "timeout_seconds",
            self.timeout_seconds,
            must_be_positive=True,
        )
        max_retries = _validate_exact_int("max_retries", self.max_retries, minimum=0)
        max_output_tokens = _validate_exact_int(
            "max_output_tokens",
            self.max_output_tokens,
            minimum=1,
        )
        prompt_path = _validate_prompt_path(self.prompt_path.resolve())

        return GenerationSettings(
            model_name=model_name,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_output_tokens=max_output_tokens,
            prompt_path=prompt_path,
        )
