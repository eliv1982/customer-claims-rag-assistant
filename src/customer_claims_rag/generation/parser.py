"""Strict JSON parsing for grounded generation model output."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import ValidationError

from customer_claims_rag.exceptions import GenerationParseError
from customer_claims_rag.generation.models import RawGenerationDraft

_MARKDOWN_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\r?\n(.*?)\r?\n```\s*$",
    re.DOTALL,
)


def _strip_markdown_fence(text: str) -> str:
    """Strip optional markdown code fences from model output."""
    m = _MARKDOWN_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise GenerationParseError(f"duplicate JSON key: {key}")
        seen.add(key)
        result[key] = value
    return result


def parse_generation_draft(raw: str) -> RawGenerationDraft:
    """Parse raw model output into a generation draft."""
    if not isinstance(raw, str):
        raise GenerationParseError("model output must be a string")

    text = raw.strip()
    if not text:
        raise GenerationParseError("model output is empty")

    text = _strip_markdown_fence(text)

    try:
        decoded = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except GenerationParseError:
        raise
    except json.JSONDecodeError as exc:
        raise GenerationParseError("model output is not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise GenerationParseError("model output must be a JSON object")

    try:
        return RawGenerationDraft.model_validate(decoded, strict=True)
    except ValidationError as exc:
        raise GenerationParseError("model output does not match generation schema") from exc
