"""Strict JSON parsing for grounded generation model output."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import ValidationError

from customer_claims_rag.exceptions import GenerationParseError
from customer_claims_rag.generation.models import RawGenerationDraft


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
