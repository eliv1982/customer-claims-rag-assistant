"""Answer a single customer claim via the production application pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from customer_claims_rag.application.factory import build_customer_claims_pipeline
from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import ApplicationSettings
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.exceptions import GenerationError, RetrievalError

_INPUT_ERROR_MESSAGE = "Error: message must not be empty or whitespace-only"
_RUNTIME_ERROR_MESSAGE = "Error: claim handling failed"
_CONFIGURATION_ERROR_MESSAGE = "Error: application configuration is invalid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a single customer claim using the production RAG pipeline.",
    )
    parser.add_argument(
        "--message",
        required=True,
        help="Customer claim message text",
    )
    return parser


def format_answer_payload(result: CustomerClaimsResult) -> dict[str, Any]:
    """Map a pipeline result to the stable CLI JSON schema."""
    response = result.response
    generation = response.generation
    risk = response.risk_assessment
    return {
        "answer": generation.customer_response,
        "response_mode": generation.response_mode,
        "generation_outcome": response.generation_outcome,
        "risk_level": risk.risk_floor.value,
        "handoff_required": risk.handoff_required,
        "priority_handoff": risk.priority_handoff,
        "handoff_notice": response.handoff_notice,
        "citations": [
            {
                "key": citation.citation_key,
                "heading": citation.heading,
                "document_id": citation.document_id,
            }
            for citation in generation.citations
        ],
    }


def serialize_answer_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def run_answer(
    *,
    message: str,
    settings_loader: Callable[[], ApplicationSettings] | None = None,
    pipeline_factory: Callable[
        [ApplicationSettings],
        CustomerClaimsPipeline,
    ] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    resolve_settings_loader = (
        settings_loader if settings_loader is not None else ApplicationSettings.from_env
    )
    resolve_pipeline_factory = (
        pipeline_factory if pipeline_factory is not None else build_customer_claims_pipeline
    )
    try:
        request = CustomerClaimsRequest(customer_query=message)
    except ValidationError:
        print(_INPUT_ERROR_MESSAGE, file=sys.stderr)
        return 2, None

    try:
        settings = resolve_settings_loader()
        pipeline = resolve_pipeline_factory(settings)
        result = pipeline.handle(request)
    except RetrievalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except ValidationError:
        print(_CONFIGURATION_ERROR_MESSAGE, file=sys.stderr)
        return 1, None
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except Exception:
        print(_RUNTIME_ERROR_MESSAGE, file=sys.stderr)
        return 1, None

    payload = format_answer_payload(result)
    print(serialize_answer_payload(payload))
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _ = run_answer(message=args.message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
